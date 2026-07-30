"""
Program Database for storing and managing generated programs during rollout.
Uses SQLite for persistent storage with thread-safety.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import sqlite3
import json
import threading
import os
from pathlib import Path
import random
import logging
import math
import sys
import statistics
import statistics
from airaevo_experience import (
    build_operator_experience_memory,
    compute_parent_utilities,
    is_debug_candidate,
    is_success_program,
    program_selection_score,
)

logger = logging.getLogger("mle_agent")


class DatabaseLifecycle(Enum):
    """Lifecycle modes for Program Database."""

    ROLLOUT = "rollout"  # One database per rollout
    STEP = "step"  # One database per step
    TASK = "task"  # One database per task


@dataclass
class Program:
    """Represents a program in the database."""

    id: Optional[int] = None  # Database ID
    task_id: str = ""  # Task identifier (uuid)
    task_name: str = ""  # Task name
    status_code: int = 0  # Status code from sandbox execution
    payload: dict = field(default_factory=dict)  # Payload from sandbox execution
    code: str = ""  # The code text
    score: Optional[float] = None  # Score from sandbox execution (None = no score, different from 0.0)
    running_time: float = 0.0  # Total observed sandbox running time in seconds
    reward: float = 0.0  # Reward calculated from score (base_reward)
    fitness: float | None = None  # Fitness used by search selection
    base_reward: float = 0.0  # Base reward before considering parent
    exploit_coefficient: float = 0.5  # Normalized coefficient from base_reward
    explore_coefficient: float = 0.5  # Normalized coefficient from child_base_reward_variance
    cooling_coefficient: float = 1.0  # Cooling value derived from visit_count
    visit_count: int = 0  # Number of times this node is selected as parent
    child_base_reward_variance: float | None = None  # Variance of direct child base_reward values (None=no child)
    parent_id: Optional[int] = None  # Parent program ID (for improve mode)
    parent_code: str = ""  # Parent program code (for logging)
    generation_mode: str = "draft"  # Generation mode: 'draft' | 'improve' | 'debug' | 'crossover'
    raw_text: str = ""  # Raw response text from model
    hack: int = 0  # Hack check result: 0 = valid ML code, 1 = cheating/guessing
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def __post_init__(self) -> None:
        if self.fitness is None:
            # Fitness is recomputed at task level from multiple components after insertion.
            self.fitness = 0.0
        clear_run_log = self.metadata.get("clear_run_log")
        if clear_run_log is not None:
            self.run_log = str(clear_run_log)

    def to_dict(self) -> Dict[str, Any]:
        """Convert program to dictionary for storage."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "status_code": self.status_code,
            "payload": self.payload,
            "code": self.code,
            "score": self.score,
            "running_time": self.running_time,
            "reward": self.reward,
            "fitness": self.fitness,
            "base_reward": self.base_reward,
            "exploit_coefficient": self.exploit_coefficient,
            "explore_coefficient": self.explore_coefficient,
            "cooling_coefficient": self.cooling_coefficient,
            "visit_count": self.visit_count,
            "child_base_reward_variance": self.child_base_reward_variance,
            "parent_id": self.parent_id,
            "parent_code": self.parent_code,
            "generation_mode": self.generation_mode,
            "raw_text": self.raw_text,
            "hack": self.hack,
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        """Create program from dictionary."""
        metadata = (
            json.loads(data.get("metadata", "{}"))
            if isinstance(data.get("metadata"), str)
            else data.get("metadata", {})
        )
        payload = (
            json.loads(data.get("payload", "{}")) if isinstance(data.get("payload"), str) else data.get("payload", {})
        )
        return cls(
            id=data.get("id"),
            task_id=data.get("task_id", ""),
            task_name=data.get("task_name", ""),
            status_code=data.get("status_code", 0),
            payload=payload,
            code=data.get("code", ""),
            score=data.get("score"),  # Allow None to distinguish from 0
            running_time=data.get("running_time", 0.0),
            reward=data.get("reward", 0.0),
            fitness=data.get("fitness", 0.0),
            base_reward=data.get("base_reward", 0.0),
            exploit_coefficient=data.get("exploit_coefficient", 0.5),
            explore_coefficient=data.get("explore_coefficient", 0.5),
            visit_count=data.get("visit_count", 0),
            child_base_reward_variance=data.get("child_base_reward_variance"),
            cooling_coefficient=data.get("cooling_coefficient", 1.0),
            parent_id=data.get("parent_id"),
            parent_code=data.get("parent_code", ""),
            generation_mode=data.get("generation_mode", "draft"),
            raw_text=data.get("raw_text", ""),
            hack=data.get("hack", 0),  # Default to 0 (not hacking, valid) for backward compatibility
            metadata=metadata,
        )


class ProgramDatabase:
    """Database for storing and managing programs during rollout.

    Uses SQLite for persistent storage with connection pooling and thread-safety.
    Each task maintains its top-k programs by reward.
    """

    _local = threading.local()
    _lock = threading.Lock()
    _SQLITE_INT_MIN = -(2**63)
    _SQLITE_INT_MAX = 2**63 - 1
    _SQLITE_FLOAT_MAX = sys.float_info.max

    def __init__(self, db_path: str = "program_database.db", max_per_task: int = 10):
        """
        Initialize Program Database.

        Args:
            db_path: Path to SQLite database file
            max_per_task: Maximum number of programs to keep per task (top k by reward)
        """
        self.db_path = db_path
        self.max_per_task = max_per_task
        self._child_variance_nullable = True

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) if os.path.dirname(db_path) else ".", exist_ok=True)

        # Initialize database schema
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS programs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                task_name TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                payload TEXT NOT NULL,
                code TEXT NOT NULL,
                score REAL,
                running_time REAL NOT NULL DEFAULT 0.0,
                reward REAL NOT NULL,
                fitness REAL NOT NULL,
                base_reward REAL NOT NULL,
                exploit_coefficient REAL NOT NULL DEFAULT 0.5,
                explore_coefficient REAL NOT NULL DEFAULT 0.5,
                cooling_coefficient REAL NOT NULL DEFAULT 1.0,
                visit_count INTEGER NOT NULL DEFAULT 0,
                child_base_reward_variance REAL,
                parent_id INTEGER,
                parent_code TEXT,
                generation_mode TEXT DEFAULT 'draft',
                raw_text TEXT,
                hack INTEGER DEFAULT 0,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_id) REFERENCES programs(id)
            )
        """
        )

        cursor.execute("PRAGMA table_info(programs)")  # Read table schema metadata.
        table_info = cursor.fetchall()
        columns = {row["name"] for row in table_info}  # Collect column names.

        if "fitness" not in columns:  # Add fitness column when missing.
            cursor.execute("ALTER TABLE programs ADD COLUMN fitness REAL NOT NULL DEFAULT 0.0")  # Add fitness column.
            cursor.execute("UPDATE programs SET fitness = reward")  # Backfill fitness from reward.

        if "hack" not in columns:  # Add hack column when missing.
            cursor.execute("ALTER TABLE programs ADD COLUMN hack INTEGER DEFAULT 0")  # Add hack column.
            cursor.execute("UPDATE programs SET hack = 0")  # Backfill hack as 0 (valid) for existing programs.
        if "running_time" not in columns:
            cursor.execute("ALTER TABLE programs ADD COLUMN running_time REAL NOT NULL DEFAULT 0.0")
            cursor.execute("UPDATE programs SET running_time = 0.0")
        if "exploit_coefficient" not in columns:
            cursor.execute("ALTER TABLE programs ADD COLUMN exploit_coefficient REAL NOT NULL DEFAULT 0.5")
            cursor.execute("UPDATE programs SET exploit_coefficient = 0.5")
        if "explore_coefficient" not in columns:
            cursor.execute("ALTER TABLE programs ADD COLUMN explore_coefficient REAL NOT NULL DEFAULT 0.5")
            cursor.execute("UPDATE programs SET explore_coefficient = 0.5")
        if "visit_count" not in columns:
            cursor.execute("ALTER TABLE programs ADD COLUMN visit_count INTEGER NOT NULL DEFAULT 0")
            cursor.execute("UPDATE programs SET visit_count = 0")
        if "child_base_reward_variance" not in columns:
            cursor.execute("ALTER TABLE programs ADD COLUMN child_base_reward_variance REAL")
        if "cooling_coefficient" not in columns:
            cursor.execute("ALTER TABLE programs ADD COLUMN cooling_coefficient REAL NOT NULL DEFAULT 1.0")
            cursor.execute("UPDATE programs SET cooling_coefficient = 1.0")

        # Backward compatibility: old DB may enforce NOT NULL on child_base_reward_variance.
        cursor.execute("PRAGMA table_info(programs)")
        table_info = cursor.fetchall()
        info_by_name = {row["name"]: row for row in table_info}
        cbrv = info_by_name.get("child_base_reward_variance")
        self._child_variance_nullable = bool(cbrv is None or int(cbrv["notnull"]) == 0)

        # Create index for faster task-based queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_name ON programs(task_name)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_id ON programs(task_id)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_reward ON programs(task_name, reward DESC)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_id_reward ON programs(task_id, reward DESC)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_fitness ON programs(task_name, fitness DESC)
        """
        )

        conn.commit()

    def get_task_scores(self, *, task_id: str = "", task_name: str = "") -> list[float]:
        """Return all finite stored scores for a task."""
        if not task_id and not task_name:
            return []
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            if task_id:
                cursor.execute(
                    """
                    SELECT score
                    FROM programs
                    WHERE task_id = ? AND score IS NOT NULL
                    ORDER BY id ASC
                """,
                    (task_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT score
                    FROM programs
                    WHERE task_name = ? AND score IS NOT NULL
                    ORDER BY id ASC
                """,
                    (task_name,),
                )
            scores: list[float] = []
            for row in cursor.fetchall():
                try:
                    score = float(row["score"])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(score):
                    scores.append(score)
            return scores

    def merge_program_metadata(self, program_id: int, metadata_updates: Dict[str, Any]) -> bool:
        """Merge metadata fields into an existing program row."""
        if not metadata_updates:
            return False
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT metadata
                FROM programs
                WHERE id = ?
            """,
                (int(program_id),),
            )
            row = cursor.fetchone()
            if row is None:
                return False
            try:
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            except Exception:
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            metadata.update(metadata_updates)
            cursor.execute(
                """
                UPDATE programs
                SET metadata = ?
                WHERE id = ?
            """,
                (json.dumps(metadata), int(program_id)),
            )
            conn.commit()
            return True

    def add(self, program: Program) -> int:
        """
        Add a program to the database, maintaining top k per task by reward.

        Returns:
            The database ID of the inserted program
        """

        def _to_sqlite_int(value: Any, *, default: int | None = None) -> int | None:
            if value is None:
                return default
            try:
                iv = int(value)
            except (TypeError, ValueError):
                return default
            if iv > self._SQLITE_INT_MAX:
                logger.warning("Clamping int value %s to SQLite max", iv)
                return self._SQLITE_INT_MAX
            if iv < self._SQLITE_INT_MIN:
                logger.warning("Clamping int value %s to SQLite min", iv)
                return self._SQLITE_INT_MIN
            return iv

        def _to_sqlite_real(value: Any, *, default: float | None = None) -> float | None:
            if value is None:
                return default
            try:
                fv = float(value)
            except (TypeError, ValueError):
                return default
            if not math.isfinite(fv):
                if fv == 0.0:
                    return 0.0
                logger.warning("Clamping non-finite float value %s to max", fv)
                return math.copysign(self._SQLITE_FLOAT_MAX, fv)
            return fv

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            status_code = _to_sqlite_int(program.status_code, default=0)
            parent_id = _to_sqlite_int(program.parent_id, default=None)
            hack_value = _to_sqlite_int(program.hack, default=0)
            score_value = _to_sqlite_real(program.score, default=None)
            running_time_value = _to_sqlite_real(program.running_time, default=0.0) or 0.0
            reward_value = _to_sqlite_real(program.reward, default=0.0)
            # Persist a placeholder value; task-level recomputation immediately updates fitness.
            fitness_value = _to_sqlite_real(program.fitness, default=0.0)
            base_reward_value = _to_sqlite_real(program.base_reward, default=0.0)
            exploit_coef_value = _to_sqlite_real(program.exploit_coefficient, default=0.5)
            explore_coef_value = _to_sqlite_real(program.explore_coefficient, default=0.5)
            visit_count_value = _to_sqlite_int(program.visit_count, default=0) or 0
            child_variance_value = _to_sqlite_real(
                program.child_base_reward_variance,
                default=None if self._child_variance_nullable else 0.5,
            )
            cooling_value = _to_sqlite_real(program.cooling_coefficient, default=1.0)

            # Insert the program
            cursor.execute(
                """
                INSERT INTO programs (task_id, task_name, status_code, payload, code, score, running_time, reward, fitness, base_reward,
                                     exploit_coefficient, explore_coefficient, cooling_coefficient,
                                     visit_count, child_base_reward_variance,
                                     parent_id, parent_code, generation_mode, raw_text, hack, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    program.task_id,
                    program.task_name,
                    status_code,
                    json.dumps(program.payload),  # Convert dict to JSON string
                    program.code,
                    score_value,
                    running_time_value,
                    reward_value,
                    fitness_value,
                    base_reward_value,
                    exploit_coef_value,
                    explore_coef_value,
                    cooling_value,
                    visit_count_value,
                    child_variance_value,
                    parent_id,
                    program.parent_code,
                    program.generation_mode,
                    program.raw_text,
                    hack_value,
                    json.dumps(program.metadata),
                ),
            )
            program_id = cursor.lastrowid

            # Recompute task fitness after each insertion.
            self._recompute_task_fitness(program.task_name, conn, cursor)

            # Maintain top k programs per task when max_per_task > 0.
            cursor.execute(
                """
                SELECT COUNT(*) as count FROM programs WHERE task_name = ?
            """,
                (program.task_name,),
            )
            count = cursor.fetchone()["count"]

            if self.max_per_task > 0 and count > self.max_per_task:
                # Delete programs beyond top k
                cursor.execute(
                    """
                    DELETE FROM programs
                    WHERE task_name = ? AND id NOT IN (
                        SELECT id FROM programs
                        WHERE task_name = ?
                        ORDER BY fitness DESC
                        LIMIT ?
                    )
                """,
                    (program.task_name, program.task_name, self.max_per_task),
                )

            conn.commit()
            return program_id

    @staticmethod
    def _normalize_minmax(values: list[float], neutral: float = 0.5) -> list[float]:
        if not values:
            return []
        lo = min(values)
        hi = max(values)
        if abs(hi - lo) < 1e-12:
            return [neutral for _ in values]
        denom = hi - lo
        return [(v - lo) / denom for v in values]

    @staticmethod
    def _normalize_minmax_optional(values: list[float | None], neutral: float = 0.5) -> list[float]:
        if not values:
            return []
        finite_idx_vals = [(i, v) for i, v in enumerate(values) if v is not None]
        if not finite_idx_vals:
            return [neutral for _ in values]
        finite_vals = [float(v) for _, v in finite_idx_vals]
        lo = min(finite_vals)
        hi = max(finite_vals)
        out = [neutral for _ in values]
        if abs(hi - lo) < 1e-12:
            for i, _ in finite_idx_vals:
                out[i] = neutral
            return out
        denom = hi - lo
        for i, v in finite_idx_vals:
            out[i] = (float(v) - lo) / denom
        return out

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            fv = float(v)
            if math.isfinite(fv):
                return fv
        except Exception:
            pass
        return default

    def _extract_parent_ids_from_metadata(self, metadata: dict[str, Any]) -> list[int]:
        parent_ids: list[int] = []
        maybe_ids = metadata.get("crossover_parent_ids")
        if isinstance(maybe_ids, list):
            for pid in maybe_ids:
                try:
                    pid_i = int(pid)
                    parent_ids.append(pid_i)
                except Exception:
                    continue
        return parent_ids

    def _recompute_task_fitness(self, task_name: str, conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """
            SELECT * FROM programs
            WHERE task_name = ?
            """,
            (task_name,),
        )
        rows = cursor.fetchall()
        if not rows:
            return

        programs = [Program.from_dict(dict(r)) for r in rows]
        by_id = {p.id: p for p in programs if p.id is not None}
        if not by_id:
            return

        # 1) Utilization value: node base_reward (higher is better).
        utilization_raw = {pid: self._safe_float(p.base_reward, default=0.0) for pid, p in by_id.items()}

        # 2) Learning value: variance of child base_reward (higher is better), default 0.5 when no children.
        child_rewards: dict[int, list[float]] = {pid: [] for pid in by_id}
        for child in programs:
            linked_parents: set[int] = set()
            if child.parent_id is not None and child.parent_id in by_id:
                linked_parents.add(int(child.parent_id))
            if isinstance(child.metadata, dict):
                for pid in self._extract_parent_ids_from_metadata(child.metadata):
                    if pid in by_id:
                        linked_parents.add(pid)
            for pid in linked_parents:
                child_rewards[pid].append(self._safe_float(child.base_reward, default=0.0))

        learning_raw: dict[int, float | None] = {}
        for pid, rewards in child_rewards.items():
            if len(rewards) == 0:
                learning_raw[pid] = None
            elif len(rewards) == 1:
                learning_raw[pid] = 0.0
            else:
                mean_reward = sum(rewards) / len(rewards)
                learning_raw[pid] = float(sum((x - mean_reward) ** 2 for x in rewards) / len(rewards))

        # 3) Cooling coefficient: fewer visits is better.
        # Visit count is parent selection count from child links.
        visit_counts: dict[int, int] = {pid: 0 for pid in by_id}
        for child in programs:
            linked_parents: set[int] = set()
            if child.parent_id is not None and child.parent_id in by_id:
                linked_parents.add(int(child.parent_id))
            if isinstance(child.metadata, dict):
                for pid in self._extract_parent_ids_from_metadata(child.metadata):
                    if pid in by_id:
                        linked_parents.add(pid)
            for pid in linked_parents:
                visit_counts[pid] += 1

        max_visit = max(visit_counts.values()) if visit_counts else 0
        cooling_raw: dict[int, float] = {}
        for pid, count in visit_counts.items():
            if max_visit <= 0:
                cooling_raw[pid] = 1.0
            else:
                cooling_raw[pid] = max(0.0, 1.0 - (float(count) / float(max_visit)))

        ordered_ids = list(by_id.keys())
        exploit_coef = self._normalize_minmax([utilization_raw[pid] for pid in ordered_ids], neutral=0.5)
        explore_coef = self._normalize_minmax_optional([learning_raw[pid] for pid in ordered_ids], neutral=0.5)
        cool_norm = self._normalize_minmax([cooling_raw[pid] for pid in ordered_ids], neutral=0.5)

        for idx, pid in enumerate(ordered_ids):
            fitness = float(exploit_coef[idx] + explore_coef[idx] + cool_norm[idx])
            variance_value = learning_raw.get(pid)
            if variance_value is None and not self._child_variance_nullable:
                variance_value = 0.5
            cursor.execute(
                """
                UPDATE programs
                SET fitness = ?,
                    exploit_coefficient = ?,
                    explore_coefficient = ?,
                    visit_count = ?,
                    child_base_reward_variance = ?,
                    cooling_coefficient = ?
                WHERE id = ?
                """,
                (
                    fitness,
                    float(exploit_coef[idx]),
                    float(explore_coef[idx]),
                    int(visit_counts.get(pid, 0)),
                    None if variance_value is None else float(variance_value),
                    float(cooling_raw.get(pid, 1.0)),
                    pid,
                ),
            )

    def get_by_id(self, program_id: int) -> Optional[Program]:
        """Get a program by its ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM programs WHERE id = ?", (program_id,))
        row = cursor.fetchone()

        if row:
            return Program.from_dict(dict(row))
        return None

    def is_empty(self, task_name: str = None, task_id: str = None) -> bool:
        """Check if database is empty for a task.

        Args:
            task_name: Task name to check (deprecated, use task_id instead)
            task_id: Task ID to check
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if task_id is not None:
            cursor.execute("SELECT COUNT(*) as count FROM programs WHERE task_id = ?", (task_id,))
        elif task_name is not None:
            cursor.execute("SELECT COUNT(*) as count FROM programs WHERE task_name = ?", (task_name,))
        else:
            raise ValueError("Either task_id or task_name must be provided")

        count = cursor.fetchone()["count"]
        return count == 0

    def size(self, task_name: str) -> int:
        """Get the number of programs for a task."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as count FROM programs WHERE task_name = ?", (task_name,))
        return cursor.fetchone()["count"]

    def clear(self, task_name: Optional[str] = None):
        """Clear programs from the database.

        Args:
            task_name: If provided, only clear programs for this task.
                      If None, clear all programs.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            if task_name:
                cursor.execute("DELETE FROM programs WHERE task_name = ?", (task_name,))
            else:
                cursor.execute("DELETE FROM programs")

            conn.commit()

    def close(self):
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def get_all(self, task_name: str) -> list[Program]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM programs
            WHERE task_name = ?
            ORDER BY id ASC
            """,
            (task_name,),
        )
        rows = cursor.fetchall()
        return [Program.from_dict(dict(row)) for row in rows]

    def sample_by_fitness(
        self, task_name: str, sample_size: int = 1, exclude_ids: Optional[set[int]] = None
    ) -> list[Program]:
        candidates = self.get_all(task_name)
        if exclude_ids:
            candidates = [p for p in candidates if p.id is not None and p.id not in exclude_ids]
        if not candidates or sample_size <= 0:
            return []

        selected: list[Program] = []
        pool = list(candidates)
        k = min(sample_size, len(pool))
        for _ in range(k):
            weights = [max(self._safe_float(p.fitness, default=0.0), 0.0) for p in pool]
            if sum(weights) <= 0.0:
                idx = random.randrange(len(pool))
            else:
                idx = random.choices(range(len(pool)), weights=weights, k=1)[0]
            selected.append(pool.pop(idx))
        return selected

    def get_all_base_rewards(self, *, include_hack: bool = False) -> list[float]:
        """
        Fetch base_reward for all programs in the DB.

        NOTE: This returns the stored `base_reward` field (i.e., the score->reward mapped value
        before any "improve" shaping). It is useful for computing global baselines / percentiles.

        Args:
            include_hack: If False, exclude programs marked as hacking (hack=1).
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            if include_hack:
                cursor.execute("SELECT base_reward FROM programs")
            else:
                cursor.execute("SELECT base_reward FROM programs WHERE hack = 0")
            rows = cursor.fetchall()

        # base_reward column is NOT NULL by schema, but be defensive.
        vals: list[float] = []
        for r in rows:
            try:
                v = r["base_reward"] if isinstance(r, sqlite3.Row) else r[0]
            except Exception:
                v = None
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                continue
        return vals

    def save_snapshot(self, snapshot_path: str):
        """
        Create a snapshot copy of the database.

        Args:
            snapshot_path: Path where the snapshot should be saved
        """
        import shutil

        with self._lock:
            # Ensure all pending transactions are committed
            conn = self._get_connection()
            conn.commit()

            # Create directory if needed
            os.makedirs(
                os.path.dirname(os.path.abspath(snapshot_path)) if os.path.dirname(snapshot_path) else ".",
                exist_ok=True,
            )

            # Copy the database file
            shutil.copy2(self.db_path, snapshot_path)
            print(f"[ProgramDatabase] Saved snapshot to {snapshot_path}")


class SearchAlgorithm:
    """Base class for search algorithms to select programs from database."""

    def __init__(self, draft_probability: float = 0.5):
        """
        Initialize search algorithm.

        Args:
            draft_probability: Probability of using draft mode instead of improve mode
        """
        self.draft_probability = draft_probability

    def select(
        self,
        database: ProgramDatabase,
        task_id: str,
        task_description: str,
        data_description: str,
        public_system_prompt: str,
        public_user_prompt: str,
        data_dir: str,
        max_steps: int = 1,
        task_name: str = None,
        evaluation: bool = False,
    ) -> tuple[tuple[str, str] | None, Optional[Program], str, Optional[Program]]:
        """
        Select a program and build prompt.

        This method decides whether to use draft or improve mode, selects a parent program
        if needed, and builds the appropriate prompt.

        Args:
            database: The ProgramDatabase to select from
            task_id: The task ID to select programs for
            task_description: Task description for prompt building
            data_description: Data description for prompt building
            public_system_prompt: Existing system prompt
            public_user_prompt: Existing user prompt
            data_dir: Virtual data directory path
            max_steps: Maximum number of steps
            task_name: Task name (optional, for logging purposes)
            evaluation: If True, force draft mode (draft_probability=1.0)

        Returns:
            Tuple of (prompts, parent_program, mode, secondary_parent_program)
            - prompts: Tuple of (system_prompt, user_prompt)
            - parent_program: Selected parent Program or None if draft mode
            - mode: One of 'draft' | 'improve' | 'debug' | 'crossover'
            - secondary_parent_program: Used only in crossover mode
        """
        raise NotImplementedError


class GreedySearch(SearchAlgorithm):
    """Greedy search: always select the program with highest reward."""

    def _select_parent(self, database: ProgramDatabase, task_id: str) -> Optional[Program]:
        """Select the program with the highest reward for a given task_id."""
        conn = database._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM programs
            WHERE task_id = ?
            ORDER BY reward DESC
            LIMIT 1
        """,
            (task_id,),
        )

        row = cursor.fetchone()
        if row:
            return Program.from_dict(dict(row))
        return None

    def select(
        self,
        database: ProgramDatabase,
        task_id: str,
        task_description: str,
        data_description: str,
        public_system_prompt: str,
        public_user_prompt: str,
        data_dir: str,
        max_steps: int = 1,
        task_name: str = None,
        evaluation: bool = False,
    ) -> tuple[tuple[str, str] | None, Optional[Program], str, Optional[Program]]:
        """
        Select best program and decide draft vs improve mode.

        Args:
            database: The ProgramDatabase to select from
            task_id: The task ID to select programs for
            task_description: Task description for prompt building
            data_description: Data description for prompt building
            public_system_prompt: Existing system prompt
            public_user_prompt: Existing user prompt
            data_dir: Virtual data directory path
            max_steps: Maximum number of steps
            task_name: Task name (optional, for logging purposes)
            evaluation: If True, force draft mode (draft_probability=1.0)

        Returns:
            Tuple of (prompt, parent_program, mode)
        """

        # Decide whether to use draft or improve mode
        use_improve = False
        parent_program = None

        # Force draft mode during evaluation
        if evaluation:
            log_task = task_name if task_name else task_id
            mode = "draft"
            print(f"[SEARCH] Using RAW EVAL prompt for task {log_task} (task_id={task_id}) [EVALUATION MODE]")
            return (public_system_prompt, public_user_prompt), None, mode, None
        # Only choose between draft and improve if task_id exists in database
        elif not database.is_empty(task_id=task_id):
            # Database has programs for this task_id, decide whether to improve or draft
            if random.random() > self.draft_probability:  # (1 - draft_prob) chance to use improve
                parent_program = self._select_parent(database, task_id)
                if parent_program is not None:
                    use_improve = True
                    log_task = task_name if task_name else task_id
                    print(
                        f"[SEARCH] Using IMPROVE mode for task {log_task} (task_id={task_id}, parent_id={parent_program.id}, parent_reward={parent_program.reward:.4f})"
                    )

        if not use_improve and not evaluation:
            log_task = task_name if task_name else task_id
            print(f"[SEARCH] Using DRAFT mode for task {log_task} (task_id={task_id})")

        # Build prompt based on mode
        mode = "improve" if use_improve else "draft"
        from prompt_builder import build_prompt

        prompt = build_prompt(
            mode=mode,
            task_description=task_description,
            data_description=data_description,
            public_system_prompt=public_system_prompt,
            public_user_prompt=public_user_prompt,
            virtual_data_dir=data_dir,
            parent_program=parent_program,
            max_steps=max_steps,
        )

        return prompt, parent_program, mode, None


class AIRAGreedySearch(SearchAlgorithm):
    def __init__(self, draft_probability: float = 0.5, debug_probability: float = 0.2):
        self.draft_prob = draft_probability or 0.0
        self.debug_prob = debug_probability or 0.0
        assert self.draft_prob + self.debug_prob <= 1.0

    @staticmethod
    def _best_scored_parent(database: ProgramDatabase, task_name: str) -> Program | None:
        cands = [p for p in database.get_all(task_name) if float(p.reward or 0.0) > 0.0]
        if not cands:
            return None
        return max(cands, key=lambda p: float(p.fitness or 0.0))

    @staticmethod
    def _random_debug_parent(database: ProgramDatabase, task_name: str) -> Program | None:
        cands = [p for p in database.get_all(task_name) if float(p.reward or 0.0) <= 0.0]
        if not cands:
            return None
        return random.choice(cands)

    def select(
        self,
        database: ProgramDatabase,
        task_id: str,
        task_description: str,
        data_description: str,
        public_system_prompt: str,
        public_user_prompt: str,
        data_dir: str,
        max_steps: int = 1,
        task_name: str = None,
        evaluation: bool = False,
    ) -> tuple[tuple[str, str] | None, Program | None, str, Program | None]:
        # The first solution must always be a draft.
        # Also force draft mode during evaluation
        if evaluation:
            mode = "draft"
            parent_program = None
            logger.info("[SEARCH] Using RAW EVAL prompt for task %s [EVALUATION MODE]", task_name)
            return (public_system_prompt, public_user_prompt), parent_program, mode, None

        if database.is_empty(task_id=task_id):
            mode = "draft"
            parent_program = None
            logger.info("[SEARCH] Using DRAFT mode for task %s", task_name)
        else:
            mode = "draft"
            parent_program = None

            # Sample initial action first, then apply fallback chain.
            # Probabilities: draft=self.draft_prob, debug=self.debug_prob, improve=remaining.
            sample = random.random()
            if sample < self.draft_prob:
                mode = "draft"
            elif sample < self.draft_prob + self.debug_prob:
                mode = "debug"
            else:
                mode = "improve"

            if mode == "improve":
                parent_program = self._best_scored_parent(database, task_name)
                if parent_program is None:
                    # Requested fallback: improve -> debug.
                    parent_program = self._random_debug_parent(database, task_name)
                    mode = "debug" if parent_program is not None else "draft"
            elif mode == "debug":
                parent_program = self._random_debug_parent(database, task_name)
                if parent_program is None:
                    # Requested fallback: debug -> improve.
                    parent_program = self._best_scored_parent(database, task_name)
                    mode = "improve" if parent_program is not None else "draft"

            if parent_program is not None:
                logger.info(
                    "[SEARCH] Using %s mode for task %s (parent_id=%s, parent_fitness=%.4f)",
                    mode.upper(),
                    task_name,
                    parent_program.id,
                    float(parent_program.fitness),
                )
            else:
                logger.info("[SEARCH] Using DRAFT mode for task %s", task_name)
        from prompt_builder import build_prompt

        prompt = build_prompt(
            mode=mode,
            task_description=task_description,
            data_description=data_description,
            public_system_prompt=public_system_prompt,
            public_user_prompt=public_user_prompt,
            virtual_data_dir=data_dir,
            parent_program=parent_program,
            max_steps=max_steps,
        )
        return prompt, parent_program, mode, None


class AIRAEvoSearch(SearchAlgorithm):
    """Evolution-style search with fitness-proportional sampling and optional crossover."""

    def __init__(
        self,
        draft_probability: float = 0.5,
        improve_probability: float = 0.25,
        debug_probability: float = 0.25,
        crossover_probability: float = 0.5,
    ):
        self.draft_prob = max(float(draft_probability or 0.0), 0.0)
        self.improve_prob = max(float(improve_probability or 0.0), 0.0)
        self.debug_prob = max(float(debug_probability or 0.0), 0.0)
        self.crossover_prob = max(float(crossover_probability or 0.0), 0.0)
        self.enable_crossover = self.crossover_prob > 0.0

    @staticmethod
    def _weighted_sample_without_replacement(candidates: list[Program], k: int) -> list[Program]:
        if k <= 0 or not candidates:
            return []
        pool = list(candidates)
        selected: list[Program] = []
        for _ in range(min(k, len(pool))):
            weights = [max(float(p.fitness or 0.0), 0.0) for p in pool]
            total_weight = sum(weights)
            if total_weight <= 0.0:
                idx = random.randrange(len(pool))
                selected_prob = 1.0 / float(len(pool))
                probs = [selected_prob for _ in pool]
            else:
                idx = random.choices(range(len(pool)), weights=weights, k=1)[0]
                probs = [w / total_weight for w in weights]
                selected_prob = probs[idx]

            # Emit sampling diagnostics for rollout debugging.
            prob_rows = []
            for i, prog in enumerate(pool):
                prob_rows.append((prog, probs[i]))
            prob_rows_sorted = sorted(prob_rows, key=lambda x: x[1], reverse=True)
            top5 = ", ".join(
                f"id={p.id},fitness={float(p.fitness or 0.0):.6f},prob={prob:.6f}" for p, prob in prob_rows_sorted[:5]
            )

            chosen = pool[idx]
            logger.info(
                "[EVO SAMPLE] selected id=%s fitness=%.6f prob=%.6f | top5_probs=[%s]",
                chosen.id,
                float(chosen.fitness or 0.0),
                float(selected_prob),
                top5,
            )
            selected.append(pool.pop(idx))
        return selected

    def _sample_improve_parent(self, database: ProgramDatabase, task_name: str) -> Program | None:
        # Improve/crossover should sample from "scored" nodes only.
        all_nodes = database.get_all(task_name)
        improve_candidates = [p for p in all_nodes if float(p.reward or 0.0) > 0.0]
        picked = self._weighted_sample_without_replacement(improve_candidates, k=1)
        return picked[0] if picked else None

    def _sample_crossover_parents(
        self, database: ProgramDatabase, task_name: str
    ) -> tuple[Program | None, Program | None]:
        all_nodes = database.get_all(task_name)
        improve_candidates = [p for p in all_nodes if float(p.reward or 0.0) > 0.0]
        picked = self._weighted_sample_without_replacement(improve_candidates, k=2)
        if len(picked) != 2:
            return None, None
        return picked[0], picked[1]

    def _sample_debug_parent(self, database: ProgramDatabase, task_name: str) -> Program | None:
        # Debug should prefer "no reward" parents.
        # print(1111)
        all_nodes = database.get_all(task_name)
        debug_candidates = [p for p in all_nodes if float(p.reward or 0.0) <= 0.0]
        if not debug_candidates:
            return None
        picked = self._weighted_sample_without_replacement(debug_candidates, k=1)
        return picked[0] if picked else None

    def select(
        self,
        database: ProgramDatabase,
        task_id: str,
        task_description: str,
        data_description: str,
        public_system_prompt: str,
        public_user_prompt: str,
        data_dir: str,
        max_steps: int = 1,
        task_name: str = None,
        evaluation: bool = False,
    ) -> tuple[tuple[str, str] | None, Program | None, str, Program | None]:
        if evaluation:
            mode = "draft"
            parent_program = None
            secondary_parent = None
            logger.info("[EVO SEARCH] Using RAW EVAL prompt for task %s [EVALUATION MODE]", task_name)
            return (public_system_prompt, public_user_prompt), parent_program, mode, secondary_parent

        if database.is_empty(task_id=task_id):
            mode = "draft"
            parent_program = None
            secondary_parent = None
            logger.info("[EVO SEARCH] Using DRAFT mode for task %s", task_name)
        else:
            mode = "draft"
            parent_program = None
            secondary_parent = None

            actions = ["draft", "improve", "debug"]
            weights = [self.draft_prob, self.improve_prob, self.debug_prob]
            if self.enable_crossover:
                actions.append("crossover")
                weights.append(self.crossover_prob)
            if sum(weights) <= 0.0:
                # Keep a safe fallback when all configured probabilities are zero.
                actions = ["draft", "improve", "debug"] + (["crossover"] if self.enable_crossover else [])
                weights = [1.0 for _ in actions]
            mode = random.choices(actions, weights=weights, k=1)[0]

            if mode == "improve":
                parent_program = self._sample_improve_parent(database, task_name)
                if parent_program is None:
                    # Fallback rule: improve -> debug (if possible), else draft.
                    parent_program = self._sample_debug_parent(database, task_name)
                    mode = "debug" if parent_program is not None else "draft"
            elif mode == "crossover":
                parent_program, secondary_parent = self._sample_crossover_parents(database, task_name)
                if parent_program is None or secondary_parent is None:
                    # Fallback rule for crossover: first try improve, then debug, then draft.
                    parent_program = self._sample_improve_parent(database, task_name)
                    secondary_parent = None
                    if parent_program is not None:
                        mode = "improve"
                    else:
                        parent_program = self._sample_debug_parent(database, task_name)
                        mode = "debug" if parent_program is not None else "draft"
            elif mode == "debug":
                parent_program = self._sample_debug_parent(database, task_name)
                if parent_program is None:
                    # Fallback rule: debug -> improve (if possible), else draft.
                    parent_program = self._sample_improve_parent(database, task_name)
                    mode = "improve" if parent_program is not None else "draft"

            if mode == "debug" and parent_program is not None:
                logger.info(
                    "[EVO SEARCH] Using DEBUG mode for task %s (parent_id=%s, parent_fitness=%.4f)",
                    task_name,
                    parent_program.id,
                    float(parent_program.fitness),
                )
            elif mode == "improve" and parent_program is not None:
                logger.info(
                    "[EVO SEARCH] Using IMPROVE mode for task %s (parent_id=%s, parent_fitness=%.4f)",
                    task_name,
                    parent_program.id,
                    float(parent_program.fitness),
                )
            elif mode == "crossover" and parent_program is not None and secondary_parent is not None:
                logger.info(
                    "[EVO SEARCH] Using CROSSOVER mode for task %s (parent_ids=%s,%s)",
                    task_name,
                    parent_program.id,
                    secondary_parent.id,
                )
            else:
                logger.info("[EVO SEARCH] Using DRAFT mode for task %s", task_name)

        from prompt_builder import build_prompt

        prompt = build_prompt(
            mode=mode,
            task_description=task_description,
            data_description=data_description,
            public_system_prompt=public_system_prompt,
            public_user_prompt=public_user_prompt,
            virtual_data_dir=data_dir,
            parent_program=parent_program,
            secondary_parent_program=secondary_parent,
            max_steps=max_steps,
        )
        return prompt, parent_program, mode, secondary_parent


class AIRAInferenceEvoSearch(SearchAlgorithm):
    """AIRA-Evo search adapted to the RL ProgramDatabase interface."""

    def __init__(
        self,
        policy: str = "rl_mixed",
        num_islands: int = 1,
        max_island_size: int = 500,
        crossover_after_generation: int = 2,
        crossover_probability: float = 0.5,
        execution_timeout: int = 7200,
        initial_temp: float = 1.0,
        final_temp: float = 1.0,
    ):
        self.policy = str(policy or "rl_mixed").strip().lower()
        if self.policy not in {"rl_mixed", "original"}:
            self.policy = "rl_mixed"
        self.num_islands = max(1, int(num_islands or 1))
        self.max_island_size = max(1, int(max_island_size or 500))
        self.crossover_after_generation = max(0, int(crossover_after_generation or 0))
        self.crossover_probability = max(0.0, min(1.0, float(crossover_probability or 0.0)))
        self.execution_timeout = int(execution_timeout or 7200)
        self.initial_temp = float(initial_temp or 1.0)
        self.final_temp = float(final_temp or self.initial_temp)
        self._task_generation: dict[str, int] = {}
        self.last_selection_metadata: dict[str, Any] = {}

    def _all_programs(self, database: ProgramDatabase, task_name: str) -> list[Program]:
        return database.get_all(task_name)

    def _active_success_programs(self, programs: list[Program]) -> list[Program]:
        active = [program for program in programs if is_success_program(program)]
        active.sort(key=lambda program: (program_selection_score(program) or float("-inf"), int(program.id or 0)), reverse=True)
        if len(active) > self.max_island_size:
            active = active[: self.max_island_size]
        return active

    def _debug_programs(self, programs: list[Program]) -> list[Program]:
        debug = [program for program in programs if is_debug_candidate(program)]
        debug.sort(key=lambda program: int(program.id or 0), reverse=True)
        return debug[: self.max_island_size]

    def _temperature(self, generation_id: int) -> float:
        _ = generation_id
        return self.initial_temp if self.initial_temp == self.final_temp else self.initial_temp

    def _island_id(self, program: Program) -> int:
        if self.num_islands <= 1:
            return 0
        return int(program.id or 0) % self.num_islands

    @staticmethod
    def _weighted_pick(candidates: list[Program], utility_items: list[dict[str, Any]], k: int) -> tuple[list[Program], list[dict[str, Any]]]:
        pool = list(zip(candidates, utility_items))
        selected: list[Program] = []
        selected_items: list[dict[str, Any]] = []
        for _ in range(min(k, len(pool))):
            weights = [max(float(item.get("probability") or 0.0), 0.0) for _, item in pool]
            total = sum(weights)
            if total <= 0.0:
                idx = random.randrange(len(pool))
            else:
                idx = random.choices(range(len(pool)), weights=weights, k=1)[0]
            program, item = pool.pop(idx)
            selected.append(program)
            selected_items.append(item)
        return selected, selected_items

    def _sample_by_utility(
        self,
        candidates: list[Program],
        all_programs: list[Program],
        k: int,
        temperature: float,
    ) -> tuple[list[Program], dict[str, Any]]:
        if not candidates:
            return [], {"enabled": True, "candidates": [], "selected_node_ids": []}
        utilities = compute_parent_utilities(
            candidates,
            all_programs=all_programs,
            lower_is_better=False,
            temperature=temperature,
        )
        selected, selected_items = self._weighted_pick(candidates, utilities, k)
        trace = {
            "enabled": True,
            "temperature": temperature,
            "candidates": utilities[:20],
            "selected_node_ids": [program.id for program in selected],
            "selected_items": selected_items,
        }
        return selected, trace

    def _sample_original(
        self,
        active: list[Program],
        all_programs: list[Program],
        mode: str,
        temperature: float,
    ) -> tuple[list[Program], int, dict[str, Any]]:
        required = 2 if mode == "crossover" else 1
        if not active or len(active) < required:
            return [], 0, {"enabled": True, "reason": "not_enough_active_nodes"}
        islands: dict[int, list[Program]] = {idx: [] for idx in range(self.num_islands)}
        for program in active:
            islands[self._island_id(program)].append(program)
        island_ids = [idx for idx, nodes in islands.items() if len(nodes) >= required]
        if not island_ids:
            return [], 0, {"enabled": True, "reason": "no_island_with_enough_nodes"}
        island_scores = []
        for island_id in island_ids:
            scores = [program_selection_score(program) or 0.0 for program in islands[island_id]]
            island_scores.append(sum(scores) / float(len(scores)))
        min_score = min(island_scores)
        max_score = max(island_scores)
        if min_score == max_score:
            weights = [1.0 for _ in island_scores]
        else:
            weights = [(score - min_score) / (max_score - min_score) for score in island_scores]
        island_id = random.choices(island_ids, weights=weights, k=1)[0] if sum(weights) > 0 else random.choice(island_ids)
        selected, trace = self._sample_by_utility(islands[island_id], all_programs, required, temperature)
        trace["island_sampling"] = {"candidate_island_ids": island_ids, "weights": weights, "selected_island_id": island_id}
        return selected, island_id, trace

    def _sample_rl_mixed(self, mode: str, programs: list[Program], active: list[Program], temperature: float) -> tuple[Program | None, Program | None, int, dict[str, Any]]:
        if mode == "debug":
            candidates = self._debug_programs(programs)
            selected, trace = self._sample_by_utility(candidates, programs, 1, temperature)
            return (selected[0] if selected else None), None, 0, trace
        if mode == "crossover":
            selected, trace = self._sample_by_utility(active, programs, 2, temperature)
            island_id = self._island_id(selected[0]) if selected else 0
            if len(selected) == 2:
                return selected[0], selected[1], island_id, trace
            return None, None, island_id, trace
        if mode == "improve":
            selected, trace = self._sample_by_utility(active, programs, 1, temperature)
            island_id = self._island_id(selected[0]) if selected else 0
            return (selected[0] if selected else None), None, island_id, trace
        return None, None, 0, {"enabled": False}

    @staticmethod
    def _choose_rl_mixed_mode() -> str:
        return random.choices(
            ["draft", "improve", "debug", "crossover"],
            weights=[0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
            k=1,
        )[0]

    def _fallback_mode(
        self,
        requested_mode: str,
        programs: list[Program],
        active: list[Program],
        temperature: float,
        include_requested: bool = True,
    ) -> tuple[str, Program | None, Program | None, int, dict[str, Any]]:
        search_order = {
            "crossover": ["crossover", "improve", "debug", "draft"],
            "improve": ["improve", "debug", "draft"],
            "debug": ["debug", "improve", "draft"],
            "draft": ["draft"],
        }.get(requested_mode, ["draft"])
        fallback_order = search_order if include_requested else search_order[1:] or ["draft"]
        for mode in fallback_order:
            if mode == "draft":
                return "draft", None, None, 0, {"enabled": False, "fallback_from": requested_mode}
            parent, secondary, island_id, trace = self._sample_rl_mixed(mode, programs, active, temperature)
            if mode == "crossover" and parent is not None and secondary is not None:
                trace["fallback_from"] = requested_mode
                return mode, parent, secondary, island_id, trace
            if mode in {"improve", "debug"} and parent is not None:
                trace["fallback_from"] = requested_mode
                return mode, parent, None, island_id, trace
        return "draft", None, None, 0, {"enabled": False, "fallback_from": requested_mode}

    def select(
        self,
        database: ProgramDatabase,
        task_id: str,
        task_description: str,
        data_description: str,
        public_system_prompt: str,
        public_user_prompt: str,
        data_dir: str,
        max_steps: int = 1,
        task_name: str = None,
        evaluation: bool = False,
    ) -> tuple[tuple[str, str] | None, Program | None, str, Program | None]:
        task_key = task_name or task_id or "unknown"
        generation_id = int(self._task_generation.get(task_key, 0))
        self.last_selection_metadata = {}
        if evaluation:
            self.last_selection_metadata = {"airaevo_policy": self.policy, "airaevo_operator": "draft", "airaevo_generation_id": generation_id}
            return (public_system_prompt, public_user_prompt), None, "draft", None

        programs = self._all_programs(database, task_key)
        active = self._active_success_programs(programs)
        temperature = self._temperature(generation_id)
        requested_mode = "draft"
        parent_program: Program | None = None
        secondary_parent: Program | None = None
        island_id = 0
        selection_trace: dict[str, Any] = {"enabled": False}

        if not programs:
            mode = "draft"
        elif self.policy == "original":
            if generation_id == 0 or not active:
                mode = "draft"
            else:
                crossover_ready = generation_id >= self.crossover_after_generation and len(active) >= 2
                requested_mode = "crossover" if crossover_ready and random.random() < self.crossover_probability else "improve"
                selected, island_id, selection_trace = self._sample_original(active, programs, requested_mode, temperature)
                if requested_mode == "crossover" and len(selected) == 2:
                    mode = "crossover"
                    parent_program, secondary_parent = selected[0], selected[1]
                elif requested_mode == "improve" and len(selected) == 1:
                    mode = "improve"
                    parent_program = selected[0]
                else:
                    mode, parent_program, secondary_parent, island_id, selection_trace = self._fallback_mode(requested_mode, programs, active, temperature, include_requested=False)
        else:
            requested_mode = self._choose_rl_mixed_mode()
            if requested_mode == "draft":
                mode = "draft"
            else:
                mode, parent_program, secondary_parent, island_id, selection_trace = self._fallback_mode(requested_mode, programs, active, temperature)

        parents = [program for program in [parent_program, secondary_parent] if program is not None]
        memory = build_operator_experience_memory(
            mode,
            parents,
            all_programs=programs,
            current_program=parent_program if mode == "debug" else None,
            lower_is_better=False,
            max_related_cards=8 if mode == "debug" else 3,
        )
        from prompt_builder import build_airaevo_prompt

        prompt = build_airaevo_prompt(
            mode=mode,
            task_description=task_description,
            data_description=data_description,
            public_system_prompt=public_system_prompt,
            public_user_prompt=public_user_prompt,
            virtual_data_dir=data_dir,
            parent_program=parent_program,
            secondary_parent_program=secondary_parent,
            max_steps=max_steps,
            memory=memory,
            data_overview="",
            execution_timeout=self.execution_timeout,
        )
        self.last_selection_metadata = {
            "airaevo_policy": self.policy,
            "airaevo_operator": mode,
            "airaevo_requested_operator": requested_mode,
            "airaevo_generation_id": generation_id,
            "airaevo_island_id": island_id,
            "airaevo_active_population_size": len(active),
            "airaevo_active_population_cap": self.max_island_size,
            "airaevo_num_islands": self.num_islands,
            "airaevo_selection_trace": selection_trace,
            "airaevo_memory_present": bool(memory),
        }
        self._task_generation[task_key] = generation_id + 1
        logger.info(
            "[AIRAEVO SEARCH] task=%s policy=%s mode=%s requested=%s parent=%s secondary=%s active=%s/%s",
            task_key,
            self.policy,
            mode,
            requested_mode,
            getattr(parent_program, "id", None),
            getattr(secondary_parent, "id", None),
            len(active),
            self.max_island_size,
        )
        return prompt, parent_program, mode, secondary_parent
