"""
Program Database for storing and managing generated programs during rollout.
Uses SQLite for persistent storage with thread-safety.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import sqlite3
import threading
from typing import Any


class DatabaseLifecycle(Enum):
    """Lifecycle modes for Program Database."""
    ROLLOUT = "rollout"  # One database per rollout
    STEP = "step"  # One database per step
    TASK = "task"  # One database per task


@dataclass
class Program:
    """Represents a program in the database."""
    id: int | None = None  # Database ID
    task_id: str = ""  # Task identifier (uuid)
    task_name: str = ""  # Task name
    code: str = ""  # The code text
    score: float | None = None  # Score from sandbox execution
    reward: float = 0.0  # Reward calculated from score (base_reward)
    fitness: float | None = None  # Fitness used by search selection
    base_reward: float = 0.0  # Base reward before considering parent
    run_log: str = ""  # Execution log
    feedback: str = ""  # Feedback text from evaluator or search policy
    parent_id: int | None = None  # Parent program ID (for improve mode)
    parent_code: str = ""  # Parent program code (for logging)
    generation_mode: str = "draft"  # Generation mode: 'draft' | 'improve' | 'debug' | 'crossover'
    raw_text: str = ""  # Raw response text from model
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def __post_init__(self) -> None:
        if self.fitness is None:  # Default fitness when not provided.
            self.fitness = self.reward  # Align fitness with reward by default.
        clear_run_log = self.metadata.get("clear_run_log")
        if clear_run_log is not None:
            self.run_log = str(clear_run_log)

    def to_dict(self) -> dict[str, Any]:
        """Convert program to dictionary for storage."""
        return {
            'id': self.id,
            'task_id': self.task_id,
            'task_name': self.task_name,
            'code': self.code,
            'score': self.score,
            'reward': self.reward,
            'fitness': self.fitness,
            'base_reward': self.base_reward,
            'run_log': self.run_log,
            'feedback': self.feedback,
            'parent_id': self.parent_id,
            'parent_code': self.parent_code,
            'generation_mode': self.generation_mode,
            'raw_text': self.raw_text,
            'metadata': json.dumps(self.metadata)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Program":
        """Create program from dictionary."""
        metadata_value = data.get("metadata", {})
        if isinstance(metadata_value, str):
            metadata = json.loads(metadata_value or "{}")
        else:
            metadata = metadata_value or {}
        run_log = data.get('run_log', '')
        clear_run_log = metadata.get("clear_run_log")
        if clear_run_log is not None:
            run_log = str(clear_run_log)
        return cls(
            id=data.get('id'),
            task_id=data.get('task_id', ''),
            task_name=data.get('task_name', ''),
            code=data.get('code', ''),
            score=data.get('score'),
            reward=data.get('reward', 0.0),
            fitness=data.get('fitness'),
            base_reward=data.get('base_reward', 0.0),
            run_log=run_log,
            feedback=data.get('feedback') or '',
            parent_id=data.get('parent_id'),
            parent_code=data.get('parent_code', ''),
            generation_mode=data.get('generation_mode', 'draft'),
            raw_text=data.get('raw_text', ''),
            metadata=metadata
        )


class ProgramDatabase:
    """Database for storing and managing programs during rollout.

    Uses SQLite for persistent storage with connection pooling and thread-safety.
    Each task maintains its top-k programs by fitness.
    """

    def __init__(
        self, db_path: str = "program_database.db", max_per_task: int | None = 10
    ):
        """
        Initialize Program Database.

        Args:
            db_path: Path to SQLite database file
            max_per_task: Maximum number of programs to keep per task (top k by fitness).
                          Use None to disable pruning.
        """
        self.db_path = db_path
        self.max_per_task = max_per_task

        self._local = threading.local()  # Per-instance connections to avoid epoch cross-talk.
        self._lock = threading.Lock()  # Serialize writes per database.

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) if os.path.dirname(db_path) else ".", exist_ok=True)

        # Initialize database schema
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS programs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                task_name TEXT NOT NULL,
                code TEXT NOT NULL,
                score REAL,
                reward REAL NOT NULL,
                fitness REAL NOT NULL,
                base_reward REAL NOT NULL,
                run_log TEXT,
                feedback TEXT NOT NULL DEFAULT '',
                parent_id INTEGER,
                parent_code TEXT,
                generation_mode TEXT DEFAULT 'draft',
                raw_text TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_id) REFERENCES programs(id)
            )
        ''')

        cursor.execute('PRAGMA table_info(programs)')  # Read table schema metadata.
        columns = {row['name'] for row in cursor.fetchall()}  # Collect column names.

        if 'fitness' not in columns:  # Add fitness column when missing.
            cursor.execute('ALTER TABLE programs ADD COLUMN fitness REAL NOT NULL DEFAULT 0.0')  # Add fitness column.
            cursor.execute('UPDATE programs SET fitness = reward')  # Backfill fitness from reward.
        if 'feedback' not in columns:
            cursor.execute("ALTER TABLE programs ADD COLUMN feedback TEXT NOT NULL DEFAULT ''")
        if 'created_at' not in columns:
            cursor.execute(
                'ALTER TABLE programs ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
            )
        # if 'score' in columns and columns['score']['notnull']:
        #     cursor.execute('''
        #         CREATE TABLE IF NOT EXISTS programs_migrated (
        #             id INTEGER PRIMARY KEY AUTOINCREMENT,
        #             task_id TEXT,
        #             task_name TEXT NOT NULL,
        #             code TEXT NOT NULL,
        #             score REAL,
        #             reward REAL NOT NULL,
        #             fitness REAL NOT NULL,
        #             base_reward REAL NOT NULL,
        #             run_log TEXT,
        #             parent_id INTEGER,
        #             parent_code TEXT,
        #             generation_mode TEXT DEFAULT 'draft',
        #             raw_text TEXT,
        #             metadata TEXT,
        #             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        #             FOREIGN KEY (parent_id) REFERENCES programs(id)
        #         )
        #     ''')
        #     cursor.execute('''
        #         INSERT INTO programs_migrated (
        #             id, task_id, task_name, code, score, reward, fitness, base_reward,
        #             run_log, parent_id, parent_code, generation_mode, raw_text, metadata, created_at
        #         )
        #         SELECT
        #             id, task_id, task_name, code, score, reward, fitness, base_reward,
        #             run_log, parent_id, parent_code, generation_mode, raw_text, metadata, created_at
        #         FROM programs
        #     ''')
        #     cursor.execute('DROP TABLE programs')
        #     cursor.execute('ALTER TABLE programs_migrated RENAME TO programs')

        # Create index for faster task-based queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_task_name ON programs(task_name)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_task_fitness ON programs(task_name, fitness DESC)
        ''')

        conn.commit()

    def add(self, program: Program) -> int:
        """
        Add a program to the database, maintaining top k per task by fitness.

        Returns:
            The database ID of the inserted program
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Insert the program
            cursor.execute('''
                INSERT INTO programs (task_id, task_name, code, score, reward, fitness, base_reward,
                                     run_log, feedback, parent_id, parent_code, generation_mode,
                                     raw_text, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                program.task_id,
                program.task_name,
                program.code,
                program.score,
                program.reward,
                program.fitness,
                program.base_reward,
                program.run_log,
                program.feedback,
                program.parent_id,
                program.parent_code,
                program.generation_mode,
                program.raw_text,
                json.dumps(program.metadata)
            ))
            program_id = cursor.lastrowid

            # Maintain top k programs per task
            cursor.execute('''
                SELECT COUNT(*) as count FROM programs WHERE task_name = ?
            ''', (program.task_name,))
            count = cursor.fetchone()['count']

            if self.max_per_task is not None and count > self.max_per_task:
                # Delete programs beyond top k
                cursor.execute('''
                    DELETE FROM programs
                    WHERE task_name = ? AND id NOT IN (
                        SELECT id FROM programs
                        WHERE task_name = ?
                        ORDER BY fitness DESC
                        LIMIT ?
                    )
                ''', (program.task_name, program.task_name, self.max_per_task))

            conn.commit()
            return program_id

    def get_by_id(self, program_id: int) -> Program | None:
        """Get a program by its ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM programs WHERE id = ?', (program_id,))
        row = cursor.fetchone()

        if row:
            return Program.from_dict(dict(row))
        return None

    def is_empty(self, task_name: str) -> bool:
        """Check if database is empty for a task."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) as count FROM programs WHERE task_name = ?', (task_name,))
        count = cursor.fetchone()['count']
        return count == 0

    def size(self, task_name: str) -> int:
        """Get the number of programs for a task."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) as count FROM programs WHERE task_name = ?', (task_name,))
        return cursor.fetchone()['count']

    def count_by_generation_mode(self, task_name: str, generation_mode: str) -> int:
        """Count programs for a task by generation mode."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT COUNT(*) as count FROM programs WHERE task_name = ? AND generation_mode = ?',
            (task_name, generation_mode),
        )
        return cursor.fetchone()['count']

    def clear(self, task_name: str | None = None):
        """Clear programs from the database.

        Args:
            task_name: If provided, only clear programs for this task.
                      If None, clear all programs.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            if task_name:
                cursor.execute('DELETE FROM programs WHERE task_name = ?', (task_name,))
            else:
                cursor.execute('DELETE FROM programs')

            conn.commit()

    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def get_best(self, task_name: str) -> Program:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT * FROM programs
            WHERE task_name = ?
            ORDER BY fitness DESC
            LIMIT 1
            ''',
            (task_name,),
        )
        row = cursor.fetchone()
        return Program.from_dict(dict(row))

    def get_random_by_fitness(self, task_name: str, fitness: float) -> Program | None:
        """Randomly sample a program by fitness."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT * FROM programs
            WHERE task_name = ? AND fitness = ?
            ORDER BY RANDOM()
            LIMIT 1
            ''',
            (task_name, fitness),
        )
        row = cursor.fetchone()
        if row:
            return Program.from_dict(dict(row))
        return None

    def get_top_k(self, task_name: str, k: int) -> list[Program]:
        """Return top-k programs by fitness (descending)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM programs
            WHERE task_name = ?
            ORDER BY fitness DESC
            LIMIT ?
            """,
            (task_name, int(k)),
        )
        rows = cursor.fetchall()
        return [Program.from_dict(dict(row)) for row in rows]

    def list_by_task(self, task_name: str) -> list[Program]:
        """List programs for a task in creation order."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT * FROM programs
            WHERE task_name = ?
            ORDER BY created_at ASC, id ASC
            ''',
            (task_name,),
        )
        rows = cursor.fetchall()
        return [Program.from_dict(dict(row)) for row in rows]


class SearchAlgorithm(ABC):
    """Abstract base class for search algorithms to select programs from database."""

    @abstractmethod
    def select(
        self,
        database: ProgramDatabase,
        task_name: str,
        task_description: str,
        data_description: str,
        public_system_prompt: str,
        public_user_prompt: str,
        data_dir: str,
        max_steps: int = 1,
    ) -> tuple[tuple[str, str] | None, Program | None, str, str | None, Program | None]:
        """
        Select a program and build prompt.

        This method decides whether to use draft or improve mode, selects a parent program
        if needed, and builds the appropriate prompt.

        Args:
            database: The ProgramDatabase to select from
            task_name: The task name to select programs for
            task_description: Task description text (task-specific)
            data_description: Data description text (task-specific)
            public_system_prompt: Dataset-provided system prompt fragment to append
            public_user_prompt: Dataset-provided user prompt fragment to append
            data_dir: Virtual data directory path
            max_steps: Maximum number of steps

        Returns:
            Tuple of (prompts, parent_program, mode, model_name, secondary_parent_program)
            - prompts: (system_prompt, user_prompt)
            - parent_program: Selected parent Program or None if draft mode
            - mode: One of 'draft' | 'improve' | 'debug' | 'crossover'
            - model_name: Selected model name or None
            - secondary_parent_program: Only used for crossover mode
        """
        raise NotImplementedError

    @abstractmethod
    def select_best(self, database: ProgramDatabase, task_name: str) -> Program:
        raise NotImplementedError
