from __future__ import annotations

import csv
import io
import json
import math
import os
import statistics
import time
import zipfile
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry

from .common import json_safe, read_slugs_file


BASE_URL = "https://www.kaggle.com/api/v1/competitions/{slug}/leaderboard/download"


def _load_auth(kaggle_json: str | Path | None = None) -> tuple[str, str]:
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, key

    config_paths = []
    if kaggle_json:
        config_paths.append(Path(kaggle_json).expanduser())
    kaggle_config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    if kaggle_config_dir:
        config_paths.append(Path(kaggle_config_dir).expanduser() / "kaggle.json")
    config_paths.append(Path("~/.kaggle/kaggle.json").expanduser())

    for config_path in config_paths:
        if not config_path.exists():
            continue
        data = json.loads(config_path.read_text(encoding="utf-8"))
        username = data.get("username")
        key = data.get("key")
        if username and key:
            return username, key
    raise RuntimeError(
        "Kaggle credentials not found. Set KAGGLE_USERNAME/KAGGLE_KEY or pass --kaggle-json."
    )


def _make_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "openmle-gym/0.1",
            "Accept": "application/zip,text/csv,application/octet-stream,*/*",
        }
    )
    return session


def _looks_like_csv(content: bytes) -> bool:
    if not content:
        return False
    sample = content[:4096].decode("utf-8-sig", errors="replace").strip()
    if not sample:
        return False
    lowered = sample.lower()
    if lowered.startswith("<!doctype html") or lowered.startswith("<html"):
        return False
    if "login" in lowered and "kaggle" in lowered and "<" in lowered:
        return False
    try:
        rows = list(csv.reader(sample.splitlines()))
    except Exception:
        return False
    return bool(rows and len(rows[0]) >= 2)


def _extract_csv(content: bytes, slug: str) -> bytes:
    if not content.startswith(b"PK"):
        if _looks_like_csv(content):
            return content
        preview = content[:1000].decode("utf-8", errors="replace").replace("\n", " ")
        raise RuntimeError(f"{slug}: response is neither CSV nor ZIP containing CSV. Preview: {preview}")

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"{slug}: ZIP response has no CSV file.")
        preferred = [name for name in csv_names if "leaderboard" in Path(name).name.lower()]
        data = zf.read(preferred[0] if preferred else csv_names[0])
        if not _looks_like_csv(data):
            raise RuntimeError(f"{slug}: extracted CSV candidate does not look like CSV.")
        return data


def _to_float(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _select_score_column(header: list[str], rows: list[list[str]]) -> int | None:
    """Pick the score column: an exact/partial ``score`` header, else the
    right-most column whose present values are all numeric."""
    lowered = [cell.strip().lower() for cell in header]
    if "score" in lowered:
        return lowered.index("score")
    for index, name in enumerate(lowered):
        if "score" in name:
            return index
    for index in range(len(header) - 1, -1, -1):
        present = [row[index] for row in rows if index < len(row)]
        numeric = [value for value in present if _to_float(value) is not None]
        if present and len(numeric) == len(present):
            return index
    return None


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (``q`` in [0, 100]) over ``values``."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def summarize_leaderboard(source: str | Path | bytes) -> dict[str, Any]:
    """Summarize a Kaggle public-leaderboard CSV into score statistics.

    ``source`` is a CSV file path or its raw ``bytes``. Returns a JSON-safe dict
    with the detected ``score_column`` and ``count``/``min``/``max``/``median``/
    ``p10``/``p90``, or ``{"status": "unavailable", "reason": ...}`` when no
    numeric score column is found. The ``min``/``max`` feed the consumer task
    metadata's ``leaderboard_min``/``leaderboard_max`` fields.
    """
    if isinstance(source, bytes):
        text = source.decode("utf-8-sig", errors="replace")
    else:
        text = Path(source).read_text(encoding="utf-8-sig")

    rows = [row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        return {"status": "unavailable", "reason": "no leaderboard rows"}

    header, data_rows = rows[0], rows[1:]
    index = _select_score_column(header, data_rows)
    if index is None:
        return {"status": "unavailable", "reason": "no numeric score column"}

    values = [
        number
        for row in data_rows
        if index < len(row)
        for number in (_to_float(row[index]),)
        if number is not None
    ]
    if not values:
        return {"status": "unavailable", "reason": "score column has no numeric values"}

    return json_safe(
        {
            "status": "ok",
            "score_column": header[index].strip(),
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
            "p10": _percentile(values, 10.0),
            "p90": _percentile(values, 90.0),
        }
    )


def download_leaderboards(
    slugs_file: str | Path,
    out_dir: str | Path,
    kaggle_json: str | Path | None = None,
    timeout: int = 120,
    sleep_seconds: float = 1.0,
    execute: bool = False,
    summarize: bool = False,
) -> list[dict[str, Any]]:
    slugs = read_slugs_file(slugs_file)
    if not slugs:
        raise ValueError(f"No competition slugs found in {slugs_file}")

    output_dir = Path(out_dir)
    if not execute:
        return [
            {
                "slug": slug,
                "status": "dry-run",
                "target": str(output_dir / f"{slug}.csv"),
            }
            for slug in slugs
        ]

    username, key = _load_auth(kaggle_json)
    auth = HTTPBasicAuth(username, key)
    session = _make_session()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for index, slug in enumerate(slugs, start=1):
        target = output_dir / f"{slug}.csv"
        try:
            response = session.get(BASE_URL.format(slug=slug), auth=auth, timeout=timeout)
            if response.status_code != 200:
                preview = response.text[:1000].replace("\n", " ")
                raise RuntimeError(f"HTTP {response.status_code}: {preview}")
            csv_bytes = _extract_csv(response.content, slug)
            target.write_bytes(csv_bytes)
            result = {"slug": slug, "status": "ok", "target": str(target)}
            if summarize:
                result["summary"] = summarize_leaderboard(csv_bytes)
            results.append(result)
        except Exception as exc:
            results.append({"slug": slug, "status": "failed", "error": str(exc), "target": str(target)})
        if index < len(slugs):
            time.sleep(sleep_seconds)
    return results
