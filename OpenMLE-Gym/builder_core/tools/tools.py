import pandas as pd
import os
from collections import defaultdict
from pathlib import Path
from langchain_core.tools import tool
from openmle_gym.common import atomic_write_text

_READ_ROOT: Path | None = None
_SAVE_PATH: Path | None = None


def configure_task_paths(read_root: str | Path, save_path: str | Path) -> None:
    """Bind tool access to one task. Each task runs in its own process."""
    global _READ_ROOT, _SAVE_PATH
    _READ_ROOT = Path(read_root).resolve()
    _SAVE_PATH = Path(save_path).resolve()


def _resolve_read_path(value: str) -> Path:
    if _READ_ROOT is None:
        raise RuntimeError("Task tool read root is not configured")
    candidate = Path(value)
    if not candidate.is_absolute():
        if candidate.parts and candidate.parts[0] == _READ_ROOT.name:
            candidate = _READ_ROOT.parent / candidate
        else:
            candidate = _READ_ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(_READ_ROOT)
    except ValueError as exc:
        raise PermissionError(f"Path is outside the current task raw directory: {value}") from exc
    return resolved


def _resolve_save_path(value: str) -> Path:
    if _SAVE_PATH is None:
        raise RuntimeError("Task tool save path is not configured")
    resolved = Path(value).resolve()
    if resolved != _SAVE_PATH:
        raise PermissionError(f"save may only write the current task fileinfo.txt: {value}")
    return resolved


@tool
def list_directory_contents(path: str) -> str:
    """Lists the contents of the specified path. 
Automatically aggregates files by extension if the item count is too high. 
    Directories always appear first."""
    try:
        resolved_path = _resolve_read_path(path)
        all_items = os.listdir(resolved_path)
        if not all_items:
            return f"The directory '{path}' is empty"

        directories = []
        files_by_ext = defaultdict(list)

        for item in all_items:
            full_path = resolved_path / item
            if os.path.isdir(full_path):
                directories.append(item)
            else:
                _, ext = os.path.splitext(item)
                ext = ext.lower() if ext else "[no extension]"
                files_by_ext[ext].append(item)
                
        directories.sort()
        sorted_exts = sorted(files_by_ext.keys())

        total_count = len(all_items)
        
        if total_count <= 50:
            output = [f"Listing {path} ({total_count} items):"]
            for d in directories:
                output.append(f"  [Dir]  {d}")
            for ext in sorted_exts:
                for f in sorted(files_by_ext[ext]):
                    output.append(f"  [File] {f}")
            return "\n".join(output)

        else:
            report = [f"Directory Report: {path} ({total_count} items in total)"]
            report.append("-" * 3)
            
            if directories:
                dir_sample = directories[:5]
                report.append(f"Subdirectories: {len(directories)} in total")
                report.append(f"   Samples: {', '.join(dir_sample)}" + ("..." if len(directories) > 5 else ""))
            
            report.append(f"Categorized File Report:")
            for ext in sorted(files_by_ext, key=lambda k: len(files_by_ext[k]), reverse=True):
                count = len(files_by_ext[ext])
                samples = ", ".join(files_by_ext[ext][:2])
                report.append(f"   - {ext:12} : {count:3} in total (Samples: {samples}...)")
            
            report.append("-" * 3)
            report.append("Too many items; results have been automatically aggregated. To view specific files, please refine the search path")
            
            return "\n".join(report)

    except Exception as e:
        return f"Error: Unable to open '{path}': {str(e)}"

@tool
def get_csv_summary(csv_path: str) -> dict:
    """Analyzes a CSV file and returns a summary. Limits analysis to the first `max_cols` columns to ensure readability and performance."""
    try:
        resolved_csv = _resolve_read_path(csv_path)
        max_cols = 32
        
        # 1. Read header only to determine columns
        full_df_columns = pd.read_csv(resolved_csv, nrows=0).columns.tolist()
        
        # Apply the 32-column limit
        target_columns = full_df_columns[:max_cols]
        is_truncated = len(full_df_columns) > max_cols

        # 2. Get sample data for the limited columns
        df_sample = pd.read_csv(resolved_csv, nrows=3, usecols=target_columns)
        
        # Initialize counters
        row_count = 0
        null_counts = {col: 0 for col in target_columns}

        # 3. Process file in chunks (only reading target columns for speed)
        with pd.read_csv(resolved_csv, usecols=target_columns, chunksize=100000) as reader:
            for chunk in reader:
                row_count += len(chunk)
                # Count nulls for the limited set of columns
                chunk_nulls = chunk.isnull().sum().to_dict()
                for col, count in chunk_nulls.items():
                    null_counts[col] += count

        # 4. Get data types (from a small sample)
        df_types = pd.read_csv(resolved_csv, nrows=100, usecols=target_columns)
        dtypes_dict = df_types.dtypes.apply(lambda x: str(x)).to_dict()

        # 5. Construct column-specific details
        column_info = {}
        for col in target_columns:
            column_info[col] = {
                "data_type": dtypes_dict.get(col, "unknown"),
                "null_count": int(null_counts[col])
            }

        # 6. Build final summary
        summary = {
            "file_info": {
                "filename": os.path.basename(csv_path),
                "columns_analyzed": len(target_columns),
                "is_column_limit_reached": is_truncated
            },
            "shape": {
                "total_rows": row_count,
                "total_columns": len(full_df_columns)
            },
            "column_details": column_info,
            "sample_data": df_sample.to_dict(orient='records')
        }
        
        if is_truncated:
            summary["note"] = f"Only the first {max_cols} columns were analyzed due to the limit. For full description of columns, refer to Dataset Description"
            
        return summary

    except Exception as e:
        return {"error": f"Failed to analyze CSV at {csv_path}: {str(e)}. You may consider skipping this file"}

@tool
def read_txt_md(file_path: str) -> str:
    """Display a bounded preview of a text file. One preview is sufficient for inventory."""
    try:
        resolved_file = _resolve_read_path(file_path)
        # Check if the file exists and is a file before opening
        char_limit = 6000
        with open(resolved_file, "r", encoding="utf-8") as f:
            # Read only up to the char_limit
            content = f.read(char_limit)
            is_truncated = len(f.read(1)) > 0
            if is_truncated:
                return (
                    f"File {file_path} content (preview truncated to "
                    f"{char_limit} chars; this single preview is sufficient, "
                    "do not read the same path again):\n"
                    f"{content}\n... [End of preview]"
                )
            return f"File {file_path} content:\n{content}"
    except UnicodeDecodeError:
        return f"Error: File {file_path} contains non-UTF-8 characters or binary data. Skip this file"
    except Exception as e:
        return f"File {file_path} reading error: {e}. You may consider skipping this file"

@tool
def save(info_path: str, content: str):
    """Fully overwrite the content in fileinfo.txt. The only EXIT of tool use. Call this to save, and terminate the tool use."""
    path = _resolve_save_path(info_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, content + "\n" + "-"*20 + "\n")
    return f"Successfully saved to fileinfo.txt"
