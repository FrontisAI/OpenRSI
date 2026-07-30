# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import json
import yaml
import streamlit as st
import subprocess
from typing import Dict, List, Any, Optional, Tuple
import tempfile
from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
import re
import base64
import io
import pandas as pd
import time
from datetime import datetime, timedelta
import sys
from pathlib import Path
import psutil
import pynvml
import webbrowser
import urllib.parse
import plotly.graph_objects as go
import plotly.express as px
from dojo.config_dataclasses.run import RunConfig
from dojo.utils.experiment_logs import is_experiment

# Import ruamel.yaml for preserving YAML ordering
try:
    from ruamel.yaml import YAML

    ruamel_yaml = YAML()
    ruamel_yaml.preserve_quotes = True  # Preserve quotes in strings
    ruamel_yaml.indent(mapping=2, sequence=4, offset=2)  # Set indentation
    ruamel_yaml.width = 1000  # Don't wrap long lines
    ruamel_yaml.preserve_comments = True  # Preserve comments
    RUAMEL_AVAILABLE = True
except ImportError:
    RUAMEL_AVAILABLE = False

from viz_utils import (
    create_config_graph,
    visualize_config_graph,
    create_plotly_config_graph,
    create_config_comparison,
    visualize_parameter_heatmap,
    create_config_diff_viz,
    load_yaml_config as viz_load_yaml_config,  # Rename to avoid conflict
)

# Import streamlit-ace for YAML editing
try:
    from streamlit_ace import st_ace

    ACE_AVAILABLE = True
except ImportError:
    ACE_AVAILABLE = False


@st.cache_data
def list_experiments(meta_exp_path: Path) -> List[str]:
    """List all experiment folders within a meta experiment directory."""
    experiments_path = meta_exp_path
    if not experiments_path.exists():
        return []

    return [d.name for d in experiments_path.iterdir() if is_experiment(d)]


@st.cache_data
def list_files_recursive(directory: Path, max_depth: int = 3, current_depth: int = 0) -> Dict:
    """Recursively list files and folders in a directory up to max_depth."""
    if not directory.exists() or current_depth > max_depth:
        return {}

    result = {}
    try:
        for item in directory.iterdir():
            if item.is_dir():
                result[f"{item.name}/"] = list_files_recursive(item, max_depth, current_depth + 1)
            else:
                result[item.name] = str(item.absolute())
    except PermissionError:
        st.warning(f"Permission denied accessing directory: {directory}", icon="⚠️")
        return {"error/": f"Permission denied accessing {directory.name}"}
    except Exception as e:
        st.error(f"Error listing directory {directory}: {e}")
        return {"error/": f"Error listing {directory.name}"}

    return result


@st.cache_data
def find_log_files(exp_path: Path, pattern: str = "*.jsonl") -> List[str]:
    """Find all log files matching a pattern in the experiment directory."""
    return [str(p) for p in exp_path.glob(f"**/{pattern}")]


@st.cache_data
def analyze_meta_experiment(meta_exp_path: Path) -> Dict:
    """Generate basic statistics about a meta experiment."""
    stats = {}
    exp_path = meta_exp_path

    if not exp_path.exists():
        return {"error": f"Experiments directory not found: {exp_path}"}

    # Count experiments
    try:
        experiments = [d for d in exp_path.iterdir() if d.is_dir()]
        stats["total_experiments"] = len(experiments)
    except PermissionError:
        return {"error": f"Permission denied accessing experiments directory: {exp_path}"}
    except Exception as e:
        return {"error": f"Error accessing experiments directory: {exp_path} - {e}"}

    # Look for common files
    try:
        stats["experiments_with_journal"] = sum(1 for exp in experiments if (exp / "json" / "JOURNAL.jsonl").exists())
        stats["experiments_with_config"] = sum(1 for exp in experiments if (exp / "dojo_config.json").exists())
    except Exception as e:
        st.warning(f"Error checking files within experiments: {e}")
        stats["experiments_with_journal"] = "Error"
        stats["experiments_with_config"] = "Error"

    # Check for slurm logs
    try:
        # Fetch slurm ids from configs
        slurm_ids = [
            RunConfig.load_from_json(exp / "dojo_config.json").metadata.slurm_id
            for exp in experiments
            if (exp / "dojo_config.json").exists()
        ]
        slurm_ids = [sid for sid in slurm_ids if sid != ""]

        submitit_logs_path = meta_exp_path.parent.parent / "slurm_logs"
        has_slurm = submitit_logs_path.exists()
        stats["has_slurm_logs"] = has_slurm
        # get paths to relevant slurm logs
        if has_slurm:
            slurm_logs = []
            for sid in slurm_ids:
                slurm_logs.extend(list(submitit_logs_path.glob(f"**/*{sid}*.out")))
            stats["slurm_job_count"] = len(slurm_logs)

    except Exception as e:
        st.warning(f"Error checking for Slurm logs: {e}")
        stats["has_slurm_logs"] = False
        stats["slurm_job_count"] = "Error"

    return stats


# Note: execute_utility performs actions, so it shouldn't be cached directly
def execute_utility(utility_name: str, meta_exp_path: Path, output_dir: Optional[Path] = None) -> Tuple[bool, str]:
    """Execute a utility script on the meta experiment data."""
    if not output_dir:
        output_dir = meta_exp_path

    actual_exp_path = meta_exp_path

    # Import the utility module dynamically
    try:
        # Add the project root to sys.path if necessary, assuming this script is two levels down from dojo
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        if utility_name == "log_error_parsing":
            from dojo.analysis_utils.meta_error_summary import generate_error_reports

            generate_error_reports(actual_exp_path, output_dir)
            # Construct the expected report path based on the utility's implementation
            report_path = output_dir / "error_analysis_report.md"
            return True, f"Generated error analysis report at {report_path}"

        elif utility_name == "parse_tree_stats":
            from dojo.analysis_utils.meta_tree_analysis import generate_tree_reports_and_stats

            generate_tree_reports_and_stats(actual_exp_path)
            # Note: This utility saves reports within experiment folders and aggregated stats in meta_exp_path/tree_stats
            return True, f"Generated tree statistics and reports within experiment folders under {actual_exp_path}"

        elif utility_name == "parse_jsonlines_logs":
            # This utility now primarily converts JOURNAL.jsonl to journal.json within the exp folder.
            experiments = list_experiments(meta_exp_path)
            if not experiments:
                return False, "No experiments found in the meta experiment directory to parse logs for."

            from dojo.analysis_utils.journal_to_tree import visualise_all_trees

            # Process all experiments
            visualise_all_trees(actual_exp_path)  # This generates json and html trees for all
            return True, f"Generated journal.json and tree.html for all experiments in {actual_exp_path}"

        else:
            return False, f"Unknown or unsupported utility: {utility_name}"

    except ImportError as e:
        # Provide more context in the error message
        import traceback

        tb_str = traceback.format_exc()
        return False, f"Failed to import utility module for '{utility_name}': {str(e)}\nDetails:\n{tb_str}"
    except Exception as e:
        import traceback

        tb_str = traceback.format_exc()
        return False, f"Error executing utility '{utility_name}': {str(e)}\nDetails:\n{tb_str}"


@st.cache_data
def display_file_content(file_path: Path, max_lines: int = 1000, mode: str = "head") -> str:
    """Display the content of a file, supporting head, tail, or full (truncated) views."""
    if not file_path.exists():
        return "File not found"

    try:
        # Handle binary files like images differently
        if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".svg"]:
            # Return a special indicator or handle differently if needed
            # For now, let's return an indicator, display_image handles the actual display
            return f"Image file: {file_path.name}"

        # Try to read the file as text
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                total_lines = len(lines)

                if mode == "head":
                    display_lines = lines[:max_lines]
                    prefix = ""
                    suffix = (
                        f"\n... (showing first {len(display_lines)} of {total_lines} lines)"
                        if total_lines > max_lines
                        else ""
                    )
                elif mode == "tail":
                    display_lines = lines[-max_lines:]
                    prefix = (
                        f"... (showing last {len(display_lines)} of {total_lines} lines)\n"
                        if total_lines > max_lines
                        else ""
                    )
                    suffix = ""
                    # Note: Reading all lines for tail is inefficient for huge files.
                    # Consider alternative strategies if performance becomes an issue.
                else:  # mode == "full" or default
                    display_lines = lines[:max_lines]
                    prefix = ""
                    suffix = (
                        f"\n... (truncated, showing first {len(display_lines)} of {total_lines} lines)"
                        if total_lines > max_lines
                        else ""
                    )

                return prefix + "".join(display_lines) + suffix

        except UnicodeDecodeError:
            # This is likely a binary file
            return f"Error reading file: This appears to be a binary file and cannot be displayed as text."
        except Exception as read_e:
            return f"Error reading text file content: {str(read_e)}"

    except Exception as e:
        return f"Error accessing file: {str(e)}"


# display_image reads the file itself, caching might be complex depending on PIL/Streamlit internals.
# Let's leave it uncached for now, but it could be a target for optimization if image loading is slow.
def display_image(file_path: Path) -> None:
    """Display an image file using Streamlit's image component."""
    try:
        # Open the image file and display it
        import PIL.Image

        # Add a caption with image info
        img = PIL.Image.open(file_path)
        width, height = img.size
        size_kb = file_path.stat().st_size / 1024

        # Display image info
        st.caption(f"Image: {file_path.name} ({width}x{height}, {size_kb:.1f} KB)")

        # Create columns for display options and the image
        col1, col2 = st.columns([1, 3])

        with col1:
            # Add display options
            use_width = st.slider(
                "Width (%)", min_value=25, max_value=100, value=100, step=5, key=f"img_width_{hash(str(file_path))}"
            )

            # Option to fit image to container
            fit_to_container = st.checkbox(
                "Fit to container", value=False, key=f"fit_container_{hash(str(file_path))}"
            )

            # Download button for the image
            with open(file_path, "rb") as file:
                btn = st.download_button(
                    label="Download Image",
                    data=file,
                    file_name=file_path.name,
                    mime=f"image/{file_path.suffix.lower().strip('.')}",
                )

        with col2:
            # Display the image with the selected width
            display_width = None if fit_to_container else use_width * 8  # Scale factor for better sizing
            st.image(str(file_path), width=display_width, use_container_width=fit_to_container)

    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")
        st.warning("The file might not be a valid image or might be corrupted.")
