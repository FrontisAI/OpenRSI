# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import streamlit as st
import pandas as pd
from pathlib import Path
import datetime
import mimetypes  # For detecting file types

from dojo.ui.components.exp_analysis import (
    analyze_meta_experiment,
    list_experiments,
    execute_utility,
)
from dojo.ui.components.exp_analysis import display_image, display_file_content
from dojo.utils.environment import get_log_dir
from dojo.utils.experiment_logs import is_meta_experiment


def initialize_session_state():
    """Initialize session state variables needed for the dashboard."""
    defaults = {
        "base_meta_exp_dir": os.path.join(get_log_dir(), "aira-dojo"),
        "found_meta_experiments": [],
        "selected_meta_exp": None,  # Name of the selected meta-experiment directory
        "meta_exp_data_path": "",  # Full path to data
        "selected_file": None,
        "file_preview_lines": 1000,
        "selected_specific_experiment": "-- All Experiments --",  # Default focus
        "meta_exp_selector": "-- Select --",  # Added key for selector default
        "scanned_base_dir_first_time_in": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def handle_meta_exp_change():
    selected_choice = st.session_state.meta_exp_selector
    base_dir = st.session_state.base_meta_exp_dir  # Assumes base_dir is stable in state
    if selected_choice != "-- Select --":
        base_path = os.path.join(st.session_state.base_meta_exp_dir, selected_choice)
        if os.path.isdir(base_path):
            st.session_state.meta_exp_data_path = base_path
            st.session_state.selected_meta_exp = selected_choice
            st.session_state.selected_file = None  # Reset file view
            st.session_state.selected_specific_experiment = "-- All Experiments --"  # Reset focus
            st.rerun()  # Rerun needed here to update tabs based on new path
        else:
            st.error("Invalid Meta Exp folder selected.")
            st.session_state.meta_exp_data_path = ""  # Clear path on error
    elif selected_choice == "-- Select --":
        # User deselected timestamp
        st.session_state.meta_exp_data_path = ""
        st.session_state.selected_file = None
        st.session_state.selected_specific_experiment = "-- All Experiments --"
        st.rerun()


def scan_base_directory():
    current_base_dir = st.session_state.base_meta_exp_dir  # Use state value
    if current_base_dir and os.path.isdir(current_base_dir):
        try:
            subdirs = sorted(
                d
                for d in os.listdir(current_base_dir)
                if os.path.isdir(os.path.join(current_base_dir, d))
                and is_meta_experiment(os.path.join(current_base_dir, d))
            )
            # All meta experiments should start with "user_" prefix, see run and runner config
            subdirs = [d for d in subdirs if is_meta_experiment]
            st.session_state.found_meta_experiments = subdirs
            if not subdirs:
                st.warning(f"No subdirectories found in '{current_base_dir}'")
            else:
                st.success(f"Found {len(subdirs)} potential meta experiments. Please select one below.")
            # Reset downstream selections on new scan
            st.session_state.selected_meta_exp = None
            st.session_state.meta_exp_selector = "-- Select --"  # Reset selector widget
            st.session_state.meta_exp_data_path = ""
            st.session_state.selected_file = None
            st.session_state.selected_specific_experiment = "-- All Experiments --"
            # No explicit rerun needed here, state change triggers it.
        except OSError as e:
            st.error(f"Error scanning directory '{current_base_dir}': {e}")
            st.session_state.found_meta_experiments = []
    else:
        st.error("Please enter a valid directory path.")


def main():
    st.set_page_config(page_title="Meta Experiment Data Dashboard", layout="wide")
    initialize_session_state()
    st.title("📂 Meta Experiment Data")

    # 1. Base directory input and scan
    base_dir = st.text_input(
        "Base Directory Path",
        value=st.session_state.base_meta_exp_dir,
        help="Path containing your meta experiment folders",
        key="base_meta_exp_dir_input",
        on_change=lambda: setattr(
            st.session_state, "base_meta_exp_dir", st.session_state.base_meta_exp_dir_input
        ),  # Ensure state updates
    )

    # Scan default base directory on first run
    if not st.session_state.scanned_base_dir_first_time_in:
        st.session_state.scanned_base_dir_first_time_in = True
        scan_base_directory()

    # 1. To Scan base directory and return meta experiments
    if st.button("Scan Base Directory", key="scan_meta_exp_base"):
        scan_base_directory()

    # 2. Meta experiment selection - Using the callback
    if st.session_state.found_meta_experiments:
        st.divider()
        options = ["-- Select --"] + st.session_state.found_meta_experiments

        # Determine index based on the *widget's* state key
        idx = 0
        if st.session_state.meta_exp_selector in options:
            idx = options.index(st.session_state.meta_exp_selector)

        st.selectbox(
            "Select Meta Experiment",
            options,
            index=idx,
            key="meta_exp_selector",  # Assign a key
            on_change=handle_meta_exp_change,  # Use the callback
        )
        # The logic is now handled by the callback, no need for if/else here
        # to process the selection immediately after the widget call.

    # 3. Main meta-experiment UI
    if st.session_state.meta_exp_data_path and os.path.isdir(st.session_state.meta_exp_data_path):
        exp_path = Path(st.session_state.meta_exp_data_path)
        st.divider()
        header = f"Meta Experiment: **`{st.session_state.selected_meta_exp}`**"
        st.subheader(header)

        # Discover specific experiments (assuming list_experiments is robust)
        try:
            experiments = list_experiments(exp_path)
        except Exception as e:
            st.error(f"Error listing specific experiments in {exp_path}: {e}")
            experiments = []

        exp_opts = ["-- All Experiments --"] + experiments
        cur_idx = 0
        # Use .get() for safety in case key doesn't exist yet
        current_specific_selection = st.session_state.get("selected_specific_experiment", "-- All Experiments --")
        if current_specific_selection in exp_opts:
            cur_idx = exp_opts.index(current_specific_selection)
        else:  # If previous selection is no longer valid, default to All
            st.session_state.selected_specific_experiment = "-- All Experiments --"
            cur_idx = 0

        # Key for specific experiment selector
        spec_exp_key = f"specific_exp_{st.session_state.meta_exp_data_path}"

        sel_exp = st.selectbox(
            "Focus on Specific Experiment (Optional)",
            exp_opts,
            index=cur_idx,
            key=spec_exp_key,
            # Add on_change if this becomes sticky too
            on_change=lambda: setattr(
                st.session_state, "selected_specific_experiment", st.session_state[spec_exp_key]
            ),
        )
        # Update state if needed (on_change handles direct update)
        # if sel_exp != current_specific_selection:
        #     st.session_state.selected_specific_experiment = sel_exp
        #     # Optionally reset file view when specific exp changes
        #     # st.session_state.selected_file = None
        #     # st.rerun() # Might not be needed if using on_change

        # Tabs for overview, list, tree, files, utilities
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📊 Overview", "📋 Experiments List", "🌳 Tree Visualization", "📁 File Explorer", "⚙️ Analysis Utilities"]
        )

        # Define current_focus_exp once for use in tabs
        current_focus_exp = st.session_state.selected_specific_experiment

        with tab1:
            st.subheader("Overview")
            # Ensure exp_path is valid before analyzing
            if exp_path.is_dir():
                stats = analyze_meta_experiment(exp_path)
                if "error" in stats:
                    st.error(stats["error"])
                else:
                    cols = st.columns(2)
                    with cols[0]:
                        st.metric("Total Experiments", stats.get("total_experiments", 0))
                        st.metric("With Config", stats.get("experiments_with_config", 0))
                    with cols[1]:
                        st.metric("With Journal", stats.get("experiments_with_journal", 0))
                        st.metric("Slurm Jobs", stats.get("slurm_job_count", "N/A"))
            else:
                st.warning(f"Cannot generate overview, path not found: {exp_path}")

        with tab2:
            st.subheader("Experiments List")
            if experiments:
                df = pd.DataFrame(experiments, columns=["Experiment Name"])
                st.dataframe(df, use_container_width=True)
                try:
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download List as CSV",
                        data=csv,
                        file_name=f"{st.session_state.selected_meta_exp}_experiments.csv",
                        mime="text/csv",
                        key=f"dl_csv_{st.session_state.selected_meta_exp}",
                    )
                except Exception as e:
                    st.error(f"Failed to generate CSV for download: {e}")
            else:
                st.info("No specific experiments found in this meta-experiment directory.")

        with tab3:  # Logic for the Tree Visualization tab
            st.subheader("Tree Visualization")
            # Use the centrally defined current_focus_exp
            if current_focus_exp != "-- All Experiments --":
                # Path relative to the *current* exp_path (which could be a timestamp folder)
                # Ensure exp_path is valid first
                if exp_path.is_dir():
                    # Construct path robustly
                    tree_html_path = exp_path.joinpath(current_focus_exp, "tree.html")

                    if tree_html_path.exists() and tree_html_path.is_file():
                        st.info(f"Displaying tree for experiment: `{current_focus_exp}`")
                        try:
                            with open(tree_html_path, "r", encoding="utf-8") as f:
                                tree_html_content = f.read()
                            st.components.v1.html(tree_html_content, height=800, scrolling=True)
                            # Add download button
                            st.download_button(
                                label="Download Tree HTML",
                                data=tree_html_content,
                                file_name=f"{current_focus_exp}_tree.html",
                                mime="text/html",
                                key=f"dl_tree_{current_focus_exp}",
                            )
                        except Exception as e:
                            st.error(f"Error reading or displaying {tree_html_path.name}: {e}")
                    else:
                        st.warning(
                            f"Tree visualization (`tree.html`) not found for experiment: `{current_focus_exp}` at `{tree_html_path}`"
                        )
                        st.info(
                            "This file can be generated by the 'Generate Trees' utility in the 'Analysis Utilities' tab."
                        )
                else:
                    st.warning(f"Cannot show tree, base path not found: {exp_path}")
            else:
                st.info("Select a specific experiment from the dropdown above to view its tree visualization here.")

        with tab4:  # File Explorer
            st.subheader("File Explorer")
            # Use the centrally defined current_focus_exp
            # Determine root based on focus, ensuring exp_path is valid
            if exp_path.is_dir():
                if current_focus_exp != "-- All Experiments --":
                    root = exp_path.joinpath(current_focus_exp)
                    if not root.is_dir():
                        st.error(f"Specific experiment directory not found: {root}")
                        root = None  # Prevent further processing
                else:
                    root = exp_path  # Root is the main meta-exp path
            else:
                root = None  # Base path invalid
                st.warning(f"Cannot explore files, base path not found: {exp_path}")

            if root and root.is_dir():  # Proceed only if root is valid
                # Add a filter box
                file_filter = st.text_input("Filter files (case-insensitive)", key=f"file_filter_{root}").lower()

                try:
                    # Use Path.rglob for cleaner recursion
                    files = [p for p in root.rglob("*") if p.is_file()]
                    all_rels = sorted([str(p.relative_to(root)) for p in files])
                except OSError as e:
                    st.error(f"Error listing files in '{root}': {e}")
                    all_rels = []
                except Exception as e:  # Catch other potential errors during listing
                    st.error(f"Unexpected error listing files: {e}")
                    all_rels = []

                # Apply filter
                if file_filter:
                    rels = [r for r in all_rels if file_filter in r.lower()]
                else:
                    rels = all_rels

                # Unique key for selectbox based on root and filter
                sel_file_key = f"sel_file_{root}_{file_filter}"
                sel_file = st.selectbox("Select File to View", ["--"] + rels, key=sel_file_key)

                if sel_file and sel_file != "--":
                    fpath = root / sel_file

                    # Show file stats (check existence again just in case)
                    if fpath.exists() and fpath.is_file():
                        try:
                            file_stats = fpath.stat()
                            size_kb = file_stats.st_size / 1024
                            size_mb = size_kb / 1024
                            modified_time = datetime.datetime.fromtimestamp(file_stats.st_mtime)

                            # Choose appropriate size unit
                            if size_mb >= 1:
                                size_str = f"{size_mb:.2f} MB"
                            elif size_kb >= 1:
                                size_str = f"{size_kb:.2f} KB"
                            else:
                                size_str = f"{file_stats.st_size} Bytes"

                            st.info(
                                f"📄 File: `{sel_file}`  |  📏 Size: {size_str}  |  🕒 Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}"
                            )

                            # Add download button for any file
                            with open(fpath, "rb") as file_download:
                                # Try to get mimetype
                                mime_type, _ = mimetypes.guess_type(fpath.name)
                                if not mime_type:
                                    mime_type = "application/octet-stream"  # Default binary type

                                st.download_button(
                                    label=f"📥 Download {fpath.name}",
                                    data=file_download,
                                    file_name=fpath.name,
                                    mime=mime_type,
                                    key=f"dl_btn_{fpath}",
                                )
                        except Exception as e:
                            st.error(f"Error getting file stats or preparing download: {e}")

                        # Determine file type for proper display
                        ext = fpath.suffix.lower()

                        # Handle image files
                        if ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"]:
                            try:
                                display_image(fpath)
                            except Exception as e:
                                st.error(f"Error displaying image {fpath.name}: {e}")
                        # Handle PDF (provide download link)
                        elif ext == ".pdf":
                            st.warning("PDF preview not supported. Use the download button above.")
                        # Handle other potentially non-text files best effort via download
                        elif ext in [".zip", ".gz", ".tar", ".parquet", ".pt", ".pth", ".onnx", ".pkl", ".ckpt"]:
                            st.info(f"Detected binary/archive file ({ext}). Use the download button above.")
                        else:
                            # Assume text-based file, add preview size control
                            preview_col1, preview_col2 = st.columns([3, 1])
                            with preview_col1:
                                # Use a more robust key for the slider
                                slider_key = f"preview_slider_{fpath}"
                                if slider_key not in st.session_state:
                                    st.session_state[slider_key] = (
                                        st.session_state.file_preview_lines
                                    )  # Initialize if needed

                                preview_lines = st.slider(
                                    "Preview lines",
                                    min_value=50,
                                    max_value=10000,
                                    value=st.session_state[slider_key],
                                    step=50,
                                    key=slider_key,
                                    on_change=lambda: setattr(
                                        st.session_state, "file_preview_lines", st.session_state[slider_key]
                                    ),
                                )
                                # Update global default if slider changes
                                # st.session_state.file_preview_lines = preview_lines # Done via on_change

                            with preview_col2:
                                mode_key = f"preview_mode_{fpath}"
                                if mode_key not in st.session_state:
                                    st.session_state[mode_key] = "head"

                                preview_mode = st.selectbox(
                                    "View mode",
                                    options=["head", "tail", "full"],
                                    index=["head", "tail", "full"].index(st.session_state[mode_key]),
                                    key=mode_key,
                                )

                            # Get content with selected parameters
                            try:
                                content = display_file_content(fpath, max_lines=preview_lines, mode=preview_mode)

                                # Determine language for syntax highlighting
                                language = None
                                # (Keep the extensive language detection)
                                if ext in [".py", ".pyx", ".pyd", ".pyi"]:
                                    language = "python"
                                elif ext in [".js", ".jsx", ".ts", ".tsx"]:
                                    language = "javascript"
                                elif ext in [".yaml", ".yml"]:
                                    language = "yaml"
                                elif ext in [".json", ".jsonl"]:
                                    language = "json"
                                elif ext in [".md", ".markdown"]:
                                    language = "markdown"
                                elif ext in [".html", ".htm"]:
                                    language = "html"
                                elif ext in [".css"]:
                                    language = "css"
                                elif ext in [".sh", ".bash", ".zsh"]:
                                    language = "bash"
                                elif ext in [".java", ".scala", ".kt"]:
                                    language = "java"
                                elif ext in [".c", ".cpp", ".h", ".hpp", ".cc"]:
                                    language = "c"
                                elif ext in [".rs"]:
                                    language = "rust"
                                elif ext in [".go"]:
                                    language = "go"
                                elif ext in [".sql"]:
                                    language = "sql"
                                elif ext in [".xml"]:
                                    language = "xml"
                                elif ext in [".r", ".R"]:
                                    language = "r"
                                elif ext in [".jl"]:
                                    language = "julia"
                                elif ext in [".log", ".txt", ""]:  # Treat no extension as text
                                    language = "text"  # Explicitly text

                                # Show content with syntax highlighting
                                st.code(content, language=language)

                                # Add copy button for text content
                                if content:
                                    # Use st.expander for a less intrusive copy button
                                    with st.expander("Copy Content"):
                                        st.code(content)  # Show again for context
                                        # Need a way to actually copy to clipboard - Streamlit doesn't have a direct widget.
                                        # Workaround: Show it in a text area for manual copy
                                        st.text_area(
                                            "Copy the text below:", content, height=150, key=f"copy_area_{fpath}"
                                        )
                            except Exception as e:
                                st.error(f"Error reading or displaying file content for {fpath.name}: {e}")
                                st.warning("The file might be binary or have encoding issues. Try downloading it.")
                    else:
                        st.warning(f"Selected file no longer exists or is not accessible: {fpath}")
                elif not rels and file_filter:  # No files match filter
                    st.info(f"No files found matching filter: '{file_filter}'")
                elif not rels and not file_filter:  # No files found at all
                    st.info("No files found in this directory.")

        with tab5:  # Analysis Utilities
            st.subheader("Analysis Utilities")
            # Ensure exp_path is valid before showing utilities
            if exp_path.is_dir():
                util_map = {
                    "log_error_parsing": "Generate Crash/Error Reports",
                    "parse_tree_stats": "Generate Tree Statistics",
                    "parse_jsonlines_logs": "Generate JSON/HTML Trees from Logs",
                }
                # Use a key based on the path to avoid state issues if path changes
                util_key = f"util_select_{exp_path}"
                selected_util = st.selectbox(
                    "Select Utility", list(util_map.keys()), format_func=lambda k: util_map[k], key=util_key
                )

                run_button_key = f"run_util_{exp_path}_{selected_util}"
                if st.button("Run Utility", key=run_button_key):
                    # Use a status indicator
                    with st.spinner(f"Running '{util_map[selected_util]}'..."):
                        try:
                            # Ensure output directory exists (use exp_path itself)
                            output_dir = exp_path
                            output_dir.mkdir(parents=True, exist_ok=True)

                            success, message = execute_utility(selected_util, exp_path, output_dir)
                            if success:
                                st.success(f"Utility finished: {message or 'Completed successfully.'}")
                            else:
                                st.error(f"Utility failed: {message or 'An unspecified error occurred.'}")
                        except Exception as e:
                            st.error(f"Error running utility: {e}")
                            st.code(traceback.format_exc())  # Provide traceback for debugging
            else:
                st.warning(f"Cannot run utilities, base path not found: {exp_path}")

    else:
        # More specific guidance
        if not st.session_state.base_meta_exp_dir:
            st.info("⬅️ Enter a Base Directory path above and click 'Scan Base Directory' to begin.")
        elif not st.session_state.found_meta_experiments:
            # This case might be brief due to rerun, but good to have
            st.info("⬅️ No experiments found in the scanned directory. Check the path or scan again.")
        elif not st.session_state.selected_meta_exp:
            st.info("⬅️ Select a Meta Experiment from the dropdown above.")
        # Fallback if state is somehow inconsistent
        else:
            st.info("⬅️ Select experiment to load data.")


if __name__ == "__main__":
    # Add traceback import if used in utilities section
    import traceback

    main()
