# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures
from datetime import datetime
import json
import curses
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import concurrent
import jsonlines

from dojo.core.solvers.utils import tree_export
from dojo.core.solvers.utils.journal import Journal
from dojo.config_dataclasses.run import RunConfig


def journal_log_into_json(
    file_path: Union[str, Path],
    seconds_cutoff: Optional[float | int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse a JSONL journal and return all nodes whose timestamp is within
    *seconds_cutoff* seconds of the first entry.

    Parameters
    ----------
    file_path
        Path (or str) to a *.jsonl* file whose lines look like::
            {"timestamp": "2025-05-06T10:32:19.614691", "data": {...}}

    seconds_cutoff
        ``None`` → keep the whole file (default).
        ``int`` / ``float`` → keep entries where
        ``(entry_ts - first_ts).total_seconds() <= seconds_cutoff``.
    """
    file_path = Path(file_path)
    if seconds_cutoff is not None and seconds_cutoff < 0:
        raise ValueError("seconds_cutoff must be non-negative or None")

    nodes: List[Dict[str, Any]] = []
    first_ts: Optional[datetime] = None

    with jsonlines.open(file_path) as reader:
        for entry in reader:
            ts_str = entry.get("timestamp")
            data = entry.get("data")

            # Skip malformed lines
            if ts_str is None or data is None:
                continue

            ts = datetime.fromisoformat(ts_str)  # "YYYY‑MM‑DDTHH:MM:SS.ssssss"

            if first_ts is None:
                first_ts = ts

            # Stop if we've exceeded the requested window
            if seconds_cutoff is not None and (ts - first_ts).total_seconds() > seconds_cutoff:
                break

            node = dict(data)  # shallow copy – don’t mutate caller’s dict
            node["timestamp"] = ts_str  # keep the original format
            nodes.append(node)

    return {"nodes": nodes}


def save_journal_log_as_json(
    file_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    output_filename: str = "journal.json",
    seconds_cutoff: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Reads a journal JSONL file, parses it into a structured tree, and saves it as a JSON file.
    The output file is saved in the experiment folder where the JSONL file is located.

    Args:
        file_path (Union[str, Path]): Path to the input journal JSONL file.
        output_dir (Optional[Union[str, Path]]): Directory where the output JSON file should be saved.
            Defaults to the experiment directory where the JSONL file is located.
        output_filename (str): Name of the output JSON file. Defaults to "journal.json".
        seconds_cutoff (Optional[float]): Time window in seconds to limit the entries.
            If None, all entries are included. Defaults to None.

    Returns:
        Dict[str, Any]: The parsed tree structure saved to the file.
    """
    file_path = Path(file_path)
    experiment_dir = file_path.parent.parent  # Navigate up to the experiment folder
    output_dir = Path(output_dir) if output_dir else experiment_dir
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    output_file = output_dir / output_filename

    parsed_json = journal_log_into_json(file_path, seconds_cutoff=seconds_cutoff)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_json, f, indent=2)

    print(f"Journal successfully saved to {output_file}")

    return parsed_json


def journal_into_tree(
    journal: Journal, exp_folder: Union[str, Path], tree_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Generates and exports a visualization of the journal tree from a journal instance.

    Args:
        journal (Journal): The journal instance to visualize.
        tree_path (Optional[Union[str, Path]]): Path where the visualization will be saved.
            Defaults to the experiment directory where the JSONL file is located.
        experiment_name (str): Name of the experiment for labeling purposes. Defaults to "tree".
    """
    if tree_path is None:
        tree_path = Path("./tree.html")

    exp_folder = Path(exp_folder)
    # Load Experiment config
    config_path = exp_folder / "dojo_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    cfg = RunConfig.load_from_json(config_path)

    tree_export.generate(cfg, journal, str(tree_path))


def json_into_tree(
    json_data: Dict[str, Any], exp_folder: Union[str, Path], tree_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Generates and exports a visualization of the journal tree from JSON data.

    Args:
        json_data (dict): The json data representing the journal instance to visualize.
        tree_path (Optional[Union[str, Path]]): Path where the visualization will be saved.
            Defaults to the experiment directory where the JSONL file is located.
        experiment_name (str): Name of the experiment for labeling purposes. Defaults to "tree".
    """
    # Load journal from structured JSON and generate tree visualization
    journal = Journal.from_export_data(json_data)

    journal_into_tree(journal, exp_folder, tree_path)


def log_to_tree(exp_folder: Path):
    log_dir = exp_folder / "json" / "JOURNAL.jsonl"
    json_data = save_journal_log_as_json(log_dir)
    output_file = exp_folder / "tree.html"
    json_into_tree(json_data, exp_folder, output_file)


def visualise_all_trees(meta_experiment_path: Path):
    """Concurrently processes multiple experiments and visualises trees."""
    meta_experiment_path = Path(meta_experiment_path)

    if not meta_experiment_path.exists():
        print(f"Experiments path does not exist: {meta_experiment_path}")
        return

    exp_fold = [folder for folder in meta_experiment_path.iterdir() if folder.is_dir()]

    errors_occurred = False

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_experiment = {executor.submit(log_to_tree, log_file): log_file for log_file in exp_fold}
        for future in concurrent.futures.as_completed(future_to_experiment):
            log_file = future_to_experiment[future]
            try:
                future.result()
                print(f"Processed experiment: {log_file}")
            except Exception as exc:
                errors_occurred = True
                print(f"Experiment {log_file} generated an exception: {exc}")

    if errors_occurred:
        print("\nProcessing completed with errors.")
    else:
        print("\nProcessing completed successfully.")


def main_menu(stdscr):
    curses.curs_set(0)
    current_row = 0
    menu = ["Visualize Single Experiment", "Visualize All Experiments", "Exit"]

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        for idx, row in enumerate(menu):
            x = w // 2 - len(row) // 2
            y = h // 2 - len(menu) // 2 + idx
            if idx == current_row:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, row)
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, row)

        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(menu) - 1:
            current_row += 1
        elif key in [curses.KEY_ENTER, ord("\n")]:
            if current_row == 0:
                curses.endwin()
                log_dir = input("Enter path to single journal log: ")
                log_to_tree(Path(log_dir))
                input("\nPress Enter to continue...")
            elif current_row == 1:
                curses.endwin()
                meta_path = input("Enter path to meta-experiment directory: ")
                visualise_all_trees(Path(meta_path))
                input("\nPress Enter to continue...")
            elif current_row == 2:
                break


def setup_curses(stdscr):
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    main_menu(stdscr)


def main():
    # argparse
    import argparse

    parser = argparse.ArgumentParser(description="Visualize journal logs.")
    parser.add_argument("log_dir", type=str, help="Path to the journal log directory", nargs="?")
    parser.add_argument("--all", action="store_true", help="Visualize all experiments")
    args = parser.parse_args()

    if not args.log_dir:
        curses.wrapper(lambda stdscr: setup_curses(stdscr))
        return

    log_dir = Path(args.log_dir)
    if args.all:
        visualise_all_trees(log_dir)
    else:
        log_to_tree(log_dir)


if __name__ == "__main__":
    main()
