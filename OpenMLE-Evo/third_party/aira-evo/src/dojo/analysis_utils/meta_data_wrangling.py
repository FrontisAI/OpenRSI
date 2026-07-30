# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import concurrent.futures
import copy
import itertools
import json
import math
import os
import shutil
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Union, Optional, Any, Set
import pickle
import subprocess
import re
import mmap
from datetime import datetime, timedelta
from collections import deque
from omegaconf import OmegaConf

from mlebench.registry import Competition, Registry, registry

from dojo.config_dataclasses.run import RunConfig
from dojo.utils.experiment_logs import is_experiment, is_meta_experiment
from dojo.utils.environment import get_mlebench_data_dir
from dojo.analysis_utils.journal_to_tree import save_journal_log_as_json


# Section: Submitit Wrangling
#######################################


def metaexp2ids(metaexp_path: Path) -> List[str]:
    """
    Extract job IDs from a meta-experiment directory.

    Args:
        metaexp_path: Path to the meta-experiment directory

    Returns:
        List of job IDs
    """
    slurm_ids = [
        RunConfig.load_from_json(subdir / "dojo_config.json").metadata.slurm_id
        for subdir in metaexp_path.iterdir()
        if is_experiment(subdir)
    ]

    return slurm_ids


def run_bash_command(command: str) -> str:
    """
    Run a bash command and return the output.

    Args:
        command: Bash command to run

    Returns:
        Command output

    Raises:
        Exception: If command fails
    """
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        return result.stdout
    else:
        raise Exception(f"Error: {result.stderr}")


def render_output_paths(job: Dict[str, Any]) -> Tuple[str, str]:
    """
    Render the output paths for stdout and stderr based on job data.

    Args:
        job: Dictionary containing job data

    Returns:
        Tuple of (stdout_path, stderr_path)
    """

    def has_valid_pattern(pat):
        set_pat = set(pat)
        required_patterns = {"%a", "%A"}
        return set_pat.issubset(required_patterns) and set_pat.issubset(pat)

    std_out = job.get("stdout")
    stderr = job.get("stderr")
    pattern_stdout = re.findall(r"%[a-zA-Z]", std_out)
    pattern_stderr = re.findall(r"%[a-zA-Z]", stderr)
    if has_valid_pattern(pattern_stdout) and has_valid_pattern(pattern_stderr):
        base_job_id = job["array"]["job_id"]
        task_id = job["array"]["task_id"]["number"]
        std_out_path = std_out.replace("%A", str(base_job_id)).replace("%a", str(task_id))
        std_err_path = stderr.replace("%A", str(base_job_id)).replace("%a", str(task_id))
    else:
        # Fallback to the old way of getting stdout and stderr paths
        std_out_path = job.get("stdout_expanded")
        std_err_path = job.get("stderr_expanded")
    return std_out_path, std_err_path


def _get_job_data(job_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Get job data from Slurm for specified job IDs.

    Args:
        job_ids: List of job IDs

    Returns:
        Dictionary mapping job IDs to job data

    Raises:
        ValueError: If job data cannot be retrieved
    """
    # Build the comma-separated string of job IDs
    job_ids_str = ",".join(str(jid) for jid in job_ids)

    cmd = ["sacct", "-j", job_ids_str, "--format=ALL", "--json"]

    # Convert the command list to a string (if run_bash_command expects a string)
    cmd_str = " ".join(cmd)

    try:
        # Execute the sacct command
        result = run_bash_command(cmd_str)
        data = json.loads(result)
    except Exception as e:
        raise ValueError(f"Error retrieving job data: {e}")

    # Check that we received job information
    jobs = data.get("jobs")
    if not jobs:
        raise ValueError("No job information found in the JSON response.")

    results = {}

    for job in jobs:
        # Extract the basic fields from each job entry
        job_name = job.get("name")
        base_job_id = job["array"]["job_id"]
        task_id = job["array"]["task_id"]["number"]
        full_job_id = f"{base_job_id}_{task_id}"

        # Extract time information directly from JSON
        submit_time_epoch = job.get("time", {}).get("submission")
        start_time_epoch = job.get("time", {}).get("start")
        end_time_epoch = job.get("time", {}).get("end")
        elapsed_seconds = job.get("time", {}).get("elapsed")

        submit_str = datetime.fromtimestamp(submit_time_epoch).isoformat() if submit_time_epoch else "Unknown"
        start_str = datetime.fromtimestamp(start_time_epoch).isoformat() if start_time_epoch else "Unknown"

        elapsed_str = str(timedelta(seconds=elapsed_seconds)) if elapsed_seconds is not None else "Unknown"
        # job.get("stdout_expanded") and job.get("stderr_expanded") was not always giving me a valid path
        # So I will try, if possible to render the path myself. The reason is because there exist 2 ids:
        # job["array"]["job_id"] and job["job_id"]
        # These 2 don't always match and we use job["array"]["job_id"] for our stdout and stderr paths
        # but somehow slurm expands stdout_expanded and stderr_expanded with job["job_id"]
        # However, this is very specific to the current loggin path structure. So will attempt to find the pattern
        # and if not, I will resort to the old way of getting the stdout and stderr paths
        std_out_path, std_err_path = render_output_paths(job)

        # Calculate WaitingTime
        waiting_time_str = None
        if submit_time_epoch and start_time_epoch:
            try:
                dt_submit = datetime.fromtimestamp(submit_time_epoch)
                dt_start = datetime.fromtimestamp(start_time_epoch)
                waiting_delta = dt_start - dt_submit
                waiting_time_str = str(waiting_delta)
            except ValueError:  # Should not happen with timestamps, but good practice
                pass

        results[full_job_id] = {
            "job_name": job_name,
            "state": job.get("state"),
            "exit_code": job.get("derived_exit_code"),
            "stdout_path": std_out_path,  # See comment above where render_output_paths is called
            "stderr_path": std_err_path,  # See comment above where render_output_paths is called
            "JobID": full_job_id,  # Ensure JobID is included, matching previous _get_sacct_times output
            "Submit": submit_str,
            "Start": start_str,
            "Elapsed": elapsed_str,
            "WaitingTime": waiting_time_str,
        }

    return results


def get_slurm_data(job_ids: Union[str, List[str]], k: int = 10) -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive job data from Slurm for the specified job IDs.

    Args:
        job_ids: Job ID or list of job IDs
        k: Batch size for querying Slurm

    Returns:
        Dictionary mapping job IDs to job data
    """
    results = {}

    if isinstance(job_ids, str):
        job_ids = [job_ids]

    for i in range(0, len(job_ids), k):
        batch = job_ids[i : i + k]
        batch_results = _get_job_data(batch)
        results.update(batch_results)

    return results


def extract_path(file_path: Path) -> Optional[str]:
    """
    Extract output directory path from a log file.

    Args:
        file_path: Path to log file

    Returns:
        Extracted output directory path, or None if not found
    """
    matching_lines = []
    with open(file_path, "r") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            for line in iter(mm.readline, b""):
                if b"- Output dir:" in line:
                    matching_lines.append(line.rstrip(b"\n").decode("utf-8"))

    pattern = re.compile(r"Output dir:\s+(\/.*)")
    output_dirs = []

    for line in matching_lines:
        match = re.search(pattern, line)
        if match:
            path = match.group(1)
            # Strip ANSI color codes if present
            output_dirs.append(path.split("\x1b")[0])

    if len(output_dirs) == 0:
        print(f"WARNING: Output dir was not available in {str(file_path)}")
        return None

    return output_dirs[-1]


def get_submitit_logs2exp_mapping(mexp_dir: Union[str, Path]) -> Dict[str, str]:
    """
    Map submitit job IDs to experiment directories.

    Args:
        mexp_dir: Meta-experiment directory

    Returns:
        Dictionary mapping job IDs to experiment directories
    """
    job_id2experiment_id = {
        RunConfig.load_from_json(subdir / "dojo_config.json").metadata.slurm_id: subdir.relative_to(mexp_dir)
        for subdir in mexp_dir.iterdir()
        if is_experiment(subdir)
    }
    return job_id2experiment_id


def get_exp_name_from_cmd(cmd: RunConfig) -> str:
    """
    Extract experiment name from a command string.

    Args:
        cmd: Command string

    Returns:
        Experiment name

    Raises:
        Exception: If experiment name not found
    """

    return cmd.id


def get_competition_id_from_cmd(cmd: RunConfig) -> str:
    """
    Extract competition ID from a command string.

    Args:
        cmd: Command string

    Returns:
        Competition ID

    Raises:
        Exception: If competition ID not found
    """

    return cmd.task.name


def get_seed_from_cmd(cmd: RunConfig) -> str:
    """
    Extract seed from a command string.

    Args:
        cmd: Command string

    Returns:
        Seed value

    Raises:
        Exception: If seed not found
    """

    return cmd.metadata.seed


def get_submitit_stdout_stderr_paths(mexp_dir: Union[str, Path], job_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Get paths to stdout and stderr files for a job.

    Args:
        mexp_dir: Meta-experiment directory
        job_id: Job ID

    Returns:
        Tuple of (stdout_path, stderr_path), each may be None if not found
    """
    mexp_dir = Path(mexp_dir).resolve()
    std_out = None
    err_out = None

    slurm_logs_dir = mexp_dir.parent.parent / "slurm_logs" / job_id
    std_files = list(slurm_logs_dir.glob("*.out"))
    err_files = list(slurm_logs_dir.glob("*.err"))

    # used os instead of pathlib as they are not compatible with the relative_to method anymore
    if len(std_files) > 0:
        std_out = os.path.relpath(std_files[0], start=mexp_dir)
    if len(err_files) > 0:
        err_out = os.path.relpath(err_files[0], start=mexp_dir)

    return std_out, err_out


def link_jobs_to_experiments(mexp_dir: Union[str, Path]) -> Dict[str, Tuple[str, str, Optional[str], str]]:
    """
    Link Slurm jobs to experiment directories and extract relevant information.

    Args:
        mexp_dir: Meta-experiment directory

    Returns:
        Dictionary mapping job IDs to tuples of (cmd, exp_name, exp_dir, competition_id)
    """
    mexp_dir = Path(mexp_dir).resolve()
    submitit_logs_path = mexp_dir.parent.parent / "slurm_logs"
    job_id2exp_mapping = get_submitit_logs2exp_mapping(mexp_dir)

    pkl_files = list(submitit_logs_path.rglob("*submitted.pkl"))
    job_id2cmd__exp_name__exp_dir__competition_id = {}

    for pkl_path in pkl_files:
        job_id = pkl_path.parent.name
        if job_id not in job_id2exp_mapping.keys():
            continue

        with pkl_path.open("rb") as file:
            data = pickle.load(file)

        # Get the matching experiment directory
        cmd = data.args[0]
        competition_id = get_competition_id_from_cmd(cmd)
        exp_name = get_exp_name_from_cmd(cmd)
        # could have gotten it from cmd as well
        exp_dir = job_id2exp_mapping[job_id]

        job_id2cmd__exp_name__exp_dir__competition_id[job_id] = (cmd, exp_name, exp_dir, competition_id)

    return job_id2cmd__exp_name__exp_dir__competition_id


def try_get_clean_err_messsage(err_file: Path, k: int) -> Union[str, List[str]]:
    """
    Extract a clean error message from stderr logs.

    Args:
        err_file: Path to stderr file
        k: Maximum number of lines to analyze

    Returns:
        Clean error message or list of last lines if clean message not found
    """
    with err_file.open("r") as file:
        last_lines = deque(file, maxlen=k)

    results = []
    found = False
    for line in last_lines:
        if not found and line.startswith("Error executing job with overrides:"):
            found = True

        if found:
            results.append(line)
            if line.startswith("submitit ERROR ("):
                break

        if not found and line.startswith("submitit ERROR ("):
            found = True

    if len(results) == 0:
        print(f"WARNING: The start of the error message was not found in {str(err_file)}.")
        return last_lines

    return "".join(results)


def prepare_meta_exp_slurm_dataframe(metaexp_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Prepare a DataFrame with meta-experiment data.

    Args:
        metaexp_dir: Meta-experiment directory

    Returns:
        DataFrame with meta-experiment data
    """
    metaexp_dir = Path(metaexp_dir).resolve()

    # Get job IDs
    ids = metaexp2ids(metaexp_dir)
    print(f"Processing {len(ids)} array jobs associated with job id {ids[0].split('_')[0]}")

    # Get job data
    data = get_slurm_data(ids)
    df = pd.DataFrame(data).T

    df = df.reset_index()
    df["taskID"] = df["index"].str.split("_").str[1]
    df = df.set_index("taskID")
    df.drop(columns="index", inplace=True)

    # Validate job IDs and names
    job_ids = df["JobID"].apply(lambda x: x.split("_")[0]).unique()
    assert len(job_ids) == 1, f"Not all jobs have the same base job id: {str(job_ids)}"
    print("Base job ID:", job_ids[0])

    job_names = df["job_name"].unique()
    assert len(job_names) == 1, f"Not all jobs have the same job name: {str(job_names)}"
    mexp_name = job_names[0]
    print("Meta exp name:", mexp_name)
    df.drop(columns="job_name", inplace=True)

    # Clean up DataFrame
    df_clean = df.copy()

    # Add submitit stdout/stderr paths
    std_outs = []
    err_outs = []

    for job_id in df_clean["JobID"]:
        std_out, err_out = get_submitit_stdout_stderr_paths(metaexp_dir, job_id)
        std_outs.append(std_out)
        err_outs.append(err_out)

    df_clean["SubmititStdOut"] = std_outs
    df_clean["SubmititStdErr"] = err_outs

    # Parse state
    df_clean["State"] = df["state"].apply(lambda x: x["current"][0])
    df_clean.rename(columns={"state": "state_dict"}, inplace=True)
    df_clean.rename(columns={"WaitingTime": "Waiting"}, inplace=True)

    columns_to_show = ["JobID", "State", "Elapsed", "Waiting", "SubmititStdOut", "SubmititStdErr"]
    df_clean = df_clean[columns_to_show]
    df_clean.index = pd.to_numeric(df_clean.index)
    df_clean.sort_index(inplace=True)

    # Link jobs to experiments
    job_id2info = link_jobs_to_experiments(metaexp_dir)

    df_clean["CompID"] = df_clean["JobID"].apply(lambda job_id: job_id2info[job_id][3])
    df_clean["ExpDir"] = df_clean["JobID"].apply(lambda job_id: Path(job_id2info[job_id][2]).resolve().stem)
    df_clean["Cmd"] = df_clean["JobID"].apply(lambda job_id: job_id2info[job_id][0])

    return df_clean


def analyze_failures(
    df_clean: pd.DataFrame, mexp_dir: Union[str, Path], std_err_k: int = 300, std_out_k: int = 50
) -> pd.DataFrame:
    """
    Analyze failures in the meta-experiment.

    Args:
        df_clean: DataFrame with meta-experiment data
        mexp_dir: Meta-experiment directory
        std_err_k: Maximum number of stderr lines to analyze
        std_out_k: Maximum number of stdout lines to analyze

    Returns:
        DataFrame with failure analysis added
    """
    mexp_dir = Path(mexp_dir).resolve()
    df_with_logs = df_clean.copy()

    std_outs, std_errs, state_logs = [], [], []

    for index, row in df_with_logs.iterrows():
        if row["State"] != "FAILED":
            std_outs.append("/")
            std_errs.append("/")
            state_logs.append(row["State"])
            continue

        job_id = row["JobID"]
        comp_id = row["CompID"]

        # Retrieve the first .err and .out files from the slurm logs directory
        err_file = mexp_dir / row["SubmititStdErr"]
        std_file = mexp_dir / row["SubmititStdOut"]

        std_err_message = try_get_clean_err_messsage(err_file, std_err_k)
        std_errs.append(std_err_message)

        # Get the standard output details
        with std_file.open("r") as file:
            last_stdout_lines = deque(file, maxlen=std_out_k)
        std_outs.append(last_stdout_lines)

        if (
            isinstance(std_err_message, str)
            and "UncompletedJobError: Job not requeued because: timed-out" in std_err_message
        ):
            state_logs.append("TIMEOUT")
        else:
            state_logs.append(row["State"])

    df_with_logs["StdOut_tail"] = std_outs
    df_with_logs["StdErr_tail"] = std_errs
    df_with_logs["StateFromLogs"] = state_logs

    return df_with_logs


def print_failure_summary(df_with_logs: pd.DataFrame) -> None:
    """
    Print a summary of failures in the meta-experiment.

    Args:
        df_with_logs: DataFrame with failure analysis
    """
    print("State distribution by competition ID:")
    print(pd.crosstab(df_with_logs["CompID"], df_with_logs["StateFromLogs"]))


def print_detailed_failures(df_with_logs: pd.DataFrame, show_std_err: bool = True, show_std_out: bool = False) -> None:
    """
    Print detailed information about failed jobs.

    Args:
        df_with_logs: DataFrame with failure analysis
        show_std_err: Whether to show stderr logs
        show_std_out: Whether to show stdout logs
    """
    df_failed = df_with_logs[df_with_logs["StateFromLogs"] == "FAILED"].copy()
    df_failed.sort_values(by=["CompID", "taskID"], inplace=True)

    print("Number of failed jobs:", len(df_failed))
    for index, row in df_failed.iterrows():
        job_id = row["JobID"]
        comp_id = row["CompID"]

        # Create a header for the job log output
        header = f"Job ID: {job_id} | Competition ID: {comp_id}"
        separator = "=" * len(header)
        print(separator)
        print(header)
        print(separator)

        # If enabled, print the standard error details with clear formatting
        if show_std_err:
            print("\n")
            std_err_header = "~~~~ Standard Error ~~~~"
            separator = "~" * len(std_err_header)
            print(separator)
            print(std_err_header)
            print(separator)
            print("\n")

            print(row["StdErr_tail"])

        # If enabled, print the standard output details with clear formatting
        if show_std_out:
            print("\n~~~~ Standard Output ~~~~")
            separator = "~" * len("~~~~ Standard Output ~~~~")
            print(separator)
            print(row["StdOut_tail"])

        # Add spacing after each job's logs for clarity
        print("\n" + "-" * len(header) + "\n")


def analyze_slurm_meta_exp(
    metaexp_dir: Union[str, Path], show_failures: bool = True, show_std_err: bool = True, show_std_out: bool = False
) -> pd.DataFrame:
    """
    Analyze a meta-experiment.

    Args:
        metaexp_dir: Meta-experiment directory
        show_failures: Whether to show failure summary
        show_std_err: Whether to show stderr logs for failed jobs
        show_std_out: Whether to show stdout logs for failed jobs

    Returns:
        DataFrame with meta-experiment analysis based on slurm.
    """
    metaexp_dir = Path(metaexp_dir).resolve()

    # Prepare DataFrame
    df_clean = prepare_meta_exp_slurm_dataframe(metaexp_dir)

    # Analyze failures
    df_with_logs = analyze_failures(df_clean, metaexp_dir)

    # Print summary
    print("\nSummary:")
    print_failure_summary(df_with_logs)

    # Print detailed failures if requested
    if show_failures:
        print("\nDetailed failures:")
        print_detailed_failures(df_with_logs, show_std_err, show_std_out)

    return df_with_logs


def analyze_all_slurm_meta_exps(meta_experiment_paths: List[Union[str, Path]]) -> pd.DataFrame:
    """
    Analyze multiple meta-experiments based on slurm.

    Args:
         metaexp_dirs: List of meta-experiment directories
    """

    dfs = []
    # Resolve paths first
    resolved_paths = [Path(meta_exp).resolve() for meta_exp in meta_experiment_paths]

    # Use ProcessPoolExecutor to parallelize the analysis
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks for each meta experiment path
        future_to_path = {
            executor.submit(
                analyze_slurm_meta_exp, path, show_failures=False, show_std_err=False, show_std_out=False
            ): path
            for path in resolved_paths
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    dfs.append(df)
            except Exception as exc:
                print(f"Error processing {path}: {exc}")

    if not dfs:
        print("Warning: No valid DataFrames were generated from the meta-experiments.")
        return pd.DataFrame()  # Return an empty DataFrame if no results

    final_df = pd.concat(dfs, ignore_index=True)
    final_df.reset_index(drop=True, inplace=True)
    # Ensure index is numeric before trying to sort. Check if index is already numeric.
    if not pd.api.types.is_numeric_dtype(final_df.index):
        final_df.index = pd.to_numeric(final_df.index, errors="coerce")
        # Handle potential NaNs introduced by coercion if necessary, e.g., drop or fill them
        final_df.dropna(axis=0, subset=[final_df.index.name], inplace=True)

    final_df.sort_index(inplace=True)
    return final_df


# Section: Extraction of data into dataframe
################################################


def process_experiment_folder_journal_log(
    experiment_path: Path, regenerate_tree: bool, seconds_cutoff: float = None
) -> Union[Path, None]:
    """
    Processes a single experiment folder:
        1. converts JOURNAL.jsonl -> journal_tree.json
        2. returns the path to the new file if successful, else None.
    """
    jsonl_path = experiment_path / "json" / "JOURNAL.jsonl"
    json_path = experiment_path / "journal.json"
    if jsonl_path.exists():
        if (not json_path.exists()) or regenerate_tree:
            print(f"Processing {jsonl_path} -> {json_path}")
            save_journal_log_as_json(jsonl_path, experiment_path, "journal.json", seconds_cutoff)
        return json_path
    else:
        print(f"JOURNAL.jsonl not found in {experiment_path}")
        return None


def load_competition_id(config_file: Union[str, Path]) -> str:
    """
    Reads the .hydra/config.yaml file and returns the competition_id
    under the 'task' field.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("task", {}).get("competition_id", None)


def load_seed(config_file: Union[str, Path]) -> str:
    """
    Reads the .hydra/config.yaml file and returns the seed of the experiment.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("seed", -1)


def load_method_name(meta_experiment_path: Path, path_to_method_name: Dict[str, str]) -> str:
    """
    Returns the method name for a given meta_experiment_path
    from the path_to_method_name mapping.
    """
    return path_to_method_name[str(meta_experiment_path)]


def _load_experiment_data(
    meta_path: Path, experiment_folder: Path, path_to_method_name: Dict[str, str]
) -> Dict[str, Any]:
    """
    Loads data for a single experiment folder in parallel:
      - method_name (from path_to_method_name)
      - competition_id (from .hydra/config.yaml if present)
      - experiment_name (folder name)
      - nodes (from journal.json)
    Returns a dict of the form:
      {
        "method": ...,
        "competition_id": ...,
        "experiment_name": ...,
        "nodes": [...]
      }
    or None if journal.json doesn't exist or folder is invalid.
    """
    # Identify method name
    method_name = load_method_name(meta_path, path_to_method_name)

    # Attempt to load competition_id and seed
    config_file = experiment_folder / "dojo_config.json"
    competition_id = None
    seed = None
    if config_file.is_file():
        cfg = RunConfig.load_from_json(config_file)
        competition_id = cfg.task.name
        seed = cfg.metadata.seed

    # Load journal.json
    tree_json_path = experiment_folder / "journal.json"
    if not tree_json_path.is_file():
        return None  # skip if missing

    with open(tree_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 'nodes' should be a list of step dictionaries
    nodes = data.get("nodes", [])

    return {
        "method": method_name,
        "competition_id": competition_id,
        "experiment_name": experiment_folder.name,
        "nodes": nodes,
        "seed": seed,
        "ExpDir": str(experiment_folder.stem),
    }


def process_all_meta_experiments(
    meta_experiment_paths: List[Path],
    regenerate_trees: bool,
    max_processes: int | None = None,
    seconds_cutoff: float = None,
):
    """
    Phase A:
        1. Get all experiment folders in all provided metaexperiment folders.
        2. Creates one ProcessPoolExecutor for all experiment folders to
        converting JOURNAL.jsonl -> journal.json in parallel.
    """
    all_exp_dirs = []
    for meta_exp in meta_experiment_paths:
        meta_path = Path(meta_exp)
        if not is_meta_experiment(meta_path):
            print(f"Skipping invalid meta path: {meta_path}")
            continue
        # Collect all experiment dirs in this meta path
        for exp_dir in meta_path.iterdir():
            if not is_experiment(exp_dir):
                continue
            # Check if there has been a mistake and a dir was created.
            if Path(exp_dir / "journal.json").is_dir():
                shutil.rmtree(Path(exp_dir / "journal.json"))

            check_to_generate = regenerate_trees or (not (exp_dir / "journal.json").exists())
            if check_to_generate:
                all_exp_dirs.append(exp_dir)

    # Now use a single pool for all experiment folders
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        futures = {
            executor.submit(process_experiment_folder_journal_log, d, regenerate_trees, seconds_cutoff): d
            for d in all_exp_dirs
        }
        for future in concurrent.futures.as_completed(futures):
            exp_dir = futures[future]
            try:
                result = future.result()
                if result:
                    print(f"Created: {result}")
            except Exception as exc:
                print(f"{exp_dir} generated an exception: {exc}")


def gather_all_meta_experiment_data(
    meta_experiment_paths: List[Union[str, Path]],
    path_to_method_name: dict,
    max_steps_cap: int = 500,
    max_processes: int | None = None,
):
    """
    Phase B: Iterates through each meta experiment folder and each experiment inside,
    but uses a single ProcessPoolExecutor to load them in parallel.
    Collects:
       - method_name,
       - competition_id,
       - experiment_name,
       - 'nodes' (list of step dictionaries).
    Returns:
       all_experiments, all_keys, overall_max_steps
    """
    # 1. Collect all (meta_path, experiment_folder) pairs
    all_folders = []
    for meta_exp in meta_experiment_paths:
        meta_path = Path(meta_exp)
        if not meta_path.is_dir():
            print(f"Skipping invalid meta experiment path: {meta_path}")
            continue

        experiments_dir = meta_path
        if not experiments_dir.is_dir():
            print(f"Warning: no 'experiments' folder found in {meta_path}")
            continue

        for experiment_folder in experiments_dir.iterdir():
            if is_experiment(experiment_folder):
                all_folders.append((meta_path, experiment_folder))
    # 2. Parallel load each experiment folder
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        future_to_folder = {
            executor.submit(_load_experiment_data, meta_path, folder, path_to_method_name): (meta_path, folder)
            for (meta_path, folder) in all_folders
        }

        for future in concurrent.futures.as_completed(future_to_folder):
            meta_path, folder = future_to_folder[future]
            try:
                exp_data = future.result()
                if exp_data is not None:
                    results.append(exp_data)
            except Exception as exc:
                print(f"Error loading {folder}: {exc}")

    # 3. Aggregate results -> all_experiments, all_keys, overall_max_steps
    all_experiments = []
    all_keys = set()
    overall_max_steps = 0

    for exp_data in results:
        nodes = exp_data["nodes"]
        step_count = len(nodes)
        if step_count > overall_max_steps:
            overall_max_steps = step_count

        # Collect all node keys
        for node_dict in nodes:
            all_keys.update(node_dict.keys())

        # Append to final experiments list
        all_experiments.append(exp_data)

    # 4. Cap the maximum steps if it exceeds max_steps_cap
    if overall_max_steps > max_steps_cap:
        overall_max_steps = max_steps_cap

    return all_experiments, all_keys, overall_max_steps


def build_dataframe_one_row_per_experiment(all_experiments: List[dict], all_keys: set, max_steps: int) -> pd.DataFrame:
    """
    Given:
      - all_experiments: list of experiment dicts (method, competition_id, experiment_name, nodes)
      - all_keys: union of all fields present in any node
      - max_steps: the maximum number of steps to store per experiment (lists padded to this length)

    Returns a DataFrame with:
      - columns for method, competition_id, experiment_name
      - columns for each node field in all_keys, whose values are lists of length max_steps
        (padded with NaN if an experiment has fewer than max_steps).
    """
    rows = []

    for exp in all_experiments:
        method_name = exp["method"]
        competition_id = exp["competition_id"]
        seed = exp["seed"]
        experiment_name = exp["experiment_name"]
        nodes = exp["nodes"]
        experiment_dir = exp["ExpDir"]

        # Prepare the row dict, starting with metadata
        row_data = {
            "method": method_name,
            "competition_id": competition_id,
            "experiment_name": experiment_name,
            "seed": seed,
            "ExpDir": experiment_dir,
        }

        # For each possible node field, create a list of length max_steps
        row_data.update({key: [np.nan] * max_steps for key in all_keys})

        # Fill in node field values for each step, up to max_steps
        for i, node_dict in enumerate(nodes):
            if i >= max_steps:
                break
            for key, val in node_dict.items():
                row_data[key][i] = val if val is not None else np.nan

        # Optionally convert numeric lists to arrays
        for key in row_data:
            val = row_data[key]
            if isinstance(val, list) and val and isinstance(val[0], (int, float, bool, np.number)):
                row_data[key] = np.asarray(val)
            else:
                row_data[key] = val

        # Get padding masks
        used_steps = min(len(nodes), max_steps)
        padding_mask = np.zeros(max_steps, dtype=bool)
        padding_mask[used_steps:] = True
        row_data["padding_mask"] = padding_mask
        row_data["non_padding_mask"] = np.logical_not(padding_mask)

        rows.append(row_data)

    # Build the final DataFrame
    df = pd.DataFrame(rows)
    return df


def collect_all_meta_experiments_in_one_df(
    meta_experiment_paths: List[Union[str, Path]],
    path_to_method_name: dict,
    max_steps_cap: int = 500,
    regenerate_trees: bool = False,
    seconds_cutoff: float = None,
    max_processes: int | None = None,
) -> pd.DataFrame:
    """
    Performs Phase A: Processing meta experiments to generate trees.
    Performs Phase B: Gathers meta experiments data.
    Performs Phase C: Processing each experiment to pad the series and
        build data frame with one row per experiment.
    """
    # Phase A: Process all the files to generate trees
    process_all_meta_experiments(meta_experiment_paths, regenerate_trees, max_processes, seconds_cutoff)

    # Phase B: Gather all the tree data
    all_experiments, all_keys, overall_max_steps = gather_all_meta_experiment_data(
        meta_experiment_paths,
        path_to_method_name,
        max_steps_cap=max_steps_cap,
        max_processes=max_processes,
    )

    # Build the final DataFrame with one row per experiment
    final_df = build_dataframe_one_row_per_experiment(all_experiments, all_keys, max_steps_cap)

    # Get df with slurm data
    meta_experiment_paths = [Path(path).resolve() for path in meta_experiment_paths]
    slurm_df = analyze_all_slurm_meta_exps(meta_experiment_paths)
    # # Join the data frame rows based on ExpDir
    final_df = final_df.merge(slurm_df, on="ExpDir", how="outer", validate="one_to_one")

    return final_df


def filter_dataframe_based_on_slurm(df: pd.DataFrame):
    df = df.copy()

    print("Original number of rows:", len(df))

    # Filter all rows where the method is nan AKA the experiment failed extremely early
    # and the journal.json file was not created.
    before_method_filter = len(df)
    df = df[~pd.isna(df["method"])]
    after_method_filter = len(df)
    print(f"Filtered {before_method_filter - after_method_filter} rows based on early failure.")
    # Filter all rows where the state is FAILED
    before_state_filter = len(df)
    df = df[df["StateFromLogs"] != "FAILED"]
    df = df[df["StateFromLogs"] != "NODE_FAIL"]
    df = df[df["StateFromLogs"] != "NODE_FAILED"]
    after_state_filter = len(df)
    print(f"Filtered {before_state_filter - after_state_filter} rows based on state failure.")

    print("Number of rows after filtering based on slurm:", len(df))

    return df


def filter_dataframe_based_on_node_to_node_elapsed(
    df: pd.DataFrame,
    elapsed_time: str = "0-23:00:00",
    *,
    length_cutoff: int | None = None,
    show_filtered_times: bool = True,
    max_examples: int = 5,
) -> pd.DataFrame:
    """
    Drops rows whose first node to last node duration < `elapsed_time`.

    Examples that survive either because they are long enough
    or because they ran long enough are retained.
    """
    df = df.copy()
    df["length"] = df["step"].apply(lambda x: int(pd.Series(x).max() + 1))

    elapsed_td = pd.to_timedelta(elapsed_time.replace("-", " days "))

    def get_timestamp_diff(row):
        timestamp_list = list(itertools.compress(row["timestamp"], row["non_padding_mask"]))
        start = pd.to_datetime(timestamp_list[0])
        end = pd.to_datetime(timestamp_list[-1])
        return end - start

    df["first_node_to_last_node_elapsed"] = df.apply(lambda x: get_timestamp_diff(x), axis=1)
    if length_cutoff is None:
        mask_to_drop = df["first_node_to_last_node_elapsed"] < elapsed_td
    else:
        short_enough = df["length"] < length_cutoff
        ran_too_quick = df["first_node_to_last_node_elapsed"] < elapsed_td
        mask_to_drop = short_enough & ran_too_quick

    n_dropped = mask_to_drop.sum()

    if show_filtered_times and n_dropped:
        dropped_examples = (
            df.loc[
                mask_to_drop,
                ["length", "Elapsed", "first_node_to_last_node_elapsed", "method", "seed", "competition_id"],
            ]
            .head(max_examples)
            .apply(
                lambda r: f"length={r.length}, slurm_elapsed={r.Elapsed}, first_node_to_last_node_elapsed={r.first_node_to_last_node_elapsed}, method={r.method}, seed={r.seed}, 'comp={r.competition_id}",
                axis=1,
            )
            .tolist()
        )
        example_str = ";  ".join(dropped_examples)
        more = " …" if n_dropped > max_examples else ""
        print(f"Dropped rows first_node_to_last_node_elapsed < {elapsed_td}): {example_str}{more}")

    df = df[~mask_to_drop]
    print(f"Filtered {n_dropped} rows")
    print("Remaining rows:", len(df))
    return df


def filter_dataframe_based_on_slurm_elapsed(
    df: pd.DataFrame,
    elapsed_time: str = "0-23:00:00",
    *,
    length_cutoff: int | None = None,  # NEW — e.g. 200
    show_filtered_times: bool = True,
    max_examples: int = 5,
) -> pd.DataFrame:
    """
    • Drops rows where 'method' is NaN.
    • Drops rows whose 'length' < `length_cutoff`
      *and* whose 'Elapsed' duration < `elapsed_time`.

    Examples that survive either because they are long enough
    or because they ran long enough are retained.
    """
    df = df.copy()
    print("Original rows:", len(df))

    # -- 1. basic validity check ------------------------------------------------
    df = df[~df["method"].isna()]
    print("After method NaN filter:", len(df))

    # -- 2. calculate 'length' ---------------------------------------------------
    df["length"] = df["step"].apply(lambda x: int(pd.Series(x).max() + 1))

    # -- 3. normalise 'Elapsed' to timedelta ------------------------------------
    elapsed_td = pd.to_timedelta(elapsed_time.replace("-", " days "))
    if pd.api.types.is_numeric_dtype(df["Elapsed"]):
        df["Elapsed"] = pd.to_timedelta(df["Elapsed"], unit="s")
    else:
        df["Elapsed"] = pd.to_timedelta(df["Elapsed"].astype(str).str.replace("-", " days ", regex=False))

    # -- 4. build mask -----------------------------------------------------------
    if length_cutoff is None:
        mask_to_drop = df["Elapsed"] < elapsed_td
    else:
        short_enough = df["length"] < length_cutoff
        ran_too_quick = df["Elapsed"] < elapsed_td
        mask_to_drop = short_enough & ran_too_quick

    n_dropped = mask_to_drop.sum()

    # -- 5. (optional) show examples --------------------------------------------
    if show_filtered_times and n_dropped:
        dropped_examples = (
            df.loc[mask_to_drop, ["length", "Elapsed", "method", "seed", "competition_id"]]
            .head(max_examples)
            .apply(
                lambda r: f"length={r.length}, elapsed={r.Elapsed}, {r.method}, {r.competition_id}, {r.seed}", axis=1
            )
            .tolist()
        )
        example_str = ";  ".join(dropped_examples)
        more = " …" if n_dropped > max_examples else ""
        print(f"Dropped rows (length < {length_cutoff} & elapsed < {elapsed_td}): {example_str}{more}")

    # -- 6. drop and report ------------------------------------------------------
    df = df[~mask_to_drop]
    print(f"Filtered {n_dropped} rows with combined condition")
    print("Remaining rows:", len(df))
    return df


def _sentinel(lower_is_better: bool) -> float:
    """
    Returns a numeric sentinel:  +1e12   if lower scores are better,
                                -1e12   if higher scores are better.
    Using a single constant makes post‑hoc filtering easy.
    """
    return 1e12 if lower_is_better else -1e12


def filter_dataframe_based_on_data_validity(df: pd.DataFrame, min_length: int = 5) -> pd.DataFrame:
    """
    Ensures that every (competition, method, seed) group contains at least one valid node.
    If **all** nodes in a group are invalid, their `metric` and `metric_info/score`
    arrays are overwritten with sentinel values so that the group is always treated
    as the *worst possible* submission in downstream ranking code.

    Parameters
    ----------
    df : pd.DataFrame
        Expected columns (same as in the original code):
            * step – list/array of ints
            * metric – list/array (validation score per node)
            * metric_info/score – list/array (test score per node)
            * is_buggy – list/array (bool or 0/1 per node)
            * competition_id, method, seed – scalars
    Returns
    -------
    pd.DataFrame
        A **copy** of the input with extra statistics columns and sentinel‑fixed rows.
    """

    df = df.copy()  # keep original intact
    print("Original number of rows:", len(df))
    df["length"] = df["step"].apply(lambda x: int(pd.Series(x).max() + 1))
    df["num_validation_nodes"] = df["metric"].apply(lambda x: np.count_nonzero(~np.isnan(np.asarray(x, dtype=float))))
    df["num_test_nodes"] = df["metric_info/score"].apply(
        lambda x: np.count_nonzero(~np.isnan(np.asarray(x, dtype=float)))
    )
    df["num_buggy_nodes"] = df["is_buggy"].apply(
        lambda x: int(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0).sum())
    )
    df["num_good_nodes"] = df["length"] - df["num_buggy_nodes"]
    df["good_node_proportion"] = df["num_good_nodes"] / df["length"]
    df["max_test_score"] = df["metric_info/score"].apply(
        lambda x: np.nan_to_num(np.asarray(x, dtype=float), nan=-1e12).max()
    )
    df["min_test_score"] = df["metric_info/score"].apply(
        lambda x: np.nan_to_num(np.asarray(x, dtype=float), nan=+1e12).min()
    )
    df["max_validation_score"] = df["metric"].apply(
        lambda x: np.nan_to_num(np.asarray(x, dtype=float), nan=-1e12).max()
    )
    df["min_validation_score"] = df["metric"].apply(
        lambda x: np.nan_to_num(np.asarray(x, dtype=float), nan=+1e12).min()
    )

    df["any_valid_node"] = df.apply(lambda row: bool(np.any(get_selection_mask(row))), axis=1)

    # print the number of rows with no valid nodes
    num_no_valid_nodes = len(df[df["any_valid_node"] == False])
    print(f"Number of rows with no valid nodes: {num_no_valid_nodes}")

    # Overwrite groups with no valid node
    new_registry = registry.set_data_dir(Path(get_mlebench_data_dir()))

    # Iterate over (competition, method, seed) groups
    print("Overwriting rows with no valid nodes to have worst score in competition...")
    group_cols = ["competition_id", "method", "seed"]
    for keys, group_idx in df.groupby(group_cols).groups.items():
        comp_id, method_name, seed_val = keys
        group_rows = df.loc[group_idx]

        # Quick exit if group already has ≥1 valid node
        if group_rows["any_valid_node"].any():
            continue

        # Determine optimisation direction for this competition
        comp_obj: Any = new_registry.get_competition(comp_id)
        lower_is_better: bool = is_lower_better(comp_obj)
        sentinel = _sentinel(lower_is_better)

        # ------------------------------------------------------------------
        # Actually overwrite the columns for every row in this group
        # ------------------------------------------------------------------
        # Set validation and test scores to the sentinel value
        df.loc[group_idx, "metric"] = df.loc[group_idx, "metric"].apply(
            lambda arr: np.full_like(arr, sentinel, dtype=float)
        )
        df.loc[group_idx, "metric_info/score"] = df.loc[group_idx, "metric_info/score"].apply(
            lambda arr: np.full_like(arr, sentinel, dtype=float)
        )
        # Set the medal flags to 0
        df.loc[group_idx, "metric_info/above_median"] = df.loc[group_idx, "metric_info/above_median"].apply(
            lambda arr: np.full_like(arr, 0, dtype=float)
        )
        df.loc[group_idx, "metric_info/any_medal"] = df.loc[group_idx, "metric_info/any_medal"].apply(
            lambda arr: np.full_like(arr, 0, dtype=float)
        )
        df.loc[group_idx, "metric_info/bronze_medal"] = df.loc[group_idx, "metric_info/bronze_medal"].apply(
            lambda arr: np.full_like(arr, 0, dtype=float)
        )
        df.loc[group_idx, "metric_info/silver_medal"] = df.loc[group_idx, "metric_info/silver_medal"].apply(
            lambda arr: np.full_like(arr, 0, dtype=float)
        )
        df.loc[group_idx, "metric_info/gold_medal"] = df.loc[group_idx, "metric_info/gold_medal"].apply(
            lambda arr: np.full_like(arr, 0, dtype=float)
        )

        # Re‑compute the min/max columns we created earlier so they stay in sync
        df.loc[group_idx, "max_test_score"] = sentinel
        df.loc[group_idx, "min_test_score"] = sentinel
        df.loc[group_idx, "max_validation_score"] = sentinel
        df.loc[group_idx, "min_validation_score"] = sentinel

    # Filter all rows where the length is less than 5
    print(f"Filtering rows with length < {min_length}...")
    before_length_filter = len(df)
    df = df[df["length"] >= min_length]
    after_length_filter = len(df)

    print(
        f"Filtered {before_length_filter - after_length_filter} rows based on length. There should always be at least 5 nodes in the experiment."
    )
    print("Number of rows after filtering:", len(df))

    return df


def _is_sequence_like(x: Any) -> bool:
    """True for list / np.ndarray / pd.Series; False otherwise."""
    return isinstance(x, (list, np.ndarray, pd.Series))


def _truncate_to_max(seq: Any, max_len: int) -> Any:
    """Return `seq` truncated to the first `max_len` elements if needed."""
    if not _is_sequence_like(seq):
        return seq

    # Keep the **type** of the original container
    if isinstance(seq, np.ndarray):
        return seq[:max_len]
    if isinstance(seq, pd.Series):
        return seq.iloc[:max_len]
    return seq[:max_len]  # plain list


def filter_dataframe_to_have_limited_nodes(
    df: pd.DataFrame,
    max_num_nodes: int,
) -> pd.DataFrame:
    """
    Truncates every *sequence‑like* column so that each row contains
    at most `max_num_nodes` nodes/entries.

    Parameters
    ----------
    df : pd.DataFrame
        May contain a mix of scalar and per‑node sequence columns.
    max_num_nodes : int
        Hard upper limit on how many nodes are kept per row/column.

    Returns
    -------
    pd.DataFrame
        A **copy** of the input with all sequence columns safely truncated.
    """
    df = df.copy()  # keep caller’s DataFrame intact

    # ------------------------------------------------------------------
    # 1) Find all sequence‑like columns by inspecting the first
    #    non‑null value in each column (cheap and reliable).
    # ------------------------------------------------------------------
    seq_cols = []
    for col in df.columns:
        first_non_na = df[col].dropna().iloc[0] if df[col].notna().any() else None
        if _is_sequence_like(first_non_na):
            seq_cols.append(col)

    # ------------------------------------------------------------------
    # 2) Truncate them row‑wise
    # ------------------------------------------------------------------
    for col in seq_cols:
        df[col] = df[col].apply(_truncate_to_max, max_len=max_num_nodes)

    # ------------------------------------------------------------------
    # 3) Bring the `length` column (if it exists) back in sync
    # ------------------------------------------------------------------
    if "length" in df.columns:
        df["length"] = df["length"].clip(upper=max_num_nodes)

    return df


def add_node_elapsed_from_first(
    df: pd.DataFrame,
    *,
    output_col: str = "seconds_from_first_node",
    unit: str = "seconds",  # "seconds"  (float)  or "timedelta"
) -> pd.DataFrame:
    """
    For every row, compute how much time has elapsed between each node’s
    timestamp and the *first *valid* node’s* timestamp.

    • Uses ``non_padding_mask`` to ignore padding.
    • Produces a sequence the same length as ``timestamp``.
      – First valid position → 0 (or Timedelta('0 days')).
      – Padding positions → np.nan.
    • Appends the result under ``output_col`` and returns a **copy**.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``timestamp`` (list‑like of str/datetime) and
        ``non_padding_mask`` (list/array of bools).
    output_col : str, default "seconds_from_first_node"
        Name of the new column to create.
    unit : {"seconds", "timedelta"}, default "seconds"
        Whether to return floats (seconds) or raw ``pd.Timedelta`` objects.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the extra column added.
    """
    df = df.copy()

    def _row_elapsed(row) -> List[Any]:
        ts_raw: Sequence[Any] = row["timestamp"]
        mask: Sequence[bool] = row["non_padding_mask"]

        # Guard – if *every* element is padding, return all‑NaNs
        if not any(mask):
            return [np.nan] * len(ts_raw)

        # Find the first *valid* timestamp
        first_valid_idx = next(idx for idx, m in enumerate(mask) if m)
        first_ts = pd.to_datetime(ts_raw[first_valid_idx], errors="coerce")

        elapsed_values: list[Any] = []
        for t, m in zip(ts_raw, mask):
            if not m:
                elapsed_values.append(np.nan)
                continue

            td = pd.to_datetime(t, errors="coerce") - first_ts
            elapsed_values.append(td.total_seconds() if unit == "seconds" else td)

        return elapsed_values

    df[output_col] = df.apply(_row_elapsed, axis=1)
    return df


def _slice_to_len(seq: Any, new_len: int) -> Any:
    """Return *seq* sliced to ``new_len`` while preserving its type."""
    if not _is_sequence_like(seq):
        return seq
    if isinstance(seq, np.ndarray):
        return seq[:new_len]
    if isinstance(seq, pd.Series):
        return seq.iloc[:new_len]
    return seq[:new_len]  # plain list


def truncate_dataframe_based_on_elapsed(
    df: pd.DataFrame,
    max_elapsed: str | int | float | pd.Timedelta,
    *,
    elapsed_col: str = "seconds_from_first_node",
) -> pd.DataFrame:
    """
    Remove every node that lies *after* ``max_elapsed`` from the first node.

    • Relies on an existing per‑row sequence column (default
      ``seconds_from_first_node``) produced by ``add_node_elapsed_from_first``.
    • Works on a **copy** of *df* – the original is untouched.
    • Truncates *every* list / ndarray / Series column row‑wise so that
      **all sequence columns stay perfectly aligned**.
    • Updates ``length`` (if present) to the new, shorter size.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *elapsed_col* with one scalar per node (float seconds
        or ``pd.Timedelta`` objects). Missing nodes should be ``np.nan``.
    max_elapsed : str | int | float | pd.Timedelta
        Threshold after which nodes are discarded.

        ── If *elapsed_col* holds **floats (seconds)**:
             – pass a number (seconds) **or** a timedelta‑like string
               e.g. ``"2:30:00"``.
        ── If *elapsed_col* holds **Timedeltas**:
             – pass a ``pd.Timedelta``, or a timedelta‑like string.

    elapsed_col : str, default "seconds_from_first_node"
        Name of the per‑node elapsed‑time column to consult.

    Returns
    -------
    pd.DataFrame
        A copy with all sequence columns truncated row‑wise.
    """
    if elapsed_col not in df.columns:
        raise KeyError(f"Column '{elapsed_col}' not found in the DataFrame.")

    df = df.copy()

    # ------------------------------------------------------------------
    # 1) Normalise the threshold to the same *unit* as elapsed_col
    # ------------------------------------------------------------------
    sample_val = df[elapsed_col].dropna().iloc[0][0]  # first scalar in the first non‑null row
    if isinstance(sample_val, (int, float)):
        # elapsed is in *seconds*  → convert threshold to float seconds
        if isinstance(max_elapsed, (int, float)):
            threshold = float(max_elapsed)
        else:  # str or Timedelta
            threshold = pd.to_timedelta(max_elapsed).total_seconds()
    else:
        # elapsed is Timedelta                                  → ensure threshold is Timedelta
        threshold = max_elapsed if isinstance(max_elapsed, pd.Timedelta) else pd.to_timedelta(max_elapsed)

    # ------------------------------------------------------------------
    # 2) Work out the *new* length for every row
    # ------------------------------------------------------------------
    def _new_len(elapsed_seq: Sequence[Any]) -> int:
        """
        Return the number of nodes to keep so that every retained
        node has elapsed ≤ threshold and is not NaN.
        """
        for idx, v in enumerate(elapsed_seq):
            if pd.isna(v):
                return idx  # stop at first padding
            if v > threshold:
                return idx  # stop at first node past limit
        return len(elapsed_seq)  # all nodes kept

    df["__new_len"] = df[elapsed_col].apply(_new_len)

    # ------------------------------------------------------------------
    # 3) Identify *all* sequence columns in one inexpensive sweep
    # ------------------------------------------------------------------
    seq_cols: list[str] = []
    for col in df.columns:
        first_val = df[col].dropna().iloc[0] if df[col].notna().any() else None
        if _is_sequence_like(first_val):
            seq_cols.append(col)

    # ------------------------------------------------------------------
    # 4) Truncate them row‑wise
    # ------------------------------------------------------------------
    for col in seq_cols:
        df[col] = df.apply(lambda r: _slice_to_len(r[col], r["__new_len"]), axis=1)

    # ------------------------------------------------------------------
    # 5) Keep ancillary metadata in sync
    # ------------------------------------------------------------------
    if "length" in df.columns:
        df["length"] = df["__new_len"]

    # 6) Clean up helper column
    df.drop(columns="__new_len", inplace=True)

    return df


# Section: Formatting of dataframe into ideal format
################################################


def get_selection_mask(row):
    """Get the mask of nodes that could be selected."""
    # We check if the validation metric is nan.
    metric_is_nan = np.isnan(row["metric"])
    # We inverse this to get the locations which have values.
    metric_is_not_nan = np.logical_not(metric_is_nan)
    # We check if submission.csv is valid for grader.
    is_valid_submission = np.nan_to_num(row["metric_info/valid_submission"], nan=0)
    # We then check if node not legal for selection for any other reason (see greedy code parse eval result fn)
    is_buggy_node = np.nan_to_num(row["is_buggy"].astype(float), nan=1)
    # We can get the good nodes from this
    is_good_node = np.logical_not(is_buggy_node.astype(bool)) & np.asarray(row["non_padding_mask"], dtype=bool)
    # Lastly, we and everything together
    selection_mask = np.logical_and(
        np.logical_and(metric_is_not_nan.astype(bool), is_valid_submission.astype(bool)), is_good_node.astype(bool)
    )
    # This gives us a mask for all legally selectable nodes
    return selection_mask


def get_valid_submission(row, node_idx):
    """Simply check if a node has a valid submission csv."""
    return np.nan_to_num(row["metric_info/valid_submission"], nan=0)[node_idx]


def get_expected_return_node(row, lower_is_better: bool, use_validation_score: bool = True):
    """Process the validation metric data to see what node the search would have returned."""
    # Get Validation Scores
    if use_validation_score:
        v_scores = row["metric"]
    else:
        v_scores = row["metric_info/score"]
    max_v_score = np.max(np.nan_to_num(v_scores, nan=-1e8))
    min_v_score = np.min(np.nan_to_num(v_scores, nan=1e8))

    # Get Selection Masks
    selection_mask = get_selection_mask(row)

    # Get the masked validation scores
    fill_value = (1 + max_v_score) if lower_is_better else (min_v_score - 1)
    v_scores = np.nan_to_num(v_scores, nan=fill_value)
    masked_v_scores = np.where(selection_mask, v_scores, fill_value)

    # Select the node
    if lower_is_better:
        best_node = np.argmin(masked_v_scores)
    else:
        best_node = np.argmax(masked_v_scores)

    if np.all(selection_mask == False):
        # If all nodes are invalid, we just return the last node
        best_node = -1

    return best_node


def get_metric(row, metric_key, valid_submission, expected_return_node, lower_is_better):
    metric_scores = row[metric_key]
    selected_metric_score = metric_scores[expected_return_node]
    if np.isnan(selected_metric_score):
        print(f"{row['method']} - {row['competition_id']} - {row['seed']}: selected metric ({metric_key}) is nan.")
        print("Expected return node:", expected_return_node)
        print("Valid submission:", valid_submission)
        print("Valid Node Exists?:", np.any(get_selection_mask(row)))
        print("Node is buggy:", row["is_buggy"][expected_return_node])

        # This is happening due to mlebench's grader being able to return nans even if everything supposedly works.
        if "medal" in metric_key:
            selected_metric_score = 0
        elif "above_median" in metric_key:
            selected_metric_score = 0
        else:
            if lower_is_better:
                selected_metric_score = 1e12
            else:
                selected_metric_score = -1e12

        print(
            f"{row['method']} - {row['competition_id']} - {row['seed']}: due to nan, selected metric ({metric_key}) will be {selected_metric_score}."
        )

    return selected_metric_score


def get_rank_and_percentile(
    score: float, leaderboard: pd.DataFrame, lower_is_better: bool
) -> Dict[str, Union[float, None]]:
    """
    Calculates the percentile rank of `score` as if it were an additional submission in the leaderboard.

    The function computes the average rank of `score` among all scores (including itself) and then maps it
    to a percentile between 0 and 1 using:

        percentile = (n - avg_rank) / (n - 1)

    where n is the total number of scores after including the new one.

    - A percentile of 1 indicates the best score.
    - A percentile of 0 indicates the worst score.

    If score is None, returns {"percentile": None, "rank": None}.
    """

    # Ensure the leaderboard has a 'score' column.
    if "score" not in leaderboard.columns:
        raise ValueError("Leaderboard must have a `score` column.")

    # Return early if score is None.
    if score is None or np.isnan(score):
        return {"percentile": 0, "rank": len(list(leaderboard["score"])) + 1}

    # Ensure the score is numeric.
    if not isinstance(score, (int, float)):
        raise TypeError(f"Expected score to be a float or int, got {type(score)}.")

    # Combine existing scores with the new score.
    scores_list = list(leaderboard["score"]) + [score]
    n = len(scores_list)

    # If there's only one score, it is by definition the best.
    if n == 1:
        return {"percentile": 1.0, "rank": 1}

    # Sort scores: best first. For lower_is_better, lower numbers are better.
    if lower_is_better:
        sorted_scores = sorted(scores_list)  # ascending: best is first
    else:
        sorted_scores = sorted(scores_list, reverse=True)  # descending: best is first

    # Use a tolerance-based comparison to handle floating point imprecision.
    # Find all positions (1-indexed) where the new score is "close" to an existing score.
    tol_rel = 1e-9
    tol_abs = 1e-12
    ranks = [i + 1 for i, s in enumerate(sorted_scores) if math.isclose(s, score, rel_tol=tol_rel, abs_tol=tol_abs)]

    # In case no value is considered close (shouldn't happen because score is in scores_list), fall back to exact equality.
    if not ranks:
        ranks = [i + 1 for i, s in enumerate(sorted_scores) if s == score]
    try:
        avg_rank = sum(ranks) / len(ranks)

        # Compute the percentile.
        percentile = (n - avg_rank) / (n - 1)

        return {"percentile": percentile, "rank": avg_rank}
    except Exception as e:
        print(f"Error occurred in calculating percentile and rank for Score = {score}: {e}")


def is_lower_better(competition):
    leaderboard_df = pd.read_csv(competition.leaderboard)

    try:
        _ = leaderboard_df["score"]
    except:
        raise Exception("You must run GIT LFS for mlebench.")

    return competition.grader.is_lower_better(leaderboard_df)


def get_statistics(x, label):
    if len(x) > 1:
        data = {
            f"avg_{label}": np.mean(x),
            f"min_{label}": np.min(x),
            f"max_{label}": np.max(x),
            f"std_{label}": np.std(x),
            f"q1_{label}": np.quantile(x, 0.25),
            f"q3_{label}": np.quantile(x, 0.75),
            f"raw_data_{label}": x,
        }
        return data

    return {}


def format_experiment_data(
    experiments_df: pd.DataFrame,
    max_num_seeds: int = 20,
    select_using_test: bool = False,
    node_selector_function: Optional[Callable] = None,
) -> pd.DataFrame:
    """Take in the dataframe containing experiment data per row and convert it into competition reports."""
    # Get the unique tasks and methods
    unique_tasks = experiments_df["competition_id"].unique()
    unique_methods = experiments_df["method"].unique()

    print(f"Processing {unique_tasks}")
    print(f"Processing {unique_methods}")

    # You might have a registry or some object to get competition info:
    new_registry = registry.set_data_dir(Path(get_mlebench_data_dir()))

    all_reports = []
    statistics = defaultdict(lambda: defaultdict(float))
    for task in unique_tasks:
        # Filter rows for this particular competition_id
        comp_mask = experiments_df["competition_id"] == task
        task_df = experiments_df[comp_mask].copy()

        # Get the competition details i.e. lower is better
        competition = new_registry.get_competition(task)
        lower_is_better = is_lower_better(competition)
        leaderboard_df = pd.read_csv(competition.leaderboard)
        for method in unique_methods:
            # Filter rows for this particular method
            method_mask = task_df["method"] == method
            method_df = task_df[method_mask].copy()
            total_num_runs = len(method_df)

            # Now create the report dict for each row
            reported_row_indices = []
            num_reported_rows = 0
            idx = -1
            for _, row in method_df.iterrows():
                idx += 1

                try:
                    if node_selector_function is None:
                        expected_return_node = get_expected_return_node(
                            row, lower_is_better, use_validation_score=not select_using_test
                        )
                    else:
                        expected_return_node = node_selector_function(row, lower_is_better)

                    valid_submission = get_valid_submission(row, expected_return_node)
                    validation_score = row["metric"][expected_return_node]

                    score = get_metric(
                        row, "metric_info/score", valid_submission, expected_return_node, lower_is_better
                    )
                    any_medal = get_metric(
                        row, "metric_info/any_medal", valid_submission, expected_return_node, lower_is_better
                    )
                    gold_medal = get_metric(
                        row, "metric_info/gold_medal", valid_submission, expected_return_node, lower_is_better
                    )
                    silver_medal = get_metric(
                        row, "metric_info/silver_medal", valid_submission, expected_return_node, lower_is_better
                    )
                    bronze_medal = get_metric(
                        row, "metric_info/bronze_medal", valid_submission, expected_return_node, lower_is_better
                    )
                    above_median = get_metric(
                        row, "metric_info/above_median", valid_submission, expected_return_node, lower_is_better
                    )

                    relative_placement = get_rank_and_percentile(score, leaderboard_df, lower_is_better)

                    if np.isnan(relative_placement["percentile"]) or not isinstance(
                        relative_placement["percentile"], float
                    ):
                        print(
                            f"{row['method']} - {row['competition_id']} - {row['seed']}: Percentile is Not a float = {relative_placement['percentile']}."
                        )

                    if score == 1e12 or score == -1e12:
                        assert relative_placement["percentile"] == 0, (
                            f"{row['method']} - {row['competition_id']} - {row['seed']}: Score is sentinel value = {score}."
                        )

                    report_row = {
                        "experiment_id": row["method"],
                        "competition_id": row["competition_id"],
                        "score": float(score),
                        "validation_score": float(validation_score.item()),
                        "any_medal": bool(any_medal),
                        "gold_medal": bool(gold_medal),
                        "silver_medal": bool(silver_medal),
                        "bronze_medal": bool(bronze_medal),
                        "above_median": bool(above_median),
                        "valid_submission": bool(valid_submission),
                        "length": row["length"],
                        "validation_scores": list(row["metric"]),
                        "test_scores": list(row["metric_info/score"]),
                        "rank": relative_placement["rank"],
                        "percentile": relative_placement["percentile"],
                        "lower_is_better": lower_is_better,
                        "seed": row["seed"],
                        "expected_return_node": expected_return_node,
                        "num_validation_nodes": row["num_validation_nodes"],
                        "num_test_nodes": row["num_test_nodes"],
                        "num_good_nodes": row["num_good_nodes"],
                        "num_buggy_nodes": row["num_buggy_nodes"],
                        "future_unused_nodes": row["length"] - expected_return_node,
                        "seconds_from_first_node": row["seconds_from_first_node"],
                        "any_medal_series": list(row["metric_info/any_medal"]),
                        "gold_medal_series": list(row["metric_info/gold_medal"]),
                        "silver_medal_series": list(row["metric_info/silver_medal"]),
                        "bronze_medal_series": list(row["metric_info/bronze_medal"]),
                        "above_median_series": list(row["metric_info/above_median"]),
                        "percentile_series": [
                            get_rank_and_percentile(s, leaderboard_df, lower_is_better)["percentile"]
                            for s in list(row["metric_info/score"])
                        ],
                        "val_percentile_series": [
                            get_rank_and_percentile(s, leaderboard_df, lower_is_better)["percentile"]
                            for s in list(row["metric"])
                        ],
                    }

                    all_reports.append(report_row)
                    reported_row_indices.append(idx)
                    num_reported_rows += 1

                    if num_reported_rows >= max_num_seeds:
                        break
                except Exception as e:
                    print(f"Error in processing {row['method']} - {row['competition_id']} - {row['seed']}: {e}")

            # Get the statistics
            method_df = method_df.iloc[reported_row_indices]
            statistics[task][method] = {
                **get_statistics(method_df["length"], "length"),
                **get_statistics(method_df["num_validation_nodes"], "num_validation_nodes"),
                **get_statistics(method_df["num_test_nodes"], "num_test_nodes"),
                **get_statistics(method_df["num_good_nodes"], "num_good_nodes"),
                **get_statistics(method_df["num_buggy_nodes"], "num_buggy_nodes"),
                "total_num_runs": total_num_runs,
            }

    # Finally convert all_reports into a single DataFrame:
    reports_df = pd.DataFrame(all_reports)
    return reports_df, statistics


def parse_into_aggregate_dict(report_df, metric, algorithms=None):
    if algorithms is None:
        algorithms = list(report_df["experiment_id"].unique())

    # Get the unique methods
    methods = algorithms
    print(f"Processing {methods}")
    score_dict = {}

    for method in methods:
        # Filter the dataframe for the current method
        m_df = report_df[report_df["experiment_id"] == method].copy()
        # Create a 'seed' column based on the order of rows for each task
        m_df["seed"] = m_df.groupby("competition_id").cumcount()
        # Pivot the dataframe: rows = seeds, columns = tasks, values = metric
        pivot_df = m_df.pivot(index="seed", columns="competition_id", values=metric)
        # filter out the rows where the metric is NaN
        # This is to ensure we only keep rows where the metric is not NaN
        # Remove rows with NaN values
        pivot_df = pivot_df.dropna()
        # Convert the pivoted dataframe to a numpy matrix
        score_dict[method] = pivot_df.to_numpy().astype(float)
        score_dict[method] = score_dict[method][~np.isnan(score_dict[method]).any(axis=1)]

    return score_dict
