# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dojo.core.solvers.llm_helpers.backends.gdm import GDMClient
from dojo.core.solvers.utils.response import parse_thinking_tags
from dojo.utils.code_parsing import parse_json_output
from dojo.utils.experiment_logs import is_experiment
from dojo.config_dataclasses.run import RunConfig
import os
import glob
import json
import re
from pathlib import Path
from omegaconf import OmegaConf
from collections import defaultdict
from tqdm import tqdm


def create_client():
    client_cfg = OmegaConf.create({"api": "gdm", "model_id": "gemini-2.0-flash", "provider": "gdm"})

    client = GDMClient(client_cfg)

    return client


def get_last_approx_tokens(filepath, approx_token_count=70000, avg_chars_per_token=4):
    """
    Returns approximately the last `approx_token_count` tokens from a file.
    Approximation is based on average characters per token.
    Extra whitespace and newlines are compressed to avoid wasting token space.
    """
    approx_byte_count = approx_token_count * avg_chars_per_token

    with open(filepath, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()

        # Calculate position from where to start reading
        seek_pos = max(file_size - approx_byte_count, 0)
        f.seek(seek_pos)

        # Read and decode the data
        data = f.read().decode("utf-8", errors="replace")

        # To avoid partial tokens/words at the beginning, skip to the first newline
        first_newline = data.find("\n")
        if first_newline != -1:
            data = data[first_newline + 1 :]

        # Compress multiple whitespace characters (including newlines) into a single space
        data = " ".join(data.split())
        data = re.sub(r"\n+", "\n", data)

        return data


def slurm_id_to_log(logs_folder):
    """
    Extracts 'Output dir: *' from *.out files, cleaning ANSI escape characters.
    Returns a dictionary mapping job names to cleaned output directories.
    """
    job_id2experiment_id = {
        RunConfig.load_from_json(subdir / "dojo_config.json").metadata.slurm_id: subdir
        for subdir in logs_folder.iterdir()
        if is_experiment(subdir)
    }
    return job_id2experiment_id


def likely_crashed(log_text):
    """
    Simple heuristic to guess if a job crashed
    by scanning the last 50 lines for certain keywords.
    Feel free to make this more/less robust.
    """
    crash_indicators = [
        "error",
        "exception",
        "traceback",
        "killed",
        "fatal",
        "submitit error",
        "exited with exit code 1",
        "Terminating",
        "non-zero exit",
    ]
    return any(word.lower() in log_text.lower() for word in crash_indicators)


def gather_submitit_data(logs_folder, job_ids):
    """
    - Walks the logs_folder.
    - Finds subfolders containing *.err files.
    - Reads and stores last tokens.
    - Guesses crash status.
    - Returns a JSON-like list of crashed jobs + statistics.
    """
    results = []
    num_runs = 0
    slurm_log_path = Path(logs_folder).parent.parent / "slurm_logs"

    for job_id in job_ids:
        err_files = list(slurm_log_path.glob(f"**/*{job_id}*.err"))

        if len(err_files) == 1:
            err_file = err_files[0]
            num_runs += 1

            last_tokens = get_last_approx_tokens(err_file, 10_000)
            crashed = likely_crashed(last_tokens)

            if crashed:
                results.append(
                    {
                        "job_name": job_id,
                        "err_file_path": err_file,
                        "status": "likely_crashed",
                        "last_err_tokens": last_tokens,
                    }
                )
    # Append meta-info as a separate dictionary
    results.append({"total_num_runs": num_runs, "num_crashed": len(results)})
    return results


job_summary_schema = """{
    "type": "object",
    "properties": {
        "job_name": {
            "type": "string",
            "description": "The name or identifier of the job."
        },
        "experiment_name": {
            "type": "string",
            "description": "The name of the experiment."
        },
        "details": {
            "type": "string",
            "description": "Brief details or log excerpts explaining the error or crash status."
        },
        "error_category": {
            "type": "string",
            "description": "Short label categorizing the type of error, e.g., 'CUDA OOM', 'AssertionError', 'Python Exception', or 'None' if not truly crashed."
        },
        "is_crashed": {
            "type": "boolean",
            "description": "True if the job actually crashed the outer loop, False if it handled internal exceptions without crashing."
        },
        "error_text": {
            "type": "string",
            "description": "The text responsible for indicating an error occurred."
        }
    },
    "required": ["job_name", "experiment_name", "details", "error_category", "is_crashed", "error_text"]
}"""


def summarize_single_crash(job, job_to_name):
    # If you have a raw text or JSON to embed:
    last_err_tokens = "\n".join(job.get("last_err_tokens", []))

    system_prompt = f"""
        # Role
        You are a technical assistant who provides concise analyses.

        # Goal
        You are given the last tokens of a job's .err logs. 
        The job was heuristically flagged as 'likely_crashed', 
        but this crash might be internal (allowed within the code) 
        rather than an actual crash of the outer loop. Often if the crash information
        is preceded by a Term Out: it most likely is a crash in the inner loop. Make note 
        if it is most likely an inner loop crash such that we can attempt to distinguish 
        it later.

        # Input
        job_name = {job["job_name"]}
        experiment_name = {job_to_name[job["job_name"]]}

        Here are the last tokens of the err log:
        <log_snippet>
        {last_err_tokens}
        </log_snippet>

        # Task
        1. Determine whether this run TRULY crashed the outer loop or if the crash was internal/allowed.
        2. If truly crashed, assign a short error category (e.g., 'CUDA OOM', 'AssertionError', 'Python Exception', etc.).
        3. Provide a brief detail or excerpt from the logs explaining why you concluded that.
        4. Tip: If the last line of the logs show a message with a timestamp, it most likely is still running and has not crashed.

        # Output
        Return a JSON object ONLY with fields:
        - job_name
        - experiment_name
        - is_crashed (true/false)
        - error_category (string, or 'none' if not truly crashed)
        - details (short text)
        - error text (text indicating the error)
            """.strip()

    messages = [{"role": "user", "content": system_prompt}]
    client = create_client()
    response_text, _ = client.query(
        messages=messages,
        json_schema=job_summary_schema,
        function_name="jobCrashSummary",
        function_description="Return a JSON object summarizing the job's crash status, including the job name, a short set of details from the error log, a categorized error type, and a boolean indicating if the outer loop truly crashed.",
    )

    summary_dict = parse_json_output(response_text)

    if len(summary_dict) == 0:
        summary_dict = {
            "job_name": job["job_name"],
            "is_crashed": True,
            "experiment_name": job_to_name[job["job_name"]],
            "error_category": "unknown",
            "details": "LLM returned invalid JSON. Unable to parse.",
            "error_text": "Could not find",
        }

    assert "job_name" in summary_dict
    assert "is_crashed" in summary_dict
    assert "error_category" in summary_dict
    assert "details" in summary_dict

    return summary_dict


def final_report_from_summaries(summaries):
    """
    Takes a list of per-job summaries (in JSON form) and does
    one final LLM call to produce a consolidated markdown report.
    """

    # Convert entire summary list to JSON
    summaries_json = json.dumps(summaries, indent=2)

    system_prompt = f"""
        # Role
        You are a technical assistant who provides concise analyses.

        # Goal
        You are given a JSON array, where each element is a summary of a job's crash status:
        (job_name, experiment_name, is_crashed, error_category, details).

        # Input
        JSON Summaries:
        <summaries>
        {summaries_json}
        </summaries>

        # Task
        1. Provide a final markdown report that groups jobs by is_crashed vs not_crashed.
        2. For crashed jobs, group them by error_category.
        3. Provide a short discussion of each category. 
        4. Give overall stats that give insights to help identify areas to fix: total jobs analyzed, total truly crashed, etc.
        5. Give the experiment name alongside the job name.
        
        # Output
        Return a single markdown document with your consolidated analysis. Do not surround it with a code block.
    """.strip()

    messages = [{"role": "system", "content": system_prompt}]
    client = create_client()
    out, _ = client.query(messages=messages)
    _, final_markdown = parse_thinking_tags(out)
    return final_markdown


def generate_error_reports(log_dir, save_dir):
    """
    1. Gather data from logs.
    2. For each 'likely_crashed' job, do an LLM call to refine the analysis.
    3. Accumulate all job-level summaries + meta-stats.
    4. Do a final LLM call on the entire set of summaries.
    5. Write the final output to 'logs_report.md'.
    """
    job_to_name = slurm_id_to_log(log_dir)

    data = gather_submitit_data(log_dir, list(job_to_name.keys()))

    # The last dict in 'data' has meta info about total_num_runs and num_crashed
    meta_info = data[-1] if isinstance(data[-1], dict) and "total_num_runs" in data[-1] else {}

    # Filter out the meta_info from the actual job logs
    job_candidates = data[:-1] if meta_info else data

    # Summarize each 'likely_crashed' job
    # (We only run LLM calls on these, as you requested)
    crash_summaries = []
    for job in tqdm(job_candidates, desc="Summarising Likely Crashed Jobs"):
        summary = summarize_single_crash(job, job_to_name)
        crash_summaries.append(summary)

    # Optionally, you might embed the meta_info right into our final summary:
    # E.g. as a dictionary so the final LLM knows total runs
    meta_info_dict = {
        "total_num_runs": meta_info.get("total_num_runs", 0),
        "num_likely_crashed": meta_info.get("num_crashed", 0),
    }
    # Combine them in a single list or dict:
    combined_summaries = {"meta_info": meta_info_dict, "per_job_summaries": crash_summaries}

    with open(save_dir / "error_analysis_per_job_jsons.md", "w") as f:
        f.write(str(combined_summaries))

    # Now do one final LLM call to create a single markdown report
    final_md = final_report_from_summaries(combined_summaries)

    with open(save_dir / "error_analysis_report.md", "w") as f:
        f.write(final_md)

    return final_md  # or just return it if you prefer


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python meta_error_summary.py <meta_exp_directory> <optional_save_dir>")
        sys.exit(1)

    log_dir = Path(sys.argv[1])

    if len(sys.argv) > 2:
        save_dir = Path(sys.argv[2])
    else:
        save_dir = log_dir

    generate_error_reports(log_dir, save_dir)
