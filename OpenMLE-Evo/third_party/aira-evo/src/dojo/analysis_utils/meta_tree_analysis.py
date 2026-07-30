# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import concurrent.futures
import itertools
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tiktoken
import yaml
from dojo.config_dataclasses.run import RunConfig
from dojo.utils.environment import get_mlebench_data_dir
from dojo.core.solvers.utils.journal import Journal
from dojo.core.solvers.utils.response import parse_thinking_tags
from dojo.analysis_utils.journal_to_tree import save_journal_log_as_json


def create_client(api: str = "gdm", model_id: str = "gemini-2.0-flash", provider: str = "gdm"):
    """Creates and returns a configured GDM client."""
    from omegaconf import OmegaConf
    from dojo.core.solvers.llm_helpers.backends.gdm import GDMClient

    client_cfg = OmegaConf.create({"api": api, "model_id": model_id, "provider": provider})
    return GDMClient(client_cfg)


# Global client instance for reuse across functions
client = create_client()


def generate_journal_report(journal: Journal, task_description: str, include_code: bool = False):
    """Generates a structured technical markdown report from a journal."""

    report_input = journal.generate_summary(include_code=include_code)

    system_prompt = f"""
    # Role
    You are a research assistant that always uses concise language.
    # Goal
    The goal is to write a technical report summarising the empirical findings and technical decisions.
    # Input
    You are given a raw research journal with list of design attempts and their outcomes, and a task description.
    # Output
    Your output should be a single markdown document.
    Your report should have the following sections: Introduction, Preprocessing, Modellind Methods, Results Discussion, Future Work.
    You can include subsections if needed.
    Here is the research journal of the agent: <journal>{report_input}<\\journal>, 
    and the task description is: <task>{task_description}<\\task>.
    """

    messages = [{"role": "system", "content": system_prompt}]

    out, _ = client.query(messages=messages)

    _, text_without_thinking = parse_thinking_tags(out)

    return text_without_thinking


def calculate_tree_statistics(journal: Journal) -> Dict[str, Union[int, float]]:
    """Calculates statistical metrics from a Journal tree structure."""
    nodes = journal.nodes

    stats = {}

    stats = {
        "num_total_nodes": len(nodes),
        "num_root_nodes": sum([1 if len(node.parents) == 0 else 0 for node in nodes]),
        "num_intermediate_nodes": sum(
            [1 if (len(node.parents) > 0 and len(node.children) > 0) else 0 for node in nodes]
        ),
        "num_leaf_nodes": sum([1 if len(node.children) == 0 else 0 for node in nodes]),
    }

    num_children = [len(node.children) for node in nodes]
    execution_times = [node.exec_time for node in nodes]

    if len(num_children) == 0:
        num_children = [0]

    stats.update(
        {
            "avg_num_children": sum(num_children) / stats["num_total_nodes"],
            "max_num_children": max(num_children),
            "avg_exec_time": np.mean(execution_times) if execution_times else None,
            "min_exec_time": np.min(execution_times) if execution_times else None,
            "max_exec_time": np.max(execution_times) if execution_times else None,
            "median_exec_time": np.median(execution_times) if execution_times else None,
            "total_exec_time": np.sum(execution_times) if execution_times else None,
        }
    )

    # Depth calculation helper function
    def calculate_depth(node_id, node_dict, depth=0):
        node = node_dict[node_id]
        if not node.children:
            return depth
        return max(calculate_depth(child.id, node_dict, depth + 1) for child in node.children)

    node_dict = {node.id: node for node in nodes}
    root_nodes = [node for node in nodes if len(node.parents) == 0]
    depths = [calculate_depth(root_node.id, node_dict=node_dict) for root_node in root_nodes]
    stats["max_depth"] = max(depths) if depths else 0

    # Nodes with metrics
    nodes_with_val_metrics = [1 if node.metric.value is not None else 0 for node in nodes]
    if len(nodes_with_val_metrics) == 0:
        nodes_with_val_metrics = [0]
    stats["num_nodes_with_val_metrics"] = sum(nodes_with_val_metrics)

    # Average metrics (if metrics are available)
    metrics = [node.metric.value for node in nodes if node.metric.value is not None]
    stats["average_val_metric"] = sum(metrics) / len(metrics) if metrics else None

    # Number of buggy nodes
    stats["num_buggy_nodes"] = sum([1 if node.is_buggy else 0 for node in nodes])
    stats["num_good_nodes"] = stats["num_total_nodes"] - stats["num_buggy_nodes"]

    # Buggy nodes rate
    stats["buggy_nodes_proportion"] = (
        stats["num_buggy_nodes"] / stats["num_total_nodes"] if stats.get("num_total_nodes") else 0
    )
    stats["good_nodes_proportion"] = 1 - stats["buggy_nodes_proportion"]

    # Nodes without analysis
    stats["nodes_without_analysis"] = sum([1 if (not node.analysis) else 0 for node in nodes])

    operators_used_all = []
    for node in nodes:
        operators_used_all.extend(node.operators_used)
    operators_set = set(operators_used_all)
    for op in operators_set:
        op_counter = lambda n, op: sum([1 if op_used == op else 0 for op_used in n.operators_used])
        stats[f"{op}_frequency"] = sum([op_counter(node, op) for node in nodes])
        stats[f"{op}_proportion"] = stats[f"{op}_frequency"] / stats["num_total_nodes"]

    all_operators_metrics = [node.operators_metrics for node in nodes]
    all_operators_metrics = list(itertools.chain(*all_operators_metrics))

    if all_operators_metrics:
        all_token_counts = [op_metrics["usage"]["total_tokens"] for op_metrics in all_operators_metrics]

        all_prompt_token_counts = [op_metrics["usage"]["prompt_tokens"] for op_metrics in all_operators_metrics]
        all_completion_token_counts = [
            op_metrics["usage"]["completion_tokens"] for op_metrics in all_operators_metrics
        ]
        all_latency = [op_metrics["usage"]["latency"] for op_metrics in all_operators_metrics]

        stats["avg_num_all_tokens"] = sum(all_token_counts) / len(all_token_counts)
        stats["total_num_all_tokens"] = sum(all_token_counts)

        stats["avg_num_prompt_tokens"] = sum(all_prompt_token_counts) / len(all_prompt_token_counts)
        stats["total_num_prompt_tokens"] = sum(all_prompt_token_counts)

        stats["avg_num_completion_tokens"] = sum(all_completion_token_counts) / len(all_completion_token_counts)
        stats["total_num_completion_tokens"] = sum(all_completion_token_counts)

        stats["avg_latency"] = sum(all_latency) / len(all_latency)
        stats["total_latency"] = sum(all_latency)

    # MLEbench specific metrics
    test_scores = [node.metric.info["score"] for node in nodes if node.metric.info and "score" in node.metric.info]
    if test_scores:
        stats["avg_test_score"] = sum(test_scores) / len(test_scores)
        stats["min_test_score"] = min(test_scores)
        stats["max_test_score"] = max(test_scores)

    return stats


def prettify_key(key: str) -> str:
    """Converts a snake_case or camelCase key to a human-readable format."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", key.replace("_", " ")).title()


def dict_to_markdown(data: dict) -> str:
    """Formats a dictionary into markdown."""
    return "\n".join(f"- **{prettify_key(k)}:** {v}" for k, v in data.items())


def write_tree_reports(experiment_path: Path):
    """Writes detailed markdown reports and statistics based on an experiment's journal."""
    config = RunConfig.load_from_json(experiment_path / "dojo_config.json")
    competition_id, exp_name = config.task.name, config.meta_id

    task_desc_path = Path(get_mlebench_data_dir()) / competition_id / "prepared/public/description.md"
    task_description = task_desc_path.read_text()

    journal_json_path = experiment_path / "json/JOURNAL.jsonl"
    journal = Journal.from_export_data(save_journal_log_as_json(journal_json_path, experiment_path, "journal.json"))

    report = generate_journal_report(journal, task_description)
    (experiment_path / f"{competition_id}_report.md").write_text(report)

    stats = calculate_tree_statistics(journal)
    stats_report = dict_to_markdown(stats)
    (experiment_path / f"{competition_id}_stats_report.md").write_text(stats_report + f"\n\nExperiment: {exp_name}")

    stats["competition_id"] = competition_id
    stats["exp_name"] = exp_name

    return stats


def plot_aggregate_stats(stats_list: List[Dict[str, Union[int, float]]], comp_name: str, meta_exp_dir: Path):
    """
    Plots various distributions (boxplot, histogram, and bar chart) of all numerical
    metrics across multiple runs using seaborn. Also calculates and writes raw data statistics
    (mean, median, Q1, Q3) for each metric to a text file.

    Args:
        stats_list (List[Dict[str, Union[int, float]]]): A list of dictionaries where keys are metric names
                                           and values are numerical values.
        comp_name (str): Base name for the output files (PNG and TXT).
    """

    plot_loc = meta_exp_dir / "tree_stats"
    plot_loc.mkdir(parents=True, exist_ok=True)

    # Identify all metric names from stats_list
    metric_names = set()
    for stats in stats_list:
        metric_names.update(stats.keys())

    # Collect numeric values for each metric
    metric_data = {metric: [] for metric in metric_names}
    for stats in stats_list:
        for metric, value in stats.items():
            if isinstance(value, (int, float)):
                metric_data[metric].append(value)

    # Remove metrics with no numeric data
    metric_data = {k: v for k, v in metric_data.items() if v}
    if not metric_data:
        print("No numeric metrics found to plot.")
        return

    # Write summary statistics to a text file
    with open(plot_loc / f"{comp_name}_stats.txt", "w") as f:
        for metric, values in metric_data.items():
            values_array = np.array(values)
            min_val = np.min(values_array)
            mean_val = np.mean(values_array)
            median_val = np.median(values_array)
            q1 = np.percentile(values_array, 25)
            q3 = np.percentile(values_array, 75)
            max_val = np.max(values_array)
            f.write(f"Metric: {metric}\n")
            f.write(f"  Min: {min_val:.3f}\n")
            f.write(f"  Q1 (25th percentile): {q1:.3f}\n")
            f.write(f"  Mean: {mean_val:.3f}\n")
            f.write(f"  Median: {median_val:.3f}\n")
            f.write(f"  Q3 (75th percentile): {q3:.3f}\n")
            f.write(f"  Max: {max_val:.3f}\n")
            f.write("-" * 40 + "\n")

    # Set up seaborn style
    sns.set(style="whitegrid")

    # Create subplots: one row per metric, three columns for the different plots
    num_metrics = len(metric_data)
    fig, axes = plt.subplots(num_metrics, 3, figsize=(18, 4 * num_metrics))

    # If only one metric, wrap axes into a 2D array for consistency
    if num_metrics == 1:
        axes = np.array([axes])

    # Loop through each metric and create the plots
    for ax_triplet, (metric, values) in zip(axes, metric_data.items()):
        # Prepare a DataFrame for seaborn
        df = pd.DataFrame({metric: values})
        df["Run"] = df.index  # Run index for the bar plot

        # Boxplot
        sns.boxplot(data=df, x=metric, ax=ax_triplet[0])
        ax_triplet[0].set_title(f"Boxplot of {metric}")

        # Histogram with KDE for a smoother view
        sns.histplot(data=df, x=metric, bins=20, kde=True, ax=ax_triplet[1])
        ax_triplet[1].set_title(f"Histogram of {metric}")
        ax_triplet[1].set_xlabel(metric)
        ax_triplet[1].set_ylabel("Frequency")

        # Bar chart showing each run's value
        sns.barplot(data=df, x="Run", y=metric, ax=ax_triplet[2], errorbar=None)
        ax_triplet[2].set_title(f"Bar Chart of {metric}")
        ax_triplet[2].set_xlabel("Run Index")
        ax_triplet[2].set_ylabel(metric)

    plt.tight_layout()
    plt.savefig(plot_loc / f"{comp_name}.png")
    plt.close(fig)


def generate_tree_reports_and_stats(meta_experiment_path: Path):
    """Concurrently processes multiple experiments and aggregates statistics."""
    meta_experiment_path = Path(meta_experiment_path)

    if not meta_experiment_path.exists():
        print(f"Experiments path does not exist: {meta_experiment_path}")
        return

    all_experiment_stats = defaultdict(list)
    experiment_folders = [folder for folder in meta_experiment_path.iterdir() if folder.is_dir()]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_experiment = {
            executor.submit(write_tree_reports, experiment_folder): experiment_folder
            for experiment_folder in experiment_folders
        }
        for future in concurrent.futures.as_completed(future_to_experiment):
            experiment_folder = future_to_experiment[future]
            try:
                stats = future.result()
                comp_id = stats["competition_id"]
                all_experiment_stats[comp_id].append(stats)
                print(f"Processed experiment: {experiment_folder.name}")
            except Exception as exc:
                print(f"Experiment {experiment_folder.name} generated an exception: {exc}")

    # After processing all experiments, plot aggregate stats per competition id
    for comp, stats in all_experiment_stats.items():
        plot_aggregate_stats(stats, comp, meta_experiment_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python meta_tree_analysis.py <meta_experiment_directory>")
        sys.exit(1)

    generate_tree_reports_and_stats(Path(sys.argv[1]))
