# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Weco AI Ltd
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/WecoAI/aideml/blob/main/LICENSE

"""Export journal to HTML visualization of tree + code."""

import json
import textwrap
from pathlib import Path

import numpy as np
from igraph import Graph

from dojo.core.solvers.utils.journal import Journal


def get_edges(journal: Journal):
    for node in journal:
        if node.step is None:
            continue
        for c in node.children:
            if c is None or c.step is None:
                continue
            yield (node.step, c.step)


def generate_layout(n_nodes, edges):
    layout = Graph(
        n_nodes,
        edges=edges,
        directed=True,
    ).layout("rt")
    y_max = max(layout[k][1] for k in range(n_nodes))
    layout_coords = []
    for n in range(n_nodes):
        layout_coords.append((layout[n][0], 2 * y_max - layout[n][1]))
    return np.array(layout_coords)


def normalize_layout(layout: np.ndarray):
    """Normalize layout to [0, 1]"""
    layout = (layout - layout.min(axis=0)) / (layout.max(axis=0) - layout.min(axis=0) + 1e-8)
    layout[:, 1] = 1 - layout[:, 1]
    layout[:, 1] = np.nan_to_num(layout[:, 1], nan=0)
    layout[:, 0] = np.nan_to_num(layout[:, 0], nan=0.5)
    return layout


def get_human_readable_prompt_text(messages):
    # Iterate over the messages and print them with headers and demarcations
    prompt_text = ""
    for message in messages:
        role = message["role"]

        role_str = f"R: {role.upper()}"
        header_line = "=" * len(role_str)

        # Print header with color
        prompt_text += f"\n{header_line}\n{role_str}\n{header_line}\n"

        # Wrap each line individually while preserving newline characters
        prompt_text += f"{message['content']}\n\n"

    return prompt_text


def cfg_to_tree_struct(cfg, jou: Journal):
    edges = list(get_edges(jou))
    layout = normalize_layout(generate_layout(len(jou), edges))

    metrics = np.array([0 for n in jou])
    fitnesses = np.array([n.metric.value_npsafe for n in jou]).tolist()
    node_data_list = [jou.get_node_data(n.step) for n in jou]
    metric_infos = [json.dumps(n.metric.info, indent=2) for n in jou]

    prompts = []
    for n in jou:
        prompt_text = ""
        if n.operators_metrics and len(n.operators_metrics) > 0:
            for op_metrics in n.operators_metrics:
                if isinstance(op_metrics, dict):
                    if "prompt_messages" in op_metrics:
                        messages = op_metrics["prompt_messages"].copy()
                    if "completion_text" in op_metrics:
                        messages.append({"role": "assistant", "content": op_metrics["completion_text"]})
                    prompt_text += (
                        get_human_readable_prompt_text(messages)
                        + "\n\n\n ================================================================= \n\n\n"
                    )

        prompts.append(prompt_text)

    return dict(
        edges=edges,
        layout=layout.tolist(),
        plan=[n.plan for n in jou.nodes],
        code=[n.code for n in jou],
        term_out=[n.term_out for n in jou],
        analysis=[n.analysis for n in jou],
        exp_name=getattr(cfg, "id", getattr(cfg, "exp_name", "")),
        metrics=metrics.tolist(),
        fitnesses=fitnesses,
        prompts=prompts,
        node_data_list=node_data_list,
        metric_infos=metric_infos,
    )


def generate_html(tree_graph_str: str):
    template_dir = Path(__file__).parent / "viz_templates"

    with open(template_dir / "template.js") as f:
        js = f.read()
        js = js.replace("null/*<-replacehere*/;", tree_graph_str)

    with open(template_dir / "template.html") as f:
        html = f.read()
        html = html.replace("<!-- placeholder -->", js)

        return html


def generate(cfg, jou: Journal, out_path: Path):
    try:
        tree_graph_str = json.dumps(cfg_to_tree_struct(cfg, jou))
        html = generate_html(tree_graph_str)
        with open(out_path, "w") as f:
            f.write(html)
        return True
    except Exception as e:
        print(f"Couldn't generate visualisation due to {e}")
        return False
