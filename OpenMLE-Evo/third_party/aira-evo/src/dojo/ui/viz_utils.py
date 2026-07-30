# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import yaml
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import json
from typing import Dict, List, Any, Optional
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from omegaconf import OmegaConf


def load_yaml_config(file_path: str) -> Dict:
    """Load YAML configuration file."""
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Failed to load config file {file_path}: {e}")
        return {}


def create_config_graph(config_dir: str, configs: Dict[str, List[str]]) -> nx.DiGraph:
    """Create a directed graph of configuration dependencies."""
    G = nx.DiGraph()

    # Add all configs as nodes
    for category, config_files in configs.items():
        for config_file in config_files:
            G.add_node(config_file, category=category)

    # Add edges based on defaults
    for category, config_files in configs.items():
        for config_file in config_files:
            config_path = os.path.join(config_dir, f"{config_file}.yaml")
            config_data = load_yaml_config(config_path)

            if config_data and "defaults" in config_data:
                defaults = config_data["defaults"]
                if isinstance(defaults, list):
                    for default in defaults:
                        if isinstance(default, dict):
                            # Handle dict format like {'task': 'mlebench/xyz'}
                            for key, value in default.items():
                                # Fix: Handle case when value is a list
                                if isinstance(value, list):
                                    # Handle list of values
                                    for item in value:
                                        if isinstance(item, str) and not item.startswith("/"):
                                            target = f"{key}/{item}"
                                            G.add_edge(config_file, target)
                                elif isinstance(value, str) and not value.startswith("/"):  # Skip absolute paths
                                    target = f"{key}/{value}"
                                    G.add_edge(config_file, target)
                        elif isinstance(default, str) and not default.startswith("/"):
                            # Handle string format like 'mlebench/xyz'
                            G.add_edge(config_file, default)

    return G


def visualize_config_graph(G: nx.DiGraph) -> plt.Figure:
    """Visualize the config graph using matplotlib."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use different colors for different categories
    categories = set(nx.get_node_attributes(G, "category").values())
    color_map = {}
    for i, cat in enumerate(categories):
        color_map[cat] = plt.cm.tab10(i)

    node_colors = [color_map.get(G.nodes[node].get("category", ""), "gray") for node in G.nodes]

    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, label=cat)
        for cat, color in color_map.items()
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_title("Configuration Dependencies")
    ax.axis("off")

    return fig


def create_plotly_config_graph(G: nx.DiGraph) -> go.Figure:
    """Create an interactive Plotly graph of the configuration dependencies."""
    # Use a spring layout for node positions
    pos = nx.spring_layout(G, dim=3, seed=42)

    # Extract node positions
    node_x = [pos[node][0] for node in G.nodes]
    node_y = [pos[node][1] for node in G.nodes]
    node_z = [pos[node][2] for node in G.nodes]

    # Get categories for node colors
    categories = list(set(nx.get_node_attributes(G, "category").values()))
    category_map = {cat: i for i, cat in enumerate(categories)}

    node_colors = [category_map.get(G.nodes[node].get("category", ""), 0) for node in G.nodes]

    # Create nodes trace
    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            color=node_colors,
            size=10,
            colorbar=dict(title="Category", tickvals=list(range(len(categories))), ticktext=categories),
        ),
        text=[f"Config: {node}" for node in G.nodes],
        name="Configs",
    )

    # Create edges trace
    edge_x = []
    edge_y = []
    edge_z = []

    for edge in G.edges:
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(width=1, color="rgba(128, 128, 128, 0.5)"),
        hoverinfo="none",
        name="Dependencies",
    )

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title="Configuration Dependencies (Interactive)",
        showlegend=False,
        scene=dict(
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(showticklabels=False, title=""),
        ),
        margin=dict(b=0, l=0, r=0, t=40),
        hovermode="closest",
    )

    return fig


def create_config_comparison(config_paths: List[str], config_dir: str) -> pd.DataFrame:
    """Create a comparison dataframe for multiple configurations."""
    configs = {}
    all_keys = set()

    # Load all configs and collect all possible keys
    for config_path in config_paths:
        full_path = os.path.join(config_dir, f"{config_path}.yaml")
        config_data = load_yaml_config(full_path)
        configs[config_path] = config_data

        # Recursively get all keys
        def extract_keys(data, prefix=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    all_keys.add(full_key)
                    extract_keys(value, full_key)

        extract_keys(config_data)

    # Create comparison dataframe
    comparison_data = []

    for key in sorted(all_keys):
        row = {"parameter": key}

        for config_path in config_paths:
            config_data = configs[config_path]

            # Traverse the nested dictionary to get the value
            parts = key.split(".")
            value = config_data
            try:
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = None
                        break
            except:
                value = None

            # Convert value to a displayable string
            if isinstance(value, dict) or isinstance(value, list):
                value = str(value)

            row[config_path] = value

        comparison_data.append(row)

    return pd.DataFrame(comparison_data)


def visualize_parameter_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create a heatmap visualization of parameter differences."""
    # Create a binary matrix showing which configs have each parameter
    configs = [col for col in df.columns if col != "parameter"]

    # Convert to binary presence/absence
    binary_data = []
    for _, row in df.iterrows():
        binary_row = [1 if row[config] is not None else 0 for config in configs]
        binary_data.append(binary_row)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=binary_data, x=configs, y=df["parameter"], colorscale=[[0, "white"], [1, "green"]], showscale=False
        )
    )

    fig.update_layout(
        title="Parameter Presence Across Configurations",
        xaxis_title="Configuration",
        yaxis_title="Parameter",
        height=max(500, len(df) * 20),  # Adjust height based on number of parameters
        margin=dict(l=150, r=20, t=30, b=20),
    )

    return fig


def flatten_dict(config: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    """Flatten a nested dictionary."""
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def create_config_diff_viz(config1: Dict, config2: Dict) -> go.Figure:
    """Create a visualization showing differences between two configs."""
    # Flatten both dictionaries
    flat_config1 = flatten_dict(config1)
    flat_config2 = flatten_dict(config2)

    # Get all keys
    all_keys = sorted(set(list(flat_config1.keys()) + list(flat_config2.keys())))

    # Create data for visualization
    data = []
    for key in all_keys:
        val1 = flat_config1.get(key, None)
        val2 = flat_config2.get(key, None)

        # Determine status
        if key not in flat_config1:
            status = "Only in Config 2"
        elif key not in flat_config2:
            status = "Only in Config 1"
        elif val1 != val2:
            status = "Different"
        else:
            status = "Same"

        data.append(
            {
                "parameter": key,
                "status": status,
                "value1": str(val1) if val1 is not None else None,
                "value2": str(val2) if val2 is not None else None,
            }
        )

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Create color map for status
    color_map = {"Same": "green", "Different": "orange", "Only in Config 1": "red", "Only in Config 2": "blue"}

    # Create figure
    fig = px.scatter(
        df,
        x="status",
        y="parameter",
        color="status",
        color_discrete_map=color_map,
        hover_data=["value1", "value2"],
        title="Configuration Differences",
        labels={"parameter": "Configuration Parameter", "status": "Status"},
    )

    fig.update_layout(height=max(500, len(df) * 15), legend_title="Status", margin=dict(l=150, r=20, t=30, b=20))

    return fig
