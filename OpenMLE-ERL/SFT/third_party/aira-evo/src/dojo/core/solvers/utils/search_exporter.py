# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime
import json
import logging
import tempfile
from pathlib import Path

from dataclasses import asdict

from omegaconf import DictConfig, OmegaConf

import wandb
from dojo.core.solvers.utils import tree_export
from dojo.core.solvers.utils.journal import Journal, Node
from dojo.core.solvers.utils.metric import MetricValue
from dojo.utils.logger import CollectiveLogger, LogEvent

logger = logging.getLogger(__name__)


class SearchExporter:
    def __init__(self, journal: Journal, cfg):
        self.journal = journal
        self.cfg = cfg

    def gather_and_export_search_results(
        self, tree_path: str | Path | None = None, output_file: str | Path | None = None
    ) -> dict:
        """
        Gather search data into a dictionary structure and optionally
        export it to a JSON file. Allows for a dynamically specified `tree_path`
        for the generated HTML visualization.

        Args:
            tree_path (str | Path | None, optional):
                Where to generate the tree visualization HTML. If None,
                the tree export is skipped, and the returned data has `tree_path=None`.
            output_file (str | Path | None, optional):
                If provided, the search data is written out to this file in JSON format.

        Returns:
            dict: A dictionary containing:
                - "nodes": a list of node dictionaries
                - "config": the config used by Greedy
                - "tree_path": the HTML path for tree visualization (if any)
                - "solution": the best solution code (if any)
        """

        journal_data = self.journal.export_data()

        tree_path_str = None
        if tree_path is not None:
            tree_path = Path(tree_path)
            if tree_export.generate(self.cfg, self.journal, tree_path):
                tree_path_str = str(tree_path.absolute())

        search_data = {
            **journal_data,
            "config": asdict(self.cfg),
            "tree_path": tree_path_str,
        }

        if output_file:
            output_file = Path(output_file)
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(search_data, f, indent=2)
            logger.info(f"Exported search results to {output_file.resolve()}")

        return search_data


def export_search_results(cfg: DictConfig, journal: Journal, logger: CollectiveLogger, search_method: str):
    if cfg.export_search_results:
        exp_name = cfg.exp_name
        time = datetime.now().strftime("%Y%m%d%H%M%S%f")
        default_tree_path = f"{logger.cfg.logger.output_dir}/{exp_name}_{time}_{search_method}_tree.html"
        default_data_path = f"{logger.cfg.logger.output_dir}/{exp_name}_{time}_{search_method}_search_data.json"
        search_exporter = SearchExporter(journal, cfg)
        search_data = search_exporter.gather_and_export_search_results(default_tree_path, default_data_path)
        logger.info(f"Visualisation exported to {default_data_path} and {default_tree_path}.", LogEvent.SOLVER)
        data_path = Path(default_data_path).absolute()
        logger.log_file(data_path)

        tree_path = search_data.get("tree_path")
        if tree_path:
            tree_path = Path(tree_path).absolute()
            logger.log_file(tree_path)
            with tree_path.open(encoding="utf-8") as tree_file:
                logger.log({"tree_vis": wandb.Html(tree_file)}, LogEvent.SOLVER)
        else:
            logger.info("Tree visualisation was not generated; exported search data only.", LogEvent.SOLVER)


def test_export_and_reconstruct_search_data():
    # Create a simple test journal with multiple nodes and branches
    journal = Journal()

    # Create root node
    root = Node(
        code="def hello(): print('hello')",
        plan="Initial implementation",
        analysis="Works fine",
        metric=MetricValue(value=0.8, info={"accuracy": 0.8}, maximize=True),
        is_buggy=False,
        messages=[{"role": "user", "content": "Write a hello function"}],
        operator_used="initial_solution",
    )
    journal.append(root)

    # Create first improvement child
    child1 = Node(
        code="def hello(): print('hello world')",
        plan="Improved version",
        analysis="Better output",
        metric=MetricValue(value=0.9, info={"accuracy": 0.9}, maximize=True),
        is_buggy=False,
        messages=[{"role": "user", "content": "Improve the function"}],
        operator_used="improve",
        parents=[root],
    )
    journal.append(child1)

    # Create a buggy child
    buggy_child = Node(
        code="def hello() print('hello world')",  # Missing colon
        plan="Attempt to simplify",
        analysis="Syntax error",
        metric=MetricValue(value=0.0, info={"error": "syntax"}, maximize=True),
        is_buggy=True,
        messages=[{"role": "user", "content": "Simplify the function"}],
        operator_used="improve",
        parents=[child1],
    )
    journal.append(buggy_child)

    # Create debug fix for buggy child
    debug_fix = Node(
        code="def hello(): print('hello world')",
        plan="Fix syntax error",
        analysis="Fixed missing colon",
        metric=MetricValue(value=0.9, info={"accuracy": 0.9}, maximize=True),
        is_buggy=False,
        messages=[{"role": "user", "content": "Fix the syntax error"}],
        operator_used="debug",
        parents=[buggy_child],
    )
    journal.append(debug_fix)

    # Create alternative improvement branch
    alt_child = Node(
        code="def hello(name='world'): print(f'hello {name}')",
        plan="Add parameter",
        analysis="Added name parameter",
        metric=MetricValue(value=0.95, info={"accuracy": 0.95}, maximize=True),
        is_buggy=False,
        messages=[{"role": "user", "content": "Add parameter support"}],
        operator_used="improve",
        parents=[child1],
    )
    journal.append(alt_child)

    # Create final improvement
    final_node = Node(
        code="def hello(name: str = 'world') -> None: print(f'hello {name}')",
        plan="Add type hints",
        analysis="Added type annotations",
        metric=MetricValue(value=1.0, info={"accuracy": 1.0}, maximize=True),
        is_buggy=False,
        messages=[{"role": "user", "content": "Add type hints"}],
        operator_used="improve",
        parents=[alt_child],
    )
    journal.append(final_node)

    # Create test config with required fields
    cfg = OmegaConf.create({"test_key": "test_value", "exp_name": "test_experiment", "export_search_results": True})

    # Export the search data using temporary directory without context manager
    temp_dir = Path(tempfile.mkdtemp())
    try:
        output_file = temp_dir / "search_data.json"
        tree_path = temp_dir / "tree.html"

        # Export original journal
        exporter = SearchExporter(journal, cfg)
        search_data = exporter.gather_and_export_search_results(tree_path=tree_path, output_file=output_file)

        # Verify the file was created
        assert output_file.exists()

        # Load the data back
        with output_file.open() as f:
            loaded_data = json.load(f)

        # Reconstruct journal from the loaded data
        reconstructed_journal = Journal.from_export_data(loaded_data)

        # Generate visualizations for both journals
        orig_vis_path = temp_dir / "original_tree.html"
        recon_vis_path = temp_dir / "reconstructed_tree.html"

        tree_export.generate(cfg, journal, str(orig_vis_path))
        tree_export.generate(cfg, reconstructed_journal, str(recon_vis_path))

        # Compare the visualization files if they exist
        if orig_vis_path.exists() and recon_vis_path.exists():
            with orig_vis_path.open("r") as f1, recon_vis_path.open("r") as f2:
                orig_vis = f1.read()
                recon_vis = f2.read()
                assert orig_vis == recon_vis, "Visualizations differ between original and reconstructed journals"

        # Verify the reconstruction
        assert len(reconstructed_journal.nodes) == len(journal.nodes) == 6, "Journal should have exactly 6 nodes"

        # Check node properties are preserved
        for orig_node, recon_node in zip(journal.nodes, reconstructed_journal.nodes):
            assert orig_node.code == recon_node.code, f"Code mismatch for node {orig_node.id}"
            assert orig_node.plan == recon_node.plan, f"Plan mismatch for node {orig_node.id}"
            assert orig_node.step == recon_node.step, f"Step mismatch for node {orig_node.id}"
            assert orig_node.id == recon_node.id, f"ID mismatch for node {orig_node.id}"
            assert orig_node.metric.value == recon_node.metric.value, f"Metric value mismatch for node {orig_node.id}"
            assert orig_node.metric.info == recon_node.metric.info, f"Metric info mismatch for node {orig_node.id}"
            assert orig_node.is_buggy == recon_node.is_buggy, f"Buggy status mismatch for node {orig_node.id}"
            assert orig_node.messages == recon_node.messages, f"Messages mismatch for node {orig_node.id}"
            assert orig_node.operator_used == recon_node.operator_used, f"Operator mismatch for node {orig_node.id}"

        # Get reconstructed nodes for relationship testing
        [
            root_recon,
            child1_recon,
            buggy_recon,
            debug_recon,
            alt_child_recon,
            final_recon,
        ] = reconstructed_journal.nodes

        # Check relationships with detailed messages
        assert root_recon.parents is None, "Root node should have no parents"
        assert len(root_recon.children) == 1, "Root node should have exactly one child"
        assert child1_recon in root_recon.children, "Root's child should be child1"

        assert len(child1_recon.parents) == 1, "child1 should have exactly one parent"
        assert root_recon in child1_recon.parents, "child1's parent should be root"
        assert len(child1_recon.children) == 2, "child1 should have exactly two children"
        assert buggy_recon in child1_recon.children, "child1's children should include buggy_child"
        assert alt_child_recon in child1_recon.children, "child1's children should include alt_child"

        assert len(buggy_recon.children) == 1, "buggy_child should have exactly one child"
        assert debug_recon in buggy_recon.children, "buggy_child's child should be debug_fix"
        assert len(buggy_recon.parents) == 1, "buggy_child should have exactly one parent"
        assert child1_recon in buggy_recon.parents, "buggy_child's parent should be child1"

        assert len(debug_recon.children) == 0, "debug_fix should have no children"
        assert debug_recon.parents == [buggy_recon], "debug_fix's parent should be buggy_child"

        assert len(alt_child_recon.children) == 1, "alt_child should have exactly one child"
        assert final_recon in alt_child_recon.children, "alt_child's child should be final_node"
        assert len(alt_child_recon.parents) == 1, "alt_child should have exactly one parent"
        assert child1_recon in alt_child_recon.parents, "alt_child's parent should be child1"

        assert len(final_recon.children) == 0, "final_node should have no children"
        assert final_recon.parents == [alt_child_recon], "final_node's parent should be alt_child"

        # Check metric values and ordering
        assert root_recon.metric.value == 0.8, "Root node should have metric value 0.8"
        assert child1_recon.metric.value == 0.9, "child1 should have metric value 0.9"
        assert buggy_recon.metric.value == 0.0, "buggy_child should have metric value 0.0"
        assert debug_recon.metric.value == 0.9, "debug_fix should have metric value 0.9"
        assert alt_child_recon.metric.value == 0.95, "alt_child should have metric value 0.95"
        assert final_recon.metric.value == 1.0, "final_node should have metric value 1.0"

        # Check stage names with detailed messages
        assert root_recon.stage_name == "draft", "Root node should be in 'draft' stage"
        assert child1_recon.stage_name == "improve", "child1 should be in 'improve' stage"
        assert buggy_recon.stage_name == "improve", "buggy_child should be in 'improve' stage"
        assert debug_recon.stage_name == "debug", "debug_fix should be in 'debug' stage"
        assert alt_child_recon.stage_name == "improve", "alt_child should be in 'improve' stage"
        assert final_recon.stage_name == "improve", "final_node should be in 'improve' stage"

        # Check debug depths with detailed messages
        assert root_recon.debug_depth == 0, "Root node should have debug_depth 0"
        assert child1_recon.debug_depth == 0, "child1 should have debug_depth 0"
        assert buggy_recon.debug_depth == 0, "buggy_child should have debug_depth 0"
        assert debug_recon.debug_depth == 1, "debug_fix should have debug_depth 1"
        assert alt_child_recon.debug_depth == 0, "alt_child should have debug_depth 0"
        assert final_recon.debug_depth == 0, "final_node should have debug_depth 0"

        # Check leaf status with detailed messages
        assert not root_recon.is_leaf, "Root node should not be a leaf"
        assert not child1_recon.is_leaf, "child1 should not be a leaf"
        assert not buggy_recon.is_leaf, "buggy_child should not be a leaf"
        assert debug_recon.is_leaf, "debug_fix should be a leaf"
        assert not alt_child_recon.is_leaf, "alt_child should not be a leaf"
        assert final_recon.is_leaf, "final_node should be a leaf"

        # Check best node identification
        best_node = journal.get_best_node()
        print(f"Original best node ID: {best_node.id}")
        print(f"Final node ID: {final_node.id}")
        print(f"Original best node code: {best_node.code}")
        print(f"Final node code: {final_node.code}")
        print(f"Original best node metric: {best_node.metric.value}")
        print(f"Final node metric: {final_node.metric.value}")

        assert best_node.code == final_node.code, "Best node code doesn't match final node"
        assert best_node.metric.value == final_node.metric.value == 1.0, "Best node metric doesn't match final node"
        assert best_node.id == final_node.id, "Best node ID doesn't match final node"

        # Check reconstructed journal's best node
        recon_best = reconstructed_journal.get_best_node()
        print(f"Reconstructed best node ID: {recon_best.id}")
        print(f"Reconstructed best node code: {recon_best.code}")
        print(f"Reconstructed best node metric: {recon_best.metric.value}")

        # Additional verification tests:

        # Test metric info preservation
        assert root_recon.metric.info == {"accuracy": 0.8}, "Metric info not preserved for root node"
        assert buggy_recon.metric.info == {"error": "syntax"}, "Metric info not preserved for buggy node"

        # Test creation time preservation
        assert root_recon.ctime == root.ctime, "Creation time not preserved for root node"
        assert final_recon.ctime == final_node.ctime, "Creation time not preserved for final node"

        # Test execution information preservation
        assert root_recon._term_out == root._term_out, "Terminal output not preserved"
        assert root_recon.exec_time == root.exec_time, "Execution time not preserved"
        assert root_recon.exit_code == root.exit_code, "Exception code not preserved"

        # Test message history preservation
        for orig_node, recon_node in zip(journal.nodes, reconstructed_journal.nodes):
            assert recon_node.messages == orig_node.messages, f"Messages not preserved for node {orig_node.id}"

        # Test operator preservation
        assert root_recon.operator_used == "initial_solution", "Operator not preserved for root node"
        assert debug_recon.operator_used == "debug", "Operator not preserved for debug node"
        assert final_recon.operator_used == "improve", "Operator not preserved for final node"

        # Test step numbering preservation and continuity
        steps = [node.step for node in reconstructed_journal.nodes]
        assert steps == list(range(len(steps))), "Step numbering not continuous or preserved"

        # Test UUID preservation
        for orig_node, recon_node in zip(journal.nodes, reconstructed_journal.nodes):
            assert orig_node.id == recon_node.id, f"UUID not preserved for node {orig_node.id}"

        # Test plan preservation
        assert root_recon.plan == "Initial implementation", "Plan not preserved for root node"
        assert final_recon.plan == "Add type hints", "Plan not preserved for final node"

        # Test analysis preservation
        assert root_recon.analysis == "Works fine", "Analysis not preserved for root node"
        assert buggy_recon.analysis == "Syntax error", "Analysis not preserved for buggy node"

        # Test that the reconstructed journal maintains the same order
        for i, (orig_node, recon_node) in enumerate(zip(journal.nodes, reconstructed_journal.nodes)):
            assert orig_node.step == recon_node.step == i, f"Node order not preserved at position {i}"

        # Test that the search data contains all required fields
        required_fields = {"nodes", "config", "tree_path", "solution"}
        assert all(field in search_data for field in required_fields), "Missing required fields in search data"

        # Test that the config is properly preserved
        assert search_data["config"]["test_key"] == "test_value", "Config values not preserved"
        assert search_data["config"]["exp_name"] == "test_experiment", "Config values not preserved"

        # Test that the best solution is properly preserved
        assert search_data["solution"] == final_node.code, "Best solution code not preserved"

        print("All reconstruction tests passed successfully!")

    finally:
        # Clean up temporary directory
        import shutil

        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_export_and_reconstruct_search_data()
