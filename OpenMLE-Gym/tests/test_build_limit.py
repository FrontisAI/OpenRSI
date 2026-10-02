from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from openmle_gym import build as build_module
from openmle_gym.build import build_tasks
from openmle_gym.common import read_slugs_file


def _slugs_file(directory: str, lines: list[str]) -> Path:
    path = Path(directory) / "slugs.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fake_run_all(**kwargs):
    """Record the slugs the execute path would build, and report success."""
    build_module._RUN_ALL_CALLS.append(list(kwargs["slugs"]))
    return [{"name": slug, "status": "ok", "success": True} for slug in kwargs["slugs"]]


class BuildLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        build_module._RUN_ALL_CALLS = []
        self._patcher = mock.patch.object(
            build_module, "_run_all", side_effect=_fake_run_all
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def _execute(self, directory: str, lines: list[str], limit):
        slugs = _slugs_file(directory, lines)
        info_csv = Path(directory) / "info.csv"
        info_csv.write_text("slug\n", encoding="utf-8")
        result = build_tasks(
            slugs_file=slugs,
            output_root=Path(directory) / "out",
            info_csv=info_csv,
            limit=limit,
            skip_mlebench=False,
            execute=True,
        )
        built = build_module._RUN_ALL_CALLS[0] if build_module._RUN_ALL_CALLS else []
        return result, built

    def _dry_run(self, directory: str, lines: list[str], limit):
        slugs = _slugs_file(directory, lines)
        return build_tasks(
            slugs_file=slugs,
            output_root=Path(directory) / "out",
            limit=limit,
            execute=False,
        )

    def test_duplicates_do_not_consume_limit_slots(self) -> None:
        """The duplicate line must not eat one of the two requested slots."""
        with tempfile.TemporaryDirectory() as directory:
            _, built = self._execute(directory, ["alpha", "alpha", "beta", "gamma"], 2)
            self.assertEqual(built, ["alpha", "beta"])

    def test_execute_matches_dry_run_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dry = self._dry_run(directory, ["alpha", "alpha", "beta", "gamma"], 2)
            _, built = self._execute(directory, ["alpha", "alpha", "beta", "gamma"], 2)
            self.assertEqual(built, dry["selected"])

    def test_no_duplicates_limit_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, built = self._execute(directory, ["alpha", "beta", "gamma", "delta"], 2)
            self.assertEqual(built, ["alpha", "beta"])

    def test_no_limit_builds_every_unique_slug(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, built = self._execute(directory, ["alpha", "beta", "gamma", "delta"], None)
            self.assertEqual(built, ["alpha", "beta", "gamma", "delta"])

    def test_limit_larger_than_input_builds_all(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, built = self._execute(directory, ["alpha", "beta"], 99)
            self.assertEqual(built, ["alpha", "beta"])

    def test_duplicate_is_still_reported_as_a_preflight_failure(self) -> None:
        """Deduplicating the selection must not hide the duplicate from the report."""
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self._execute(directory, ["alpha", "alpha", "beta"], 2)
            errors = [
                error
                for row in result["results"]
                for error in row.get("errors", [])
            ]
            self.assertTrue(any("Duplicate task name" in error for error in errors))

    def test_dry_run_selection_is_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            slugs = _slugs_file(directory, ["alpha", "alpha", "beta"])
            self.assertEqual(read_slugs_file(slugs), ["alpha", "beta"])
            dry = self._dry_run(directory, ["alpha", "alpha", "beta"], 2)
            self.assertEqual(dry["selected"], ["alpha", "beta"])


if __name__ == "__main__":
    unittest.main()