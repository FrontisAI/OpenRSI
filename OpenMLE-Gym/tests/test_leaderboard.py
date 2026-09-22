from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openmle_gym.leaderboard import summarize_leaderboard


class SummarizeLeaderboardTests(unittest.TestCase):
    def test_score_column_statistics(self) -> None:
        header = "TeamId,TeamName,SubmissionDate,Score\n"
        rows = "".join(f"{i},team{i},2020-01-01,{float(i)}\n" for i in range(1, 11))
        summary = summarize_leaderboard((header + rows).encode("utf-8"))
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["score_column"], "Score")
        self.assertEqual(summary["count"], 10)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 10.0)
        self.assertEqual(summary["median"], 5.5)
        self.assertAlmostEqual(summary["p10"], 1.9)
        self.assertAlmostEqual(summary["p90"], 9.1)

    def test_partial_score_header_is_detected(self) -> None:
        content = b"Rank,PublicScore\n1,0.9\n2,0.8\n3,0.7\n"
        summary = summarize_leaderboard(content)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["score_column"], "PublicScore")
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["max"], 0.9)

    def test_falls_back_to_rightmost_numeric_column(self) -> None:
        content = b"team,value\nalpha,3\nbravo,1\ncharlie,2\n"
        summary = summarize_leaderboard(content)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["score_column"], "value")
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 3.0)

    def test_reads_from_a_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "titanic.csv"
            path.write_text("TeamName,Score\na,0.5\nb,0.75\n", encoding="utf-8")
            summary = summarize_leaderboard(path)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["median"], 0.625)

    def test_non_numeric_score_column_is_unavailable(self) -> None:
        content = b"TeamName,Score\nalpha,high\nbravo,low\n"
        summary = summarize_leaderboard(content)
        self.assertEqual(summary["status"], "unavailable")

    def test_header_only_is_unavailable(self) -> None:
        summary = summarize_leaderboard(b"TeamName,Score\n")
        self.assertEqual(summary["status"], "unavailable")
        self.assertEqual(summary["reason"], "no leaderboard rows")


if __name__ == "__main__":
    unittest.main()
