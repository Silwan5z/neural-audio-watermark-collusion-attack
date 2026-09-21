from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from merge_average import merge_one  # noqa: E402


FIELDS = ("model", "k", "trial_id", "value")


def write(path: Path, trials: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for trial in trials:
            writer.writerow({
                "model": "audioseal", "k": 2,
                "trial_id": trial, "value": trial,
            })


class MergeAverageTest(unittest.TestCase):
    def test_merges_complete_disjoint_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / "k2" / "audioseal.shard0of2.csv", [0, 2])
            write(root / "k2" / "audioseal.shard1of2.csv", [1, 3])
            output = merge_one(root, "audioseal", 2, 4)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["trial_id"]) for row in rows],
                             [0, 1, 2, 3])

    def test_rejects_missing_trial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / "k2" / "audioseal.shard0of2.csv", [0, 2])
            write(root / "k2" / "audioseal.shard1of2.csv", [1])
            with self.assertRaises(RuntimeError):
                merge_one(root, "audioseal", 2, 4)


if __name__ == "__main__":
    unittest.main()
