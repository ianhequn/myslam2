"""Small end-to-end checks for the trajectory evaluation tool."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_trajectories.py"


class TrajectoryComparisonTest(unittest.TestCase):
    def test_shared_alignment_and_frame_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = np.arange(5)
            poses = np.tile(np.eye(4)[:3].reshape(1, 12), (len(ids), 1))
            poses[:, 3] = ids
            poses[:, 11] = ids ** 2
            estimated = poses.copy()
            estimated[:, 3] += 10
            after = estimated.copy()
            after[:, 3] += 2
            np.savetxt(root / "truth.txt", poses)
            np.savetxt(root / "before.txt", np.column_stack((ids, estimated)))
            np.savetxt(root / "after.txt", np.column_stack((ids, after)))
            command = [sys.executable, str(SCRIPT), "--before", str(root / "before.txt"),
                       "--after", str(root / "after.txt"), "--ground-truth",
                       str(root / "truth.txt"), "--output", str(root / "plot.png")]
            result = subprocess.run(command, text=True, capture_output=True, check=True)
            self.assertIn("before: 0.000 m", result.stdout)
            self.assertIn("after:  2.000 m", result.stdout)
            self.assertTrue((root / "plot.png").is_file())

            np.savetxt(root / "after.txt", np.column_stack((ids + 1, after)))
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("different frame ids", result.stderr)


if __name__ == "__main__":
    unittest.main()
