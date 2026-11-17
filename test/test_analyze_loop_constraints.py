"""Check loop-pose conventions against a tiny synthetic trajectory."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_loop_constraints.py"


class AnalyzeLoopConstraintsTest(unittest.TestCase):
    def test_relative_pose_and_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            poses = np.tile(np.eye(4)[:3].reshape(1, 12), (2, 1))
            poses[1, 3] = 2.0
            np.savetxt(root / "truth.txt", poses)
            np.savetxt(root / "before.txt", np.column_stack(([0, 1], poses)))
            np.savetxt(root / "loops.txt", [[0, 1, 30, -2, 0, 0, 0, 0, 0, 1]])
            result = subprocess.run([sys.executable, str(SCRIPT), "--before",
                                     str(root / "before.txt"), "--loops", str(root / "loops.txt"),
                                     "--ground-truth", str(root / "truth.txt")],
                                    check=True, text=True, capture_output=True)
            self.assertIn("Loop translation error (m): min=0.000", result.stdout)
            self.assertIn("Loop rotation error (degrees): min=0.000", result.stdout)


if __name__ == "__main__":
    unittest.main()
