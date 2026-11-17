"""Exercise the pose-graph replay executable with a small synthetic graph."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXECUTABLE = Path(__file__).resolve().parents[1] / "bin" / "replay_pose_graph"


@unittest.skipUnless(sys.platform.startswith("linux") and EXECUTABLE.exists(),
                     "Linux replay executable is not available")
class ReplayPoseGraphTest(unittest.TestCase):
    def test_correct_loop_closes_synthetic_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before, loops, after = (root / name for name in ("before.txt", "loops.txt", "after.txt"))
            before.write_text("".join(
                f"{i} 1 0 0 {-1.1 * i} 0 1 0 0 0 0 1 0\n" for i in range(11)))
            loops.write_text("0 10 30 10 0 0 0 0 0 1\n0 9 30 100 0 0 0 0 0 1\n")
            result = subprocess.run([str(EXECUTABLE), str(before), str(loops),
                                     str(after), "100", "1", "30", "lm", "20"],
                                    text=True, capture_output=True, check=True)
            self.assertIn("loop_edges=1", result.stdout)
            self.assertIn("filtered=1", result.stdout)
            output = after.read_text().splitlines()
            self.assertEqual(len(output), 11)
            self.assertAlmostEqual(float(output[-1].split()[4]), -10.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
