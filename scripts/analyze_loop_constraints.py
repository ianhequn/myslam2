#!/usr/bin/env python3
"""Compare saved loop-constraint translations with KITTI ground truth."""
import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--loops", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    args = parser.parse_args()
    poses = np.loadtxt(args.before, ndmin=2)
    loops = np.loadtxt(args.loops, ndmin=2)
    truth = np.loadtxt(args.ground_truth, ndmin=2)
    if poses.shape[1] != 13 or loops.shape[1] != 10 or truth.shape[1] != 12:
        raise ValueError("Invalid trajectory, loop constraint, or KITTI pose format")
    if not all(np.isfinite(x).all() for x in (poses, loops, truth)):
        raise ValueError("Inputs must contain finite values")
    ids = poses[:, 0].astype(int)
    pairs = loops[:, :2].astype(int)
    if (not np.array_equal(ids, poses[:, 0]) or np.any(ids < 0) or
            np.any(ids >= len(truth)) or not np.array_equal(pairs, loops[:, :2]) or
            np.any(pairs < 0) or np.any(pairs >= len(ids))):
        raise ValueError("Invalid frame or keyframe IDs")
    twc = np.tile(np.eye(4), (len(truth), 1, 1))
    twc[:, :3, :] = truth.reshape(-1, 3, 4)
    relative = np.linalg.inv(twc[ids[pairs[:, 1]]]) @ twc[ids[pairs[:, 0]]]
    true_translation = relative[:, :3, 3]
    errors = np.linalg.norm(loops[:, 3:6] - true_translation, axis=1)
    distances = np.linalg.norm(true_translation, axis=1)
    quaternions = loops[:, 6:10]
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("Invalid loop quaternion")
    x, y, z, w = (quaternions / norms[:, None]).T
    rotations = np.empty((len(loops), 3, 3))
    rotations[:, 0, 0], rotations[:, 0, 1], rotations[:, 0, 2] = 1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)
    rotations[:, 1, 0], rotations[:, 1, 1], rotations[:, 1, 2] = 2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)
    rotations[:, 2, 0], rotations[:, 2, 1], rotations[:, 2, 2] = 2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)
    rotation_error = np.swapaxes(rotations, 1, 2) @ relative[:, :3, :3]
    angles = np.degrees(np.arccos(np.clip((np.trace(rotation_error, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    print(f"Loop edges: {len(loops)}")
    for label, values in (("Ground-truth separation (m)", distances),
                          ("Loop translation error (m)", errors),
                          ("Loop rotation error (degrees)", angles)):
        p = np.percentile(values, [0, 25, 50, 75, 100])
        print(f"{label}: min={p[0]:.3f} p25={p[1]:.3f} median={p[2]:.3f} "
              f"p75={p[3]:.3f} max={p[4]:.3f}")
    print("Translation is compared in the current keyframe camera coordinates.")


if __name__ == "__main__":
    main()
