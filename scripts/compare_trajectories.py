#!/usr/bin/env python3
"""Compare keyframe camera positions against KITTI ground truth."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_keyframes(path: Path):
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 13 or len(data) < 3 or not np.isfinite(data).all():
        raise ValueError(f"{path}: expected at least three finite rows of frame id + 12 pose values")
    ids = data[:, 0].astype(int)
    if not np.array_equal(ids, data[:, 0]) or np.any(ids < 0) or np.any(np.diff(ids) <= 0):
        raise ValueError(f"{path}: frame ids must be unique, increasing nonnegative integers")
    return ids, data[:, 1:].reshape(-1, 3, 4)[:, :, 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=Path("results/keyframes_before.txt"))
    parser.add_argument("--after", type=Path, default=Path("results/keyframes_after.txt"))
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/trajectory_comparison.png"))
    args = parser.parse_args()

    ids, before = load_keyframes(args.before)
    after_ids, after = load_keyframes(args.after)
    if not np.array_equal(ids, after_ids):
        raise ValueError("Before and after trajectories contain different frame ids")
    truth = np.loadtxt(args.ground_truth, ndmin=2)
    if truth.shape[1] != 12 or ids[-1] >= len(truth) or not np.isfinite(truth).all():
        raise ValueError("Invalid or incomplete KITTI ground truth")
    reference = truth.reshape(-1, 3, 4)[ids, :, 3]

    # Fit one rigid transform on the pre-optimization trajectory; reuse it for both.
    source_center, target_center = before.mean(axis=0), reference.mean(axis=0)
    u, _, vt = np.linalg.svd((before - source_center).T @ (reference - target_center))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    before = (before - source_center) @ rotation + target_center
    after = (after - source_center) @ rotation + target_center
    errors_before = np.linalg.norm(before - reference, axis=1)
    errors_after = np.linalg.norm(after - reference, axis=1)
    rmse_before = np.sqrt(np.mean(errors_before ** 2))
    rmse_after = np.sqrt(np.mean(errors_after ** 2))
    print(f"Keyframes: {len(ids)}; frame IDs: {ids[0]}..{ids[-1]}")
    print(f"Translation ATE RMSE before: {rmse_before:.3f} m")
    print(f"Translation ATE RMSE after:  {rmse_after:.3f} m")
    print("Both estimates use the same SE(3) alignment fitted to the before trajectory; no scale fitting.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for points, label in [(reference, "Ground truth"), (before, "Before"), (after, "After")]:
        axes[0].plot(points[:, 0], points[:, 2], label=label, linewidth=1)
    axes[0].set(xlabel="x (m)", ylabel="z (m)", title="Keyframe trajectories")
    axes[0].axis("equal")
    axes[0].legend()
    axes[1].plot(ids, errors_before, label=f"Before: {rmse_before:.2f} m RMSE")
    axes[1].plot(ids, errors_after, label=f"After: {rmse_after:.2f} m RMSE")
    axes[1].set(xlabel="Frame ID", ylabel="Translation error (m)", title="Error against ground truth")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
