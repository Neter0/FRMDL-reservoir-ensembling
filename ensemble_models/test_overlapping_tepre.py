import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ensemble_lsm import simple_ensemble_lsm
from partition_schedules import (
    build_temporal_partition_schedule,
    describe_temporal_partition_schedule,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a one-shot baseline-vs-overlap TEPRE experiment on N-MNIST."
    )
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--num-res", type=int, default=3)
    parser.add_argument("--num-partitions", type=int, default=3)
    parser.add_argument("--in-conn", type=float, default=0.15)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--seed-base", type=int, default=1234)
    parser.add_argument(
        "--overlaps",
        type=float,
        nargs="+",
        default=[0.0, 0.10, 0.15, 0.20, 0.30],
        help="Fractions of a partition window shared around temporal boundaries.",
    )
    parser.add_argument(
        "--overlap-combine",
        choices=["mean", "sum"],
        default="mean",
        help="How to combine currents when multiple partitions are active.",
    )
    parser.add_argument(
        "--example-steps",
        type=int,
        default=100,
        help="Only used for human-readable temporal-window reporting.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for all outputs. Defaults to ./results/overlapping_tepre_<timestamp>.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_results_dir(results_dir):
    if results_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results") / ("overlapping_tepre_" + stamp)
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=False)
    return results_dir


def schedule_rows(args):
    rows = []
    for overlap_fraction in args.overlaps:
        schedule = build_temporal_partition_schedule(
            args.example_steps,
            args.num_partitions,
            overlap_fraction=overlap_fraction,
        )
        windows = describe_temporal_partition_schedule(
            args.example_steps,
            args.num_partitions,
            overlap_fraction=overlap_fraction,
        )
        active_counts = [len(active_parts) for active_parts in schedule]
        rows.append(
            {
                "overlap_fraction": overlap_fraction,
                "example_steps": args.example_steps,
                "windows": windows,
                "mean_active_partitions_per_step": float(np.mean(active_counts)),
                "overlap_steps": int(sum(count > 1 for count in active_counts)),
                "non_overlap_steps": int(sum(count == 1 for count in active_counts)),
            }
        )
    return rows


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(raw_rows):
    grouped = {}
    for row in raw_rows:
        grouped.setdefault(row["overlap_fraction"], []).append(row)

    baseline_by_rep = {
        row["rep"]: row["score"]
        for row in grouped.get(0.0, [])
    }
    summary_rows = []
    for overlap_fraction in sorted(grouped):
        rows = grouped[overlap_fraction]
        scores = np.array([row["score"] for row in rows], dtype=np.float64)
        deltas = np.array(
            [
                row["score"] - baseline_by_rep[row["rep"]]
                for row in rows
                if row["rep"] in baseline_by_rep
            ],
            dtype=np.float64,
        )
        summary_rows.append(
            {
                "overlap_fraction": overlap_fraction,
                "n": int(scores.size),
                "mean_accuracy": float(np.mean(scores)),
                "std_accuracy": float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0,
                "min_accuracy": float(np.min(scores)),
                "max_accuracy": float(np.max(scores)),
                "mean_delta_vs_strict": float(np.mean(deltas)) if deltas.size else 0.0,
                "std_delta_vs_strict": float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0,
            }
        )
    return summary_rows


def main():
    args = parse_args()
    if args.reps < 1:
        raise ValueError("--reps must be at least 1")
    if args.num_res < 1:
        raise ValueError("--num-res must be at least 1")
    if args.num_partitions < 1:
        raise ValueError("--num-partitions must be at least 1")
    if 0.0 not in args.overlaps:
        args.overlaps = [0.0] + args.overlaps
    args.overlaps = sorted(set(args.overlaps))

    results_dir = make_results_dir(args.results_dir)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    metadata = {
        "started_at": started_at,
        "reps": args.reps,
        "num_res": args.num_res,
        "num_partitions": args.num_partitions,
        "in_conn": args.in_conn,
        "nz": args.nz,
        "seed_base": args.seed_base,
        "overlaps": args.overlaps,
        "overlap_combine": args.overlap_combine,
        "example_steps": args.example_steps,
        "strict_baseline_note": "overlap_fraction=0.0 is the strict non-overlapping TEPRE baseline used for comparison.",
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    schedules = schedule_rows(args)
    write_csv(
        results_dir / "schedules.csv",
        schedules,
        [
            "overlap_fraction",
            "example_steps",
            "windows",
            "mean_active_partitions_per_step",
            "overlap_steps",
            "non_overlap_steps",
        ],
    )

    print("overlapping TEPRE one-shot experiment")
    print("results directory:", results_dir)
    print("strict non-overlapping TEPRE baseline is overlap_fraction=0.0")
    for row in schedules:
        print("overlap", row["overlap_fraction"], "windows", row["windows"])

    raw_rows = []
    for rep in range(args.reps):
        rep_seed = args.seed_base + rep
        print("rep", rep + 1, "of", args.reps, "seed", rep_seed)
        for overlap_fraction in args.overlaps:
            set_seed(rep_seed)
            print("running overlap fraction", overlap_fraction)
            score = simple_ensemble_lsm(
                args.in_conn,
                num_res=args.num_res,
                num_partitions=args.num_partitions,
                Nz=args.nz,
                overlap_fraction=overlap_fraction,
                overlap_combine=args.overlap_combine,
            )
            raw_rows.append(
                {
                    "rep": rep,
                    "seed": rep_seed,
                    "in_conn": args.in_conn,
                    "num_res": args.num_res,
                    "num_partitions": args.num_partitions,
                    "nz": args.nz,
                    "overlap_fraction": overlap_fraction,
                    "overlap_combine": args.overlap_combine,
                    "score": float(score),
                }
            )
            write_csv(
                results_dir / "raw_runs.csv",
                raw_rows,
                [
                    "rep",
                    "seed",
                    "in_conn",
                    "num_res",
                    "num_partitions",
                    "nz",
                    "overlap_fraction",
                    "overlap_combine",
                    "score",
                ],
            )

    summary_rows = summarize(raw_rows)
    write_csv(
        results_dir / "summary.csv",
        summary_rows,
        [
            "overlap_fraction",
            "n",
            "mean_accuracy",
            "std_accuracy",
            "min_accuracy",
            "max_accuracy",
            "mean_delta_vs_strict",
            "std_delta_vs_strict",
        ],
    )

    finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata["finished_at"] = finished_at
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    np.savez(
        results_dir / "overlapping_tepre_results.npz",
        raw_runs=np.array([tuple(row.values()) for row in raw_rows], dtype=object),
        raw_run_columns=np.array(list(raw_rows[0].keys()), dtype=object),
        summary=np.array([tuple(row.values()) for row in summary_rows], dtype=object),
        summary_columns=np.array(list(summary_rows[0].keys()), dtype=object),
        schedules=np.array([tuple(row.values()) for row in schedules], dtype=object),
        schedule_columns=np.array(list(schedules[0].keys()), dtype=object),
        metadata=np.array([metadata], dtype=object),
    )

    print("saved raw runs to", results_dir / "raw_runs.csv")
    print("saved summary to", results_dir / "summary.csv")
    print("saved schedules to", results_dir / "schedules.csv")
    print("saved metadata to", results_dir / "metadata.json")
    print("saved numpy archive to", results_dir / "overlapping_tepre_results.npz")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
