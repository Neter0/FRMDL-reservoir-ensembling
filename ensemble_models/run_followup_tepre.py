import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader

from ensemble_lsm import simple_ensemble_lsm
from partition_schedules import describe_temporal_partition_schedule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Focused follow-up runs for overlapping TEPRE without overwriting prior results."
    )
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=1234)
    parser.add_argument("--num-res", type=int, default=3)
    parser.add_argument("--num-partitions", type=int, default=3)
    parser.add_argument("--in-conn", type=float, default=0.15)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--overlaps", type=float, nargs="+", default=[0.0, 0.30])
    parser.add_argument("--schedule-modes", nargs="+", default=["uniform", "saccade"])
    parser.add_argument("--overlap-modes", nargs="+", default=["symmetric"])
    parser.add_argument("--overlap-combine", choices=["mean", "sum"], default="mean")
    parser.add_argument("--frame-window-us", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None, help="Optional explicit torch device, e.g. cuda or cpu.")
    parser.add_argument("--reuse-results-dir", default=None, help="Existing result folder to reuse matching rows from.")
    parser.add_argument("--event-density-boundary-batches", type=int, default=20)
    parser.add_argument("--results-dir", default=None)
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
        results_dir = Path("results") / ("followup_tepre_" + stamp)
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=False)
    return results_dir


def read_reusable_rows(path):
    if path is None:
        return {}
    raw_path = Path(path) / "raw_runs.csv"
    if not raw_path.exists():
        return {}

    rows = {}
    with raw_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            schedule_mode = row.get("schedule_mode", "uniform")
            overlap_mode = row.get("overlap_mode", "symmetric" if float(row["overlap_fraction"]) > 0.0 else "none")
            key = (int(row["rep"]), int(row["seed"]), float(row["overlap_fraction"]), schedule_mode, overlap_mode)
            rows[key] = {
                "rep": int(row["rep"]),
                "seed": int(row["seed"]),
                "in_conn": float(row["in_conn"]),
                "num_res": int(row["num_res"]),
                "num_partitions": int(row["num_partitions"]),
                "nz": int(row["nz"]),
                "overlap_fraction": float(row["overlap_fraction"]),
                "overlap_combine": row["overlap_combine"],
                "schedule_mode": schedule_mode,
                "overlap_mode": overlap_mode,
                "score": float(row["score"]),
                "source": str(raw_path),
            }
    return rows


def compute_event_density_boundaries(args):
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.Compose(
        [
            transforms.Denoise(filter_time=3000),
            transforms.ToFrame(sensor_size=sensor_size, time_window=args.frame_window_us),
        ]
    )
    trainset = tonic.datasets.NMNIST(save_to="../data", transform=frame_transform, train=True)
    cached_trainset = DiskCachedDataset(trainset, cache_path="../cache/nmnist/train")
    trainloader = DataLoader(
        cached_trainset,
        batch_size=args.batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        shuffle=False,
    )

    profile = None
    for batch_idx, (data, targets) in enumerate(trainloader):
        if batch_idx >= args.event_density_boundary_batches:
            break
        activity = data.reshape(data.shape[0], data.shape[1], -1).sum(dim=(1, 2)).numpy()
        if profile is None:
            profile = np.zeros_like(activity, dtype=np.float64)
        if activity.shape[0] > profile.shape[0]:
            profile = np.pad(profile, (0, activity.shape[0] - profile.shape[0]))
        profile[:activity.shape[0]] += activity

    if profile is None or np.sum(profile) == 0:
        return None

    cumulative = np.cumsum(profile)
    total = cumulative[-1]
    boundaries = [0]
    for part in range(1, args.num_partitions):
        target = total * part / args.num_partitions
        boundaries.append(int(np.searchsorted(cumulative, target)))
    boundaries.append(int(profile.shape[0]))
    return boundaries


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(raw_rows):
    grouped = {}
    for row in raw_rows:
        key = (row["schedule_mode"], row["overlap_fraction"], row["overlap_mode"])
        grouped.setdefault(key, []).append(row)

    baseline_by_mode_rep = {
        (row["schedule_mode"], row["rep"]): row["score"]
        for row in raw_rows
        if row["overlap_fraction"] == 0.0
    }

    summary = []
    for (schedule_mode, overlap, overlap_mode), rows in sorted(grouped.items()):
        scores = np.array([row["score"] for row in rows], dtype=np.float64)
        deltas = np.array(
            [
                row["score"] - baseline_by_mode_rep[(schedule_mode, row["rep"])]
                for row in rows
                if (schedule_mode, row["rep"]) in baseline_by_mode_rep
            ],
            dtype=np.float64,
        )
        summary.append(
            {
                "schedule_mode": schedule_mode,
                "overlap_fraction": overlap,
                "overlap_mode": overlap_mode,
                "n": int(scores.size),
                "mean_accuracy": float(np.mean(scores)),
                "std_accuracy": float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0,
                "mean_delta_vs_strict_same_schedule": float(np.mean(deltas)) if deltas.size else 0.0,
                "std_delta_vs_strict_same_schedule": float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0,
                "improved_reps": int(np.sum(deltas > 0.0)) if deltas.size else 0,
            }
        )
    return summary


def main():
    args = parse_args()
    if 0.0 not in args.overlaps:
        args.overlaps = [0.0] + args.overlaps
    args.overlaps = sorted(set(args.overlaps))

    results_dir = make_results_dir(args.results_dir)
    reusable = read_reusable_rows(args.reuse_results_dir)
    raw_rows = []
    event_density_boundaries = (
        compute_event_density_boundaries(args)
        if "event_density" in args.schedule_modes
        else None
    )

    metadata = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Focused follow-up: saccade-aligned partitions and more strict-vs-best-overlap paired seeds.",
        "reps": args.reps,
        "seed_base": args.seed_base,
        "num_res": args.num_res,
        "num_partitions": args.num_partitions,
        "in_conn": args.in_conn,
        "nz": args.nz,
        "overlaps": args.overlaps,
        "schedule_modes": args.schedule_modes,
        "overlap_modes": args.overlap_modes,
        "overlap_combine": args.overlap_combine,
        "frame_window_us": args.frame_window_us,
        "batch_size": args.batch_size,
        "requested_device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "reuse_results_dir": args.reuse_results_dir,
        "event_density_boundaries": event_density_boundaries,
        "event_density_boundary_batches": args.event_density_boundary_batches,
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    schedule_rows = []
    for mode in args.schedule_modes:
        for overlap in args.overlaps:
            overlap_modes = ["none"] if overlap == 0.0 else args.overlap_modes
            for overlap_mode in overlap_modes:
                schedule_rows.append(
                    {
                        "schedule_mode": mode,
                        "overlap_fraction": overlap,
                        "overlap_mode": overlap_mode,
                        "example_steps": 300,
                        "windows": describe_temporal_partition_schedule(
                            300,
                            args.num_partitions,
                            overlap_fraction=overlap,
                            schedule_mode=mode,
                            frame_window_us=args.frame_window_us,
                            overlap_mode="symmetric" if overlap_mode == "none" else overlap_mode,
                            custom_boundaries=event_density_boundaries if mode == "event_density" else None,
                            random_seed=args.seed_base,
                        ),
                    }
                )
    write_csv(results_dir / "schedules.csv", schedule_rows, ["schedule_mode", "overlap_fraction", "overlap_mode", "example_steps", "windows"])

    fieldnames = [
        "rep",
        "seed",
        "in_conn",
        "num_res",
        "num_partitions",
        "nz",
        "overlap_fraction",
        "overlap_combine",
        "schedule_mode",
        "overlap_mode",
        "score",
        "source",
    ]

    for rep in range(args.reps):
        seed = args.seed_base + rep
        for schedule_mode in args.schedule_modes:
            for overlap in args.overlaps:
                overlap_modes = ["none"] if overlap == 0.0 else args.overlap_modes
                for overlap_mode in overlap_modes:
                    effective_overlap_mode = "symmetric" if overlap_mode == "none" else overlap_mode
                    key = (rep, seed, overlap, schedule_mode, overlap_mode)
                    fallback_key = (rep, seed, overlap, schedule_mode, effective_overlap_mode)
                    reusable_key = key if key in reusable else fallback_key
                    if reusable_key in reusable:
                        row = dict(reusable[reusable_key])
                        row["overlap_mode"] = overlap_mode
                        print("reusing", key, "score", row["score"])
                    else:
                        print("running", key)
                        set_seed(seed)
                        score = simple_ensemble_lsm(
                            args.in_conn,
                            num_res=args.num_res,
                            num_partitions=args.num_partitions,
                            Nz=args.nz,
                            overlap_fraction=overlap,
                            overlap_combine=args.overlap_combine,
                            schedule_mode=schedule_mode,
                            frame_window_us=args.frame_window_us,
                            batch_size=args.batch_size,
                            device_name=args.device,
                            overlap_mode=effective_overlap_mode,
                            schedule_boundaries=event_density_boundaries if schedule_mode == "event_density" else None,
                            schedule_seed=seed,
                        )
                        row = {
                            "rep": rep,
                            "seed": seed,
                            "in_conn": args.in_conn,
                            "num_res": args.num_res,
                            "num_partitions": args.num_partitions,
                            "nz": args.nz,
                            "overlap_fraction": overlap,
                            "overlap_combine": args.overlap_combine,
                            "schedule_mode": schedule_mode,
                            "overlap_mode": overlap_mode,
                            "score": float(score),
                            "source": "new_run",
                        }
                    raw_rows.append(row)
                    write_csv(results_dir / "raw_runs.csv", raw_rows, fieldnames)

    summary = summarize(raw_rows)
    write_csv(
        results_dir / "summary.csv",
        summary,
        [
            "schedule_mode",
            "overlap_fraction",
            "overlap_mode",
            "n",
            "mean_accuracy",
            "std_accuracy",
            "mean_delta_vs_strict_same_schedule",
            "std_delta_vs_strict_same_schedule",
            "improved_reps",
        ],
    )
    metadata["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("saved follow-up results to", results_dir)


if __name__ == "__main__":
    main()
