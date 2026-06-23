import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader

from partition_schedules import build_temporal_partition_schedule, describe_temporal_partition_schedule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure how many N-MNIST events/frames are assigned to each temporal reservoir partition."
    )
    parser.add_argument("--num-partitions", type=int, default=3)
    parser.add_argument("--overlaps", type=float, nargs="+", default=[0.0, 0.30])
    parser.add_argument("--schedule-modes", nargs="+", default=["uniform", "saccade"])
    parser.add_argument("--frame-window-us", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-train-batches", type=int, default=20)
    parser.add_argument("--max-test-batches", type=int, default=10)
    parser.add_argument("--results-dir", default=None)
    return parser.parse_args()


def make_results_dir(results_dir):
    if results_dir is None:
        results_dir = Path("results") / "partition_event_counts"
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def iter_dataset(train, frame_window_us, batch_size):
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.Compose(
        [
            transforms.Denoise(filter_time=3000),
            transforms.ToFrame(sensor_size=sensor_size, time_window=frame_window_us),
        ]
    )
    split = "train" if train else "test"
    dataset = tonic.datasets.NMNIST(save_to="../data", transform=frame_transform, train=train)
    cached = DiskCachedDataset(dataset, cache_path=f"../cache/nmnist/{split}")
    return DataLoader(
        cached,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        shuffle=False,
    )


def summarize_loader(loader, split, max_batches, args):
    rows = []
    for batch_idx, (data, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        num_steps = data.shape[0]
        batch_size = data.shape[1]
        activity = data.reshape(num_steps, batch_size, -1).sum(dim=2).numpy()
        for mode in args.schedule_modes:
            for overlap in args.overlaps:
                schedule = build_temporal_partition_schedule(
                    num_steps,
                    args.num_partitions,
                    overlap_fraction=overlap,
                    schedule_mode=mode,
                    frame_window_us=args.frame_window_us,
                )
                partition_activity = np.zeros((args.num_partitions, batch_size), dtype=np.float64)
                duplicated_activity = np.zeros(batch_size, dtype=np.float64)
                total_activity = activity.sum(axis=0)

                for step, active_parts in enumerate(schedule):
                    for part in active_parts:
                        partition_activity[part] += activity[step]
                    if len(active_parts) > 1:
                        duplicated_activity += activity[step] * (len(active_parts) - 1)

                for sample_idx in range(batch_size):
                    row = {
                        "split": split,
                        "batch_idx": batch_idx,
                        "sample_idx": sample_idx,
                        "label": int(targets[sample_idx]),
                        "num_steps": int(num_steps),
                        "schedule_mode": mode,
                        "overlap_fraction": overlap,
                        "total_frame_activity": float(total_activity[sample_idx]),
                        "duplicated_boundary_activity": float(duplicated_activity[sample_idx]),
                        "duplicated_boundary_activity_fraction": (
                            float(duplicated_activity[sample_idx] / total_activity[sample_idx])
                            if total_activity[sample_idx] > 0
                            else 0.0
                        ),
                    }
                    for part in range(args.num_partitions):
                        row[f"partition_{part + 1}_activity"] = float(partition_activity[part, sample_idx])
                        row[f"partition_{part + 1}_activity_fraction"] = (
                            float(partition_activity[part, sample_idx] / total_activity[sample_idx])
                            if total_activity[sample_idx] > 0
                            else 0.0
                        )
                    rows.append(row)
    return rows


def aggregate(rows, num_partitions):
    grouped = {}
    for row in rows:
        key = (row["split"], row["schedule_mode"], row["overlap_fraction"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (split, mode, overlap), group in sorted(grouped.items()):
        out = {
            "split": split,
            "schedule_mode": mode,
            "overlap_fraction": overlap,
            "n_samples": len(group),
            "mean_total_frame_activity": float(np.mean([r["total_frame_activity"] for r in group])),
            "mean_duplicated_boundary_activity_fraction": float(
                np.mean([r["duplicated_boundary_activity_fraction"] for r in group])
            ),
        }
        for part in range(num_partitions):
            out[f"mean_partition_{part + 1}_activity_fraction"] = float(
                np.mean([r[f"partition_{part + 1}_activity_fraction"] for r in group])
            )
        summary.append(out)
    return summary


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    results_dir = make_results_dir(args.results_dir)
    train_rows = summarize_loader(
        iter_dataset(True, args.frame_window_us, args.batch_size),
        "train",
        args.max_train_batches,
        args,
    )
    test_rows = summarize_loader(
        iter_dataset(False, args.frame_window_us, args.batch_size),
        "test",
        args.max_test_batches,
        args,
    )
    rows = train_rows + test_rows
    summary = aggregate(rows, args.num_partitions)

    write_csv(results_dir / "partition_event_counts_raw.csv", rows)
    write_csv(results_dir / "partition_event_counts_summary.csv", summary)
    (results_dir / "metadata.json").write_text(json.dumps(vars(args), indent=2))
    print("saved event-count analysis to", results_dir)


if __name__ == "__main__":
    main()
