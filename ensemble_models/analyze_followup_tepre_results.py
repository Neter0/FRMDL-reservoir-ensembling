import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCHEDULE_LABELS = {
    "uniform": "Uniform",
    "saccade": "Saccade",
    "event_density": "Event density",
    "random_boundary": "Random boundary",
}

OVERLAP_LABELS = {
    "none": "Strict",
    "symmetric": "Adjacent overlap",
    "random": "Random overlap",
}

OVERLAP_COLORS = {
    "none": "#4d4d4d",
    "symmetric": "#2f7dbd",
    "random": "#d95f02",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create PNG plots and paired-delta tables for a completed "
            "run_followup_tepre.py result folder."
        )
    )
    parser.add_argument(
        "results_dir",
        help="Path to a completed results/followup_tepre_<timestamp> directory.",
    )
    return parser.parse_args()


def read_csv(path, converters=None):
    converters = converters or {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, converter in converters.items():
            row[key] = converter(row[key])
    return rows


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return sum(values) / len(values)


def sample_std(values):
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def sort_schedules(schedule_modes):
    preferred = ["uniform", "saccade", "event_density", "random_boundary"]
    return [item for item in preferred if item in schedule_modes] + sorted(
        set(schedule_modes) - set(preferred)
    )


def sort_overlap_modes(overlap_modes):
    preferred = ["none", "symmetric", "random"]
    return [item for item in preferred if item in overlap_modes] + sorted(
        set(overlap_modes) - set(preferred)
    )


def paired_delta_rows(raw_rows):
    baseline_by_schedule_rep = {}
    for row in raw_rows:
        if row["overlap_fraction"] == 0.0:
            baseline_by_schedule_rep[(row["schedule_mode"], row["rep"])] = row

    rows = []
    for row in raw_rows:
        if row["overlap_fraction"] == 0.0:
            continue
        key = (row["schedule_mode"], row["rep"])
        if key not in baseline_by_schedule_rep:
            raise ValueError(
                f"Missing strict baseline for schedule={row['schedule_mode']} rep={row['rep']}"
            )
        baseline = baseline_by_schedule_rep[key]
        delta = row["score"] - baseline["score"]
        rows.append(
            {
                "rep": row["rep"],
                "seed": row["seed"],
                "schedule_mode": row["schedule_mode"],
                "overlap_fraction": row["overlap_fraction"],
                "overlap_mode": row["overlap_mode"],
                "strict_score": baseline["score"],
                "overlap_score": row["score"],
                "delta": delta,
                "delta_percent": 100.0 * delta,
                "improved": row["score"] > baseline["score"],
                "source": row.get("source", ""),
            }
        )
    return rows


def compact_table_rows(raw_rows, deltas):
    grouped_scores = defaultdict(list)
    for row in raw_rows:
        grouped_scores[
            (row["schedule_mode"], row["overlap_fraction"], row["overlap_mode"])
        ].append(row["score"])

    grouped_deltas = defaultdict(list)
    for row in deltas:
        grouped_deltas[
            (row["schedule_mode"], row["overlap_fraction"], row["overlap_mode"])
        ].append(row)

    rows = []
    for key in sorted(grouped_scores, key=lambda item: (item[0], item[1], item[2])):
        schedule_mode, overlap_fraction, overlap_mode = key
        scores = grouped_scores[key]
        delta_group = grouped_deltas.get(key, [])
        delta_values = [row["delta"] for row in delta_group]
        rows.append(
            {
                "schedule_mode": schedule_mode,
                "overlap_fraction": overlap_fraction,
                "overlap_mode": overlap_mode,
                "n": len(scores),
                "mean_accuracy": mean(scores),
                "mean_accuracy_percent": 100.0 * mean(scores),
                "std_accuracy": sample_std(scores),
                "std_accuracy_percent": 100.0 * sample_std(scores),
                "mean_delta_vs_strict": mean(delta_values) if delta_values else 0.0,
                "mean_delta_vs_strict_percent": (
                    100.0 * mean(delta_values) if delta_values else 0.0
                ),
                "std_delta_vs_strict": sample_std(delta_values) if delta_values else 0.0,
                "std_delta_vs_strict_percent": (
                    100.0 * sample_std(delta_values) if delta_values else 0.0
                ),
                "improved_reps": (
                    sum(row["improved"] for row in delta_group) if delta_group else 0
                ),
            }
        )
    return rows


def row_lookup(rows):
    return {
        (row["schedule_mode"], row["overlap_fraction"], row["overlap_mode"]): row
        for row in rows
    }


def schedule_label(schedule_mode):
    return SCHEDULE_LABELS.get(schedule_mode, schedule_mode.replace("_", " ").title())


def overlap_label(overlap_mode):
    return OVERLAP_LABELS.get(overlap_mode, overlap_mode.replace("_", " ").title())


def set_common_style():
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def plot_delta_by_schedule(path, compact_rows):
    lookup = row_lookup(compact_rows)
    schedules = sort_schedules({row["schedule_mode"] for row in compact_rows})
    overlap_modes = [
        mode
        for mode in ["symmetric", "random"]
        if any(row["overlap_mode"] == mode for row in compact_rows)
    ]

    x = np.arange(len(schedules))
    width = 0.34 if len(overlap_modes) > 1 else 0.45

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for idx, overlap_mode in enumerate(overlap_modes):
        offset = (idx - (len(overlap_modes) - 1) / 2) * width
        means = []
        errors = []
        labels = []
        for schedule in schedules:
            row = lookup[(schedule, 0.3, overlap_mode)]
            means.append(row["mean_delta_vs_strict_percent"])
            errors.append(row["std_delta_vs_strict_percent"])
            labels.append(f'{row["improved_reps"]}/{row["n"]}')
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=errors,
            capsize=4,
            label=overlap_label(overlap_mode),
            color=OVERLAP_COLORS.get(overlap_mode),
            alpha=0.92,
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, label, value in zip(bars, labels, means):
            va = "bottom" if value >= 0 else "top"
            y = value + (0.015 if value >= 0 else -0.015)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                label,
                ha="center",
                va=va,
                fontsize=8,
                color="#333333",
            )

    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([schedule_label(item) for item in schedules], rotation=12)
    ax.set_ylabel("Accuracy change vs strict (%)")
    ax.set_title("Follow-up TEPRE Ablation: Overlap Effect by Partition Strategy")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.99,
        0.02,
        "labels show improved seeds / total seeds",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_accuracy_by_schedule(path, compact_rows):
    lookup = row_lookup(compact_rows)
    schedules = sort_schedules({row["schedule_mode"] for row in compact_rows})
    overlap_modes = sort_overlap_modes({row["overlap_mode"] for row in compact_rows})

    x = np.arange(len(schedules))
    width = 0.24 if len(overlap_modes) >= 3 else 0.34

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for idx, overlap_mode in enumerate(overlap_modes):
        offset = (idx - (len(overlap_modes) - 1) / 2) * width
        means = []
        errors = []
        for schedule in schedules:
            frac = 0.0 if overlap_mode == "none" else 0.3
            row = lookup[(schedule, frac, overlap_mode)]
            means.append(row["mean_accuracy_percent"])
            errors.append(row["std_accuracy_percent"])
        ax.bar(
            x + offset,
            means,
            width,
            yerr=errors,
            capsize=3,
            label=overlap_label(overlap_mode),
            color=OVERLAP_COLORS.get(overlap_mode),
            alpha=0.9,
            edgecolor="white",
            linewidth=0.7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([schedule_label(item) for item in schedules], rotation=12)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Mean N-MNIST Accuracy Across Follow-up Conditions")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=0.25)
    y_values = [row["mean_accuracy_percent"] for row in compact_rows]
    y_errors = [row["std_accuracy_percent"] for row in compact_rows]
    ax.set_ylim(min(y_values) - max(y_errors) - 0.15, max(y_values) + max(y_errors) + 0.12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_paired_seed_deltas(path, delta_rows):
    schedules = sort_schedules({row["schedule_mode"] for row in delta_rows})
    overlap_modes = [
        mode
        for mode in ["symmetric", "random"]
        if any(row["overlap_mode"] == mode for row in delta_rows)
    ]

    fig, axes = plt.subplots(
        1,
        len(overlap_modes),
        figsize=(8.6, 4.0),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    rng = np.random.default_rng(7)

    for ax, overlap_mode in zip(axes, overlap_modes):
        for schedule_idx, schedule in enumerate(schedules):
            values = [
                row["delta_percent"]
                for row in delta_rows
                if row["schedule_mode"] == schedule and row["overlap_mode"] == overlap_mode
            ]
            jitter = rng.uniform(-0.07, 0.07, size=len(values))
            ax.scatter(
                np.full(len(values), schedule_idx) + jitter,
                values,
                s=42,
                color=OVERLAP_COLORS.get(overlap_mode),
                alpha=0.78,
                edgecolors="white",
                linewidth=0.5,
            )
            ax.plot(
                [schedule_idx - 0.18, schedule_idx + 0.18],
                [mean(values), mean(values)],
                color="#222222",
                linewidth=1.4,
            )
        ax.axhline(0.0, color="#222222", linewidth=1.0)
        ax.set_xticks(range(len(schedules)))
        ax.set_xticklabels([schedule_label(item) for item in schedules], rotation=25)
        ax.set_title(overlap_label(overlap_mode))
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Per-seed accuracy change vs strict (%)")
    fig.suptitle("Paired Seed Deltas for Follow-up TEPRE Ablation", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def key_numbers(compact_rows):
    candidates = [
        row for row in compact_rows if row["overlap_fraction"] > 0.0 and row["overlap_mode"] != "none"
    ]
    largest_delta = max(candidates, key=lambda row: row["mean_delta_vs_strict"])
    best_accuracy = max(candidates, key=lambda row: row["mean_accuracy"])
    worst = min(candidates, key=lambda row: row["mean_delta_vs_strict"])
    strict_best_schedule = max(
        [row for row in compact_rows if row["overlap_mode"] == "none"],
        key=lambda row: row["mean_accuracy"],
    )
    uniform_adjacent = next(
        row
        for row in compact_rows
        if row["schedule_mode"] == "uniform" and row["overlap_mode"] == "symmetric"
    )
    return {
        "best_accuracy_overlap_condition": {
            "schedule_mode": best_accuracy["schedule_mode"],
            "overlap_mode": best_accuracy["overlap_mode"],
            "overlap_fraction": best_accuracy["overlap_fraction"],
            "mean_accuracy_percent": best_accuracy["mean_accuracy_percent"],
            "mean_delta_vs_strict_percent": best_accuracy["mean_delta_vs_strict_percent"],
            "improved_reps": best_accuracy["improved_reps"],
            "n": best_accuracy["n"],
        },
        "largest_delta_overlap_condition": {
            "schedule_mode": largest_delta["schedule_mode"],
            "overlap_mode": largest_delta["overlap_mode"],
            "overlap_fraction": largest_delta["overlap_fraction"],
            "mean_accuracy_percent": largest_delta["mean_accuracy_percent"],
            "mean_delta_vs_strict_percent": largest_delta["mean_delta_vs_strict_percent"],
            "improved_reps": largest_delta["improved_reps"],
            "n": largest_delta["n"],
        },
        "main_extension_condition": {
            "schedule_mode": uniform_adjacent["schedule_mode"],
            "overlap_mode": uniform_adjacent["overlap_mode"],
            "overlap_fraction": uniform_adjacent["overlap_fraction"],
            "mean_accuracy_percent": uniform_adjacent["mean_accuracy_percent"],
            "mean_delta_vs_strict_percent": uniform_adjacent[
                "mean_delta_vs_strict_percent"
            ],
            "improved_reps": uniform_adjacent["improved_reps"],
            "n": uniform_adjacent["n"],
        },
        "worst_overlap_condition": {
            "schedule_mode": worst["schedule_mode"],
            "overlap_mode": worst["overlap_mode"],
            "overlap_fraction": worst["overlap_fraction"],
            "mean_accuracy_percent": worst["mean_accuracy_percent"],
            "mean_delta_vs_strict_percent": worst["mean_delta_vs_strict_percent"],
            "improved_reps": worst["improved_reps"],
            "n": worst["n"],
        },
        "best_strict_baseline": {
            "schedule_mode": strict_best_schedule["schedule_mode"],
            "mean_accuracy_percent": strict_best_schedule["mean_accuracy_percent"],
            "n": strict_best_schedule["n"],
        },
        "interpretation": (
            "Adjacent boundary overlap gives a small positive mean effect for uniform, "
            "event-density, and random-boundary splits, but not for saccade-aligned "
            "splits. Random overlap is consistently worse than the matching strict "
            "baseline, supporting the interpretation that the useful signal comes "
            "from structured neighboring-boundary context rather than arbitrary "
            "extra duplicated frames."
        ),
    }


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    raw_path = results_dir / "raw_runs.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Could not find {raw_path}")

    raw_rows = read_csv(
        raw_path,
        {
            "rep": int,
            "seed": int,
            "in_conn": float,
            "num_res": int,
            "num_partitions": int,
            "nz": int,
            "overlap_fraction": float,
            "score": float,
        },
    )

    deltas = paired_delta_rows(raw_rows)
    compact = compact_table_rows(raw_rows, deltas)

    write_csv(
        results_dir / "followup_paired_deltas.csv",
        deltas,
        [
            "rep",
            "seed",
            "schedule_mode",
            "overlap_fraction",
            "overlap_mode",
            "strict_score",
            "overlap_score",
            "delta",
            "delta_percent",
            "improved",
            "source",
        ],
    )
    write_csv(
        results_dir / "followup_compact_table.csv",
        compact,
        [
            "schedule_mode",
            "overlap_fraction",
            "overlap_mode",
            "n",
            "mean_accuracy",
            "mean_accuracy_percent",
            "std_accuracy",
            "std_accuracy_percent",
            "mean_delta_vs_strict",
            "mean_delta_vs_strict_percent",
            "std_delta_vs_strict",
            "std_delta_vs_strict_percent",
            "improved_reps",
        ],
    )

    with (results_dir / "followup_key_numbers.json").open("w") as handle:
        json.dump(key_numbers(compact), handle, indent=2)

    set_common_style()
    plot_delta_by_schedule(results_dir / "followup_delta_by_schedule.png", compact)
    plot_accuracy_by_schedule(results_dir / "followup_accuracy_by_schedule.png", compact)
    plot_paired_seed_deltas(results_dir / "followup_paired_seed_deltas.png", deltas)

    print(f"Wrote follow-up analysis artifacts to {results_dir}")
    print("  followup_paired_deltas.csv")
    print("  followup_compact_table.csv")
    print("  followup_key_numbers.json")
    print("  followup_delta_by_schedule.png")
    print("  followup_accuracy_by_schedule.png")
    print("  followup_paired_seed_deltas.png")


if __name__ == "__main__":
    main()
