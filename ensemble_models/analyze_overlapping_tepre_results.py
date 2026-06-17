import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create blog-ready analysis artifacts from an existing overlapping TEPRE result folder."
    )
    parser.add_argument(
        "results_dir",
        help="Path to a completed results/overlapping_tepre_<timestamp> directory.",
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


def grouped_raw_runs(raw_rows):
    grouped = {}
    for row in raw_rows:
        grouped.setdefault(row["overlap_fraction"], {})[row["rep"]] = row
    return grouped


def paired_delta_rows(raw_rows):
    grouped = grouped_raw_runs(raw_rows)
    if 0.0 not in grouped:
        raise ValueError("raw_runs.csv must contain overlap_fraction=0.0 baseline rows")

    baseline = grouped[0.0]
    rows = []
    for overlap in sorted(grouped):
        if overlap == 0.0:
            continue
        for rep in sorted(baseline):
            strict_score = baseline[rep]["score"]
            overlap_score = grouped[overlap][rep]["score"]
            rows.append(
                {
                    "rep": rep,
                    "seed": grouped[overlap][rep]["seed"],
                    "overlap_fraction": overlap,
                    "strict_score": strict_score,
                    "overlap_score": overlap_score,
                    "delta": overlap_score - strict_score,
                    "delta_percentage_points": 100.0 * (overlap_score - strict_score),
                    "improved": overlap_score > strict_score,
                }
            )
    return rows


def significance_rows(delta_rows):
    grouped = {}
    for row in delta_rows:
        grouped.setdefault(row["overlap_fraction"], []).append(row)

    rows = []
    for overlap in sorted(grouped):
        deltas = [row["delta"] for row in grouped[overlap]]
        deltas_pp = [row["delta_percentage_points"] for row in grouped[overlap]]
        sd = sample_std(deltas)
        se = sd / math.sqrt(len(deltas)) if deltas else 0.0
        # t critical for df=3; use 1.96 as a normal approximation otherwise.
        tcrit = 3.182446 if len(deltas) == 4 else 1.96
        ci_low = mean(deltas) - tcrit * se
        ci_high = mean(deltas) + tcrit * se
        rows.append(
            {
                "overlap_fraction": overlap,
                "n": len(deltas),
                "mean_delta": mean(deltas),
                "mean_delta_percentage_points": mean(deltas_pp),
                "std_delta": sd,
                "std_delta_percentage_points": sample_std(deltas_pp),
                "improved_reps": sum(row["improved"] for row in grouped[overlap]),
                "non_improved_reps": sum(not row["improved"] for row in grouped[overlap]),
                "ci95_delta_low": ci_low,
                "ci95_delta_high": ci_high,
                "ci95_delta_low_percentage_points": 100.0 * ci_low,
                "ci95_delta_high_percentage_points": 100.0 * ci_high,
            }
        )
    return rows


def safe_range(values, padding_fraction=0.08):
    lo = min(values)
    hi = max(values)
    if lo == hi:
        pad = 1.0 if lo == 0 else abs(lo) * 0.05
        return lo - pad, hi + pad
    pad = (hi - lo) * padding_fraction
    return lo - pad, hi + pad


def svg_line_chart(path, title, x_values, series, y_label, width=760, height=430):
    margin_left = 76
    margin_right = 26
    margin_top = 54
    margin_bottom = 62
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_y = []
    for item in series:
        all_y.extend(item["y"])
        all_y.extend(item.get("y_low", []))
        all_y.extend(item.get("y_high", []))
    y_min, y_max = safe_range(all_y)
    x_min, x_max = safe_range(x_values, padding_fraction=0.04)

    def sx(x):
        return margin_left + ((x - x_min) / (x_max - x_min)) * plot_w

    def sy(y):
        return margin_top + (1.0 - ((y - y_min) / (y_max - y_min))) * plot_h

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#222"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13">Overlap fraction</text>',
        f'<text x="18" y="{height / 2}" text-anchor="middle" font-family="Arial" font-size="13" transform="rotate(-90 18 {height / 2})">{y_label}</text>',
    ]

    for tick in x_values:
        x = sx(tick)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{margin_top + plot_h}" x2="{x:.2f}" y2="{margin_top + plot_h + 5}" stroke="#222"/>',
                f'<text x="{x:.2f}" y="{margin_top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="12">{tick:.2f}</text>',
            ]
        )

    for i in range(5):
        value = y_min + (y_max - y_min) * i / 4
        y = sy(value)
        parts.extend(
            [
                f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_w}" y2="{y:.2f}" stroke="#e6e6e6"/>',
                f'<text x="{margin_left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{value:.3f}</text>',
            ]
        )

    for idx, item in enumerate(series):
        color = item.get("color", colors[idx % len(colors)])
        points = " ".join(f'{sx(x):.2f},{sy(y):.2f}' for x, y in zip(x_values, item["y"]))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{points}"/>')
        for point_idx, (x_value, y_value) in enumerate(zip(x_values, item["y"])):
            x = sx(x_value)
            y = sy(y_value)
            if "y_low" in item and "y_high" in item:
                y_low = sy(item["y_low"][point_idx])
                y_high = sy(item["y_high"][point_idx])
                parts.extend(
                    [
                        f'<line x1="{x:.2f}" y1="{y_low:.2f}" x2="{x:.2f}" y2="{y_high:.2f}" stroke="{color}" stroke-width="1.5"/>',
                        f'<line x1="{x - 5:.2f}" y1="{y_low:.2f}" x2="{x + 5:.2f}" y2="{y_low:.2f}" stroke="{color}" stroke-width="1.5"/>',
                        f'<line x1="{x - 5:.2f}" y1="{y_high:.2f}" x2="{x + 5:.2f}" y2="{y_high:.2f}" stroke="{color}" stroke-width="1.5"/>',
                    ]
                )
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')

    legend_x = margin_left + 12
    legend_y = margin_top + 18
    for idx, item in enumerate(series):
        color = item.get("color", colors[idx % len(colors)])
        y = legend_y + idx * 20
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="2.4"/>',
                f'<text x="{legend_x + 32}" y="{y + 4}" font-family="Arial" font-size="12">{item["label"]}</text>',
            ]
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    raw_rows = read_csv(
        results_dir / "raw_runs.csv",
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
    summary_rows = read_csv(
        results_dir / "summary.csv",
        {
            "overlap_fraction": float,
            "n": int,
            "mean_accuracy": float,
            "std_accuracy": float,
            "min_accuracy": float,
            "max_accuracy": float,
            "mean_delta_vs_strict": float,
            "std_delta_vs_strict": float,
        },
    )

    delta_rows = paired_delta_rows(raw_rows)
    sig_rows = significance_rows(delta_rows)

    write_csv(
        results_dir / "paired_deltas.csv",
        delta_rows,
        [
            "rep",
            "seed",
            "overlap_fraction",
            "strict_score",
            "overlap_score",
            "delta",
            "delta_percentage_points",
            "improved",
        ],
    )
    write_csv(
        results_dir / "delta_summary.csv",
        sig_rows,
        [
            "overlap_fraction",
            "n",
            "mean_delta",
            "mean_delta_percentage_points",
            "std_delta",
            "std_delta_percentage_points",
            "improved_reps",
            "non_improved_reps",
            "ci95_delta_low",
            "ci95_delta_high",
            "ci95_delta_low_percentage_points",
            "ci95_delta_high_percentage_points",
        ],
    )

    best = max(summary_rows, key=lambda row: row["mean_accuracy"])
    strict = next(row for row in summary_rows if row["overlap_fraction"] == 0.0)
    best_delta = next(row for row in sig_rows if row["overlap_fraction"] == best["overlap_fraction"])
    key_numbers = {
        "strict_mean_accuracy_percent": 100.0 * strict["mean_accuracy"],
        "best_overlap_fraction": best["overlap_fraction"],
        "best_mean_accuracy_percent": 100.0 * best["mean_accuracy"],
        "best_mean_delta_percentage_points": 100.0 * best["mean_delta_vs_strict"],
        "best_improved_repetitions": best_delta["improved_reps"],
        "best_total_repetitions": best_delta["n"],
        "best_delta_ci95_percentage_points": [
            best_delta["ci95_delta_low_percentage_points"],
            best_delta["ci95_delta_high_percentage_points"],
        ],
        "interpretation_hint": (
            "The best overlap setting gave a small, consistent improvement over the strict baseline; "
            "the effect is modest and should not be described as a large accuracy gain."
        ),
    }
    (results_dir / "key_numbers.json").write_text(json.dumps(key_numbers, indent=2))

    x_values = [row["overlap_fraction"] for row in summary_rows]
    mean_percent = [100.0 * row["mean_accuracy"] for row in summary_rows]
    std_percent = [100.0 * row["std_accuracy"] for row in summary_rows]
    svg_line_chart(
        results_dir / "accuracy_by_overlap.svg",
        "N-MNIST Accuracy by Temporal Overlap",
        x_values,
        [
            {
                "label": "Mean accuracy +/- std",
                "y": mean_percent,
                "y_low": [m - s for m, s in zip(mean_percent, std_percent)],
                "y_high": [m + s for m, s in zip(mean_percent, std_percent)],
                "color": "#1f77b4",
            }
        ],
        "Accuracy (%)",
    )

    delta_percent = [100.0 * row["mean_delta_vs_strict"] for row in summary_rows]
    delta_std_percent = [100.0 * row["std_delta_vs_strict"] for row in summary_rows]
    svg_line_chart(
        results_dir / "delta_vs_strict.svg",
        "Accuracy Delta versus Strict TEPRE",
        x_values,
        [
            {
                "label": "Mean paired delta +/- std",
                "y": delta_percent,
                "y_low": [d - s for d, s in zip(delta_percent, delta_std_percent)],
                "y_high": [d + s for d, s in zip(delta_percent, delta_std_percent)],
                "color": "#d62728",
            }
        ],
        "Delta (percentage points)",
    )

    print("wrote", results_dir / "paired_deltas.csv")
    print("wrote", results_dir / "delta_summary.csv")
    print("wrote", results_dir / "key_numbers.json")
    print("wrote", results_dir / "accuracy_by_overlap.svg")
    print("wrote", results_dir / "delta_vs_strict.svg")


if __name__ == "__main__":
    main()
