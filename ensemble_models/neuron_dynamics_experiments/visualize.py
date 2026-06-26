from collections import defaultdict
import json
from pathlib import Path

import os 
import matplotlib.pyplot as plt
import numpy as np

OUTPUTS_DIR = Path(__file__).parent / "outputs"
SCORES_PATH = OUTPUTS_DIR / "IF_comparison_scores.json"
BAR_COLOR = "#4C78A8"
BAR_EDGE_COLOR = "#2F4B7C"
ERROR_COLOR = "#111827"
POINT_COLOR = "#374151"


def load_scores(scores_path):
    with scores_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def model_points(scores, model_name, resonant_periods=False):
    # Take all of the scores in the scores.json file and filter them based on a given model_name
    # resonant_periods controls whether or not we are doing a resonant period study 

    if model_name not in scores:
        raise KeyError(f"Model '{model_name}' was not found in {SCORES_PATH}")

    points = []
    for Nz_key, runs in scores[model_name].items():
        if not runs or Nz_key not in [str(nz) for nz in NZs]:
            continue

        neurons = runs[0].get("neurons", int(Nz_key) * 100)
        run_scores = [run["score"] for run in runs]

        # This is not needed if we are doing general comparison, only for the resonant period studies
        t_ranges = None

        # For resonant period study: collect the ranges of resonant periods we are considering
        if resonant_periods:
            t_ranges = []
            missing_t_metadata = []

            for run_idx, run in enumerate(runs, start=1):
                params = run.get("params", {})
                t_min = params.get("T_min")
                t_max = params.get("T_max")

                if t_min is None or t_max is None:
                    missing_t_metadata.append(run_idx)
                    continue

                t_ranges.append((t_min, t_max))

            if missing_t_metadata:
                raise ValueError(
                    f"resonant_periods=True requires T_min and T_max for every "
                    f"selected run, but model '{model_name}' Nz={Nz_key} is missing "
                    f"metadata for run(s): {missing_t_metadata}"
                )
        # Make sure that not all of the resonant period sweeps are considered, but just the best settings
        else:
            if model_name == "MuLRE RF":
                run_scores = [run["score"] for run in runs if run['params'].get('T_min') == 30 and run['params'].get('T_max') == 300]
            if model_name == "TEPRE RF":
                run_scores = [run["score"] for run in runs if run['params'].get('T_min') == 100 and run['params'].get('T_max') == 100]
        

        points.append((neurons, np.mean(run_scores), run_scores, t_ranges))

    return sorted(points, key=lambda point: point[0])


def plot_mean_bars(ax, x, means, stds, score_groups):
    # Helper method for pretty histograms
    # Based on the idea that we are keeping all experiments at 
    # Nz = 12 => 1200 neurons, thus histograms are applicable 

    ax.bar(
        x,
        means,
        width=0.56,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=1.1,
        alpha=0.86,
        zorder=2,
    )
    if False:
        ax.errorbar(
            x,
            means,
            yerr=stds,
            fmt="none",
            ecolor=ERROR_COLOR,
            elinewidth=2.4,
            capsize=8,
            capthick=2.4,
            zorder=4,
        )

    for x_pos, run_scores in zip(x, score_groups):
        jitter = np.linspace(-0.10, 0.10, len(run_scores)) if len(run_scores) > 1 else [0]
        ax.scatter(
            np.full(len(run_scores), x_pos) + jitter,
            run_scores,
            s=28,
            color=POINT_COLOR,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
        )


def plot_scores(scores, model_names, resonant_periods=False, filename="scores.png"):
    # Main plotting method. 
    # It contains two branches for histograms of resonant_period against score for resonant period study
    # and model type against score for general comparison

    fig, ax = plt.subplots(figsize=(10, 6))

    if resonant_periods is False:
        # Bar graph: model name vs mean score ± std
        means = []
        stds = []
        labels = []
        score_groups = []

        for model_name in model_names:
            points = model_points(scores, model_name)
            if not points:
                continue

            # collect all run scores for this model
            all_run_scores = []
            for _, _, run_scores, _ in points:
                all_run_scores.extend(run_scores)

            if not all_run_scores:
                continue

            labels.append(model_name)
            means.append(np.mean(all_run_scores))
            stds.append(np.std(all_run_scores))
            score_groups.append(all_run_scores)

        x = np.arange(len(labels))

        plot_mean_bars(ax, x, means, stds, score_groups)
        y_min = min(m - s for m, s in zip(means, stds))
        y_max = max(m + s for m, s in zip(means, stds))
        ax.set_ylim(max(0.0, y_min - 0.01), min(1.0, y_max + 0.01))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Accuracy Score", fontsize=12)
        ax.set_title("Model performance", fontsize=14, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        filename = "model_comparison.png"

    else:
        # Bar graph: (T_min, T_max) vs score, for a single model
        if len(model_names) != 1:
            raise ValueError("When resonant_periods=True, provide exactly one model name.")

        model_name = model_names[0]
        points = model_points(scores, model_name, resonant_periods=True)
        if not points:
            raise ValueError(f"No points found for model '{model_name}'.")

        period_scores = defaultdict(list)

        for _, _, run_scores, t_ranges in points:
            for score, t_range in zip(run_scores, t_ranges):
                period_scores[str(t_range)].append(score)

        labels = list(period_scores.keys())
        means = [np.mean(period_scores[label]) for label in labels]
        stds = [np.std(period_scores[label]) for label in labels]
        score_groups = [period_scores[label] for label in labels]

        x = np.arange(len(labels))

        plot_mean_bars(ax, x, means, stds, score_groups)
        y_min = min(m - s for m, s in zip(means, stds))
        y_max = max(m + s for m, s in zip(means, stds))
        ax.set_ylim(max(0.0, y_min - 0.01), min(1.0, y_max + 0.01))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_xlabel("(T_min, T_max)", fontsize=12)
        ax.set_ylabel("Accuracy Score", fontsize=12)
        ax.set_title(f"{model_name} resonant period comparison", fontsize=14, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        filename = f"{model_name.replace(' ', '_')}_resonant_periods.png"

        
    save_path = OUTPUTS_DIR / filename

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return save_path


# Define what models you want to compare / run resonant period study on
MODEL_NAMES = [
    #"MuLRE IF",
    #"MuLRE LIF",
    #"MuLRE SLIF",
    "MuLRE RF",
    #"TEPRE IF",
    #"TEPRE LIF",
    #"TEPRE SLIF",
    #"TEPRE RF",
]

# List of NZ values to consider for plotting, this might be extended for regular scatter plots
# against multiple neuron counts
NZs = [12]  

if __name__ == "__main__":
    scores = load_scores(SCORES_PATH)
    plot_scores(scores, MODEL_NAMES, resonant_periods=True)
