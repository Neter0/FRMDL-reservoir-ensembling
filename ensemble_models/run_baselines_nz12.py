"""Run the 4 baseline methods at the SAME config as our MuLRE+TEPRE run
(in_conn=0.15, Nz=12), save real numbers, and render the final 5-method
N-MNIST comparison figure.

Methods (all at in_conn=0.15, Nz=12 -> 1200 neurons):
  LSM              = simple_ensemble_lsm(num_res=1)
  Vanilla Ensemble = simple_ensemble_lsm(num_res=3)
  MuLRE            = long_short_ensemble_lsm(long_dist1=4, long_dist2=6)
  TEPRE            = simple_ensemble_lsm(num_res=1, num_partitions=3)
  MuLRE+TEPRE      = reused from outputs/mulre_tepre_results.txt (already computed)

Usage (from anywhere in the repo):
    .venv/bin/python ensemble_models/run_baselines_nz12.py
"""
import os
import sys
import re
import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../ensemble_models
sys.path.insert(0, str(HERE))
os.chdir(HERE)                                    # so the functions' '../data' -> repo-root/data

from ensemble_lsm import simple_ensemble_lsm
from ensemble_3_reservoir_long_short_dist_lsm import long_short_ensemble_lsm

OUT = HERE / "reproductions" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS_TXT = OUT / "general_comparison_nz12_results.txt"

IN_CONN, NZ, N_PART, LONG_D1, LONG_D2 = 0.15, 12, 3, 4, 6


def log_line(s):
    print(s, flush=True)
    with open(RESULTS_TXT, "a") as f:
        f.write(s + "\n")


def reuse_mulre_tepre():
    """Read our previously computed MuLRE+TEPRE Nz=12 accuracy from its results file."""
    f = OUT / "mulre_tepre_results.txt"
    if f.exists():
        m = re.findall(r"Nz=12\b.*?test_accuracy=([0-9.]+)", f.read_text())
        if m:
            return float(m[-1])
    return None


if __name__ == "__main__":
    log_line(f"# {datetime.now():%Y-%m-%d %H:%M}  N-MNIST comparison  "
             f"(in_conn={IN_CONN}, Nz={NZ}, neurons={NZ*100}, partitions={N_PART})")

    results = {}
    print("\n=== [1/4] LSM ===", flush=True)
    results["LSM"] = simple_ensemble_lsm(IN_CONN, num_res=1, Nz=NZ)
    log_line(f"LSM                  = {results['LSM']:.4f}")

    print("\n=== [2/4] Vanilla Ensemble ===", flush=True)
    results["Vanilla Ensemble"] = simple_ensemble_lsm(IN_CONN, num_res=3, Nz=NZ)
    log_line(f"Vanilla Ensemble     = {results['Vanilla Ensemble']:.4f}")

    print("\n=== [3/4] MuLRE ===", flush=True)
    results["MuLRE"] = long_short_ensemble_lsm(IN_CONN, long_dist1=LONG_D1, long_dist2=LONG_D2, Nz=NZ)
    log_line(f"MuLRE                = {results['MuLRE']:.4f}")

    print("\n=== [4/4] TEPRE ===", flush=True)
    results["TEPRE"] = simple_ensemble_lsm(IN_CONN, num_res=1, num_partitions=N_PART, Nz=NZ)
    log_line(f"TEPRE                = {results['TEPRE']:.4f}")

    mt = reuse_mulre_tepre()
    if mt is not None:
        results["MuLRE+TEPRE"] = mt
        log_line(f"MuLRE+TEPRE (reused) = {mt:.4f}")
    else:
        log_line("MuLRE+TEPRE: NOT FOUND (run: python run_mulre_tepre.py 12)")

    # Machine-readable results + paper reference numbers for the report
    payload = {
        "dataset": "N-MNIST",
        "config": {"in_conn": IN_CONN, "Nz": NZ, "neurons": NZ * 100,
                   "num_partitions": N_PART, "long_dist1": LONG_D1, "long_dist2": LONG_D2},
        "results": results,
        "paper_reference_headline_3600_neurons": {"TEPRE_p3": 0.981, "MuLRE": 0.9765,
                                                  "note": "paper reports at 3600 neurons; this run is at 1200"},
    }
    (OUT / "general_comparison_nz12_results.json").write_text(json.dumps(payload, indent=2))

    # Render the figure (single size -> bar chart of the 5 methods)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["LSM", "Vanilla Ensemble", "MuLRE", "TEPRE", "MuLRE+TEPRE"]
    names = [n for n in order if n in results]
    vals = [results[n] for n in names]
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"][:len(names)]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.0004, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=10)
    plt.ylim(min(vals) - 0.008, max(vals) + 0.006)
    plt.ylabel("Test accuracy", fontsize=12)
    plt.title(f"N-MNIST ensemble comparison  ({NZ*100} neurons, in_conn={IN_CONN})",
              fontsize=13, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=12)
    plt.tight_layout()
    fig_path = OUT / "general_comparison_nz12.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    log_line(f"\nfigure  -> {fig_path}")
    log_line(f"numbers -> {OUT / 'general_comparison_nz12_results.json'}")
    print("DONE", flush=True)
