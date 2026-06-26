"""Run ONLY the MuLRE+TEPRE model on N-MNIST and print its test accuracy.

MuLRE+TEPRE = 3 distance-diverse reservoirs (short / long_dist1=4 / long_dist2=6),
each additionally time-partitioned into `num_partitions=3`. Same parameters as the
MuLRE+TEPRE line in reproductions/general_comparison.py -- just without the other
four methods and without matplotlib.

Usage (from anywhere in the repo):
    .venv/bin/python ensemble_models/run_mulre_tepre.py        # Nz=12 (fast)
    .venv/bin/python ensemble_models/run_mulre_tepre.py 24     # larger network
    .venv/bin/python ensemble_models/run_mulre_tepre.py 36     # largest (paper's max)
"""
import os
import sys
from datetime import datetime
from pathlib import Path

# Make this runnable from any working directory:
HERE = Path(__file__).resolve().parent      # .../ensemble_models
sys.path.insert(0, str(HERE))               # so the ensemble modules import
os.chdir(HERE)                              # so the function's '../data' -> repo-root/data

from ensemble_3_reservoir_long_short_dist_lsm import long_short_ensemble_lsm

if __name__ == "__main__":
    Nz = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    score = long_short_ensemble_lsm(
        0.15,            # in_conn
        long_dist1=4,
        long_dist2=6,
        num_partitions=3,
        Nz=Nz,
    )
    line = (f"{datetime.now():%Y-%m-%d %H:%M}  MuLRE+TEPRE  Nz={Nz}  "
            f"num_partitions=3 long_dist1=4 long_dist2=6 in_conn=0.15  "
            f"test_accuracy={score}")
    print("\n" + line)

    # Persist the result so it survives a cleared terminal
    out_file = HERE / "reproductions" / "outputs" / "mulre_tepre_results.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "a") as f:
        f.write(line + "\n")
    print(f"saved -> {out_file}")
