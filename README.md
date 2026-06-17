# snntorch-LSM Reproduction Extension

This repository is based on the authors' `snntorch-LSM` implementation for Liquid State Machines. For our reproduction project, we use the existing code path for N-MNIST TEPRE-style temporal partitioning and add one new algorithm variant: overlapping temporal partitions.

## Project Focus

The paper reports N-MNIST results for Temporal and Spatial Reservoir Ensembling Techniques for Liquid State Machines. Our extension asks:

> Does strict temporal separation in TEPRE lose useful information when temporal patterns cross partition boundaries?

The original strict temporal partitioning assigns each time step to one partition. The extension in `ensemble_models/partition_schedules.py` allows adjacent temporal partitions to overlap, so boundary frames are processed by both neighboring partitions. The default overlap combination rule is `mean`, which keeps the input current scale comparable to the strict baseline.

Important wording for the report: `overlap_fraction=0.0` is the strict non-overlapping TEPRE baseline in the same controlled script. It is not the untouched original authors' script.

## Repository Structure

- `lsm_models.py`: base LSM model definitions.
- `lsm_weight_definitions.py`: input-to-reservoir and recurrent reservoir weight definitions.
- `main.py`: original example N-MNIST implementation.
- `ensemble_models/lsm_models.py`: ensemble model definitions, including overlap-aware partitioned models.
- `ensemble_models/partition_schedules.py`: strict and overlapping temporal partition schedules.
- `ensemble_models/test_overlapping_tepre.py`: one-shot baseline-vs-overlap N-MNIST experiment.
- `ensemble_models/analyze_overlapping_tepre_results.py`: post-processing script for plots and paired-delta tables.
- `ensemble_models/results/overlapping_tepre_20260616_154157/`: completed result artifacts used for our report.

## Setup

Use Python 3.11 or 3.12. Do not use Python 3.13 for this project: `tonic==1.6.0` depends on `numpy<2.0`, and NumPy 1.26.x does not provide normal Windows wheels for Python 3.13.

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If `py -3.11` is unavailable, install Python 3.11 first and make sure your IDE uses the virtual environment interpreter:

```text
<repo-root>\.venv\Scripts\python.exe
```

## Reproduce The Extension Results

The experiment downloads/caches N-MNIST through Tonic if the data are not already present. Runtime can be several hours on CPU.

From the repository root:

```powershell
cd ensemble_models
python test_overlapping_tepre.py --reps 4 --num-res 3 --num-partitions 3 --overlaps 0.0 0.10 0.15 0.20 0.30
```

This creates a timestamped folder:

```text
ensemble_models/results/overlapping_tepre_<timestamp>/
```

The core outputs are:

- `raw_runs.csv`: one row per repetition and overlap setting.
- `summary.csv`: mean/std/min/max accuracy and paired deltas versus strict TEPRE.
- `schedules.csv`: temporal windows and the number of overlap steps.
- `metadata.json`: exact experiment settings.
- `overlapping_tepre_results.npz`: NumPy archive for further analysis.

## Post-Process Existing Results

After the training run finishes, generate plots and paired-delta tables without retraining.
Run this from `ensemble_models`:

```powershell
python analyze_overlapping_tepre_results.py results/overlapping_tepre_<timestamp>
```

For the included run, use this from `ensemble_models`:

```powershell
python analyze_overlapping_tepre_results.py results/overlapping_tepre_20260616_154157
```

This adds:

- `paired_deltas.csv`: strict-vs-overlap score differences for each paired seed.
- `delta_summary.csv`: paired delta statistics, improvement counts, and approximate 95% intervals.
- `key_numbers.json`: main values to quote in the report.
- `accuracy_by_overlap.svg`: accuracy plot with error bars.
- `delta_vs_strict.svg`: delta-vs-baseline plot.
