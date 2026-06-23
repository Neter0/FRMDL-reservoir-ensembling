# snntorch-LSM Reproduction Extension

This repository reproduces and extends part of the N-MNIST TEPRE experiments from the paper on Temporal and Spatial Reservoir Ensembling Techniques for Liquid State Machines.

Our extension tests whether strict non-overlapping temporal partitions lose useful boundary information. The original TEPRE-style split assigns each time step to exactly one reservoir partition. Our variant lets neighboring temporal partitions overlap, so boundary frames are processed by both adjacent partitions. We also add follow-up controls that compare adjacent overlap with random overlap and compare different temporal partition schedules.

## Data

The N-MNIST data are downloaded automatically by `tonic` on the first experiment run.

The following folders are intentionally not tracked by git:

- `data/`: downloaded N-MNIST files.
- `cache/`: transformed cached tensors for faster later runs.

The first run needs internet access and enough disk space. Later runs reuse the local `data/` and `cache/` folders.

## Setup

Use Python 3.11 or 3.12. Avoid Python 3.13 because `tonic==1.6.0` requires `numpy<2.0`, and NumPy 1.26.x is not normally available as a Windows wheel for Python 3.13.

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Optional, for NVIDIA GPU acceleration:

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.26.4 opencv-python==4.10.0.84
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

## Reproduce Results

Run all commands from `ensemble_models`:

```powershell
cd ensemble_models
```

### 1. Main Overlap Experiment

This reproduces strict TEPRE and overlap levels `0.10`, `0.15`, `0.20`, and `0.30` over 4 seeds:

```powershell
python test_overlapping_tepre.py --reps 4 --num-res 3 --num-partitions 3 --overlaps 0.0 0.10 0.15 0.20 0.30
python analyze_overlapping_tepre_results.py results/overlapping_tepre_<timestamp>
```

Included completed run:

```powershell
python analyze_overlapping_tepre_results.py results/overlapping_tepre_20260616_154157
```

Main outputs:

- `summary.csv`
- `paired_deltas.csv`
- `delta_summary.csv`
- `delta_vs_strict.png`
- `accuracy_by_overlap.svg`

### 2. Event-Count Analysis

This checks how much activity is assigned to each partition and how much is duplicated by overlap:

```powershell
python analyze_partition_event_counts.py --overlaps 0.0 0.30 --schedule-modes uniform saccade
```

Included completed run:

```text
results/partition_event_counts_20260622_rerun/
```

Main outputs:

- `partition_event_counts_raw.csv`
- `partition_event_counts_summary.csv`

### 3. Follow-Up Boundary and Random-Overlap Ablation

This compares uniform, saccade-aligned, event-density, and random-boundary partitions. It also compares adjacent overlap against random overlap as a negative control.

```powershell
python run_followup_tepre.py --reps 4 --overlaps 0.0 0.30 --schedule-modes uniform saccade event_density random_boundary --overlap-modes symmetric random --device cuda
python analyze_followup_tepre_results.py results/followup_tepre_<timestamp>
```

If CUDA is unavailable, replace `--device cuda` with `--device cpu`.

Included completed run:

```powershell
python analyze_followup_tepre_results.py results/followup_tepre_20260622_223915
```

Main outputs:

- `followup_compact_table.csv`
- `followup_paired_deltas.csv`
- `followup_key_numbers.json`
- `followup_delta_by_schedule.png`
- `followup_accuracy_by_schedule.png`
- `followup_paired_seed_deltas.png`

## Key Result Files

The most useful files for the report/poster are:

- `results/overlapping_tepre_20260616_154157/delta_vs_strict.png`
- `results/followup_tepre_20260622_223915/followup_delta_by_schedule.png`
- `results/followup_tepre_20260622_223915/followup_compact_table.csv`
- `results/partition_event_counts_20260622_rerun/partition_event_counts_summary.csv`
