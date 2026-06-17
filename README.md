# snntorch-LSM
This is an **snntorch** implementation of Liquid State Machine (LSM) networks. The parameters of the example network implemented in **main.py** are derived from the paper **"MAdapter: A Multimodal Adapter for Liquid State Machines configures the Input Layer for the same Reservoir to enable Vision and Speech Classification"** [link](https://ieeexplore.ieee.org/document/10191376)

## Requirements
Pytorch, Tonic, numpy, sklearn and snntorch

## Description
1. **lsm_weight_definitions.py** - contains definitions of connectivity (Input->Reservoir and Recurrent Reservoir weights)
2. **lsm_models.py** - contains the LSM model definition.
3. **main.py** - contains an example implementation with the N-MNIST dataset. Network execution must be run with **torch.no_grad()** for LSM operation
4. **main.ipynb** - same as main.py but for running in Google Colab. (Make sure to upload this repository folder to in Google Drive and modify the path in the 3rd cell accordingly to run)

## Overlapping TEPRE extension
The original temporal partitioning uses strict non-overlapping windows. The extension in
`ensemble_models/partition_schedules.py` allows adjacent temporal partitions to overlap,
so boundary frames can be processed by both neighboring partitions. This tests whether
strict temporal separation loses useful cross-boundary information.

Run the full one-shot baseline-vs-overlap sweep on N-MNIST with:

```bash
cd ensemble_models
python test_overlapping_tepre.py --reps 4 --num-res 3 --num-partitions 3 --overlaps 0.0 0.10 0.15 0.20 0.30
```

`--overlap-combine mean` is the default because it keeps the input current scale comparable
to the non-overlapping baseline. The script automatically includes `overlap_fraction=0.0`
as the strict TEPRE baseline, uses paired seeds across overlap settings, and saves all
outputs under `ensemble_models/results/overlapping_tepre_<timestamp>/`:

- `raw_runs.csv`: one row per repetition and overlap setting.
- `summary.csv`: mean/std/min/max accuracy and paired deltas versus strict TEPRE.
- `schedules.csv`: temporal windows and the number of overlap steps.
- `metadata.json`: exact experiment settings.
- `overlapping_tepre_results.npz`: NumPy archive for further analysis.
