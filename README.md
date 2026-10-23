# FSED-Pilot

Freestyle stroke-event detection from a single wearable accelerometer.

Pilot experiment for [Freestyle-Stroke-Event-Detection](https://github.com/RyanWoodward132/Freestyle-Stroke-Event-Detection).

## Overview

This project trains a convolutional neural network (CNN) to classify the stroke phases of elite freestyle swimmers from a single three-axis accelerometer worn on the upper back. For each stroke the model distinguishes eight classes: left/right arm and the four phases **entry**, **catch**, **pull**, and **push**.

Headline result across five random seeds with a stratified 80/20 train/test split:

| Input features | Accuracy (mean ± sd) | Macro F1 (mean ± sd) |
| --- | --- | --- |
| Acceleration + jerk (7 channels) | 0.588 ± 0.036 | 0.588 ± 0.035 |
| Acceleration only (3 channels) | 0.602 ± 0.038 | 0.602 ± 0.038 |

This is well above the 12.5% random baseline for eight balanced classes, but the modest macro F1 (~0.59) shows the task is genuinely hard. Two findings stand out: (1) derived jerk features do not improve over raw acceleration, and (2) roughly four in five errors pick the **wrong arm at a neighbouring phase**, rather than the wrong phase on the correct arm.

## Data

Ten competitive swimmers performed a 100 m maximal-effort freestyle swim while wearing an inertial measurement unit (IMU) on the upper back. The accelerometer recorded three axes (vertical, lateral, sagittal) at 100 Hz, and stroke events were labelled manually by reviewing synchronised video recordings.

Six participants (`AC`, `IR`, `JD`, `JH`, `KV`, `RW`) were carried through the full pipeline and used for modelling, yielding **1,313 labelled event windows** (159–169 per class, roughly balanced).

Raw accelerometer data and videos are withheld for privacy and are not part of this repository. The committed code documents the complete processing and modelling pipeline.

## Processing pipeline

Per-participant R scripts in `Processing/<ID>.R` run an identical pipeline:

1. **Load** raw accelerometer and event timestamps.
2. **Convert timestamps** to elapsed seconds from the first recording.
3. **Detect the swim start** from the vertical-acceleration peak in a known time window, then sync event timestamps to the accelerometer time axis.
4. **Filter** each axis with a 2nd-order Butterworth low-pass filter (20 Hz cutoff, 100 Hz sampling rate).
5. **Compute jerk** (time derivative of acceleration) per axis, plus jerk magnitude.
6. **Drop rows** with missing values (introduced by the derivative) and export processed CSVs.

## Modelling

### Window extraction (`src/build_windows.py`)

- Extracts 25-sample windows (0.25 s at 100 Hz) anchored at each stroke event.
- Uses 7 features: vertical, lateral, sagittal, `JerkV`, `JerkL`, `JerkS`, `JerkMag`.
- Labels each window by arm and phase (`LA`/`RA` × entry/catch/pull/push) — 8 classes.
- Saves the concatenated dataset to `Dat/Windows/windows.npz`.

### Model (`src/train_cnn.py`)

A compact 1D CNN:

```text
Conv1d(7, 64, k=5) -> BatchNorm -> ReLU
Conv1d(64, 128, k=5) -> BatchNorm -> ReLU -> MaxPool1d(2)
Conv1d(128, 128, k=3) -> BatchNorm -> ReLU
AdaptiveAvgPool1d(1) -> Flatten -> Dropout(0.3) -> Linear(128, 8)
```

Training details:

- **120 epochs**, AdamW (`lr=3e-3`, `weight_decay=1e-4`), cosine-annealing learning-rate schedule.
- **Stratified 80/20** train/test split, held constant per seed.
- **Per-channel z-score normalisation** using train-set statistics only (no test leakage).
- Checkpoint stores state dict, normalisation statistics, and class labels.

## Results

### Feature ablation: does jerk help?

`src/train_accel_only.py` trains an otherwise identical 3-channel (acceleration-only) model on the **same splits as the 7-channel model** across five seeds. Positive delta favours jerk features.

| Metric | Accel + jerk (7ch) | Accel only (3ch) | Delta |
| --- | --- | --- | --- |
| Accuracy | 0.588 ± 0.036 | 0.602 ± 0.038 | −0.014 ± 0.034 |
| Macro F1 | 0.588 ± 0.035 | 0.602 ± 0.038 | −0.014 ± 0.032 |

The deltas are smaller than the seed-to-seed spread, so jerk features provide **no measurable benefit** over raw acceleration. The simpler 3-channel model is the better default.

### Per-class F1 (mean ± sd across 5 seeds)

| Class | Support (test, summed over seeds) | F1 (7ch) | F1 (3ch) |
| --- | --- | --- | --- |
| LA entry | 160 | 0.501 ± 0.043 | 0.520 ± 0.056 |
| LA catch | 160 | 0.684 ± 0.095 | 0.706 ± 0.066 |
| LA pull | 165 | 0.669 ± 0.046 | 0.704 ± 0.024 |
| LA push | 165 | 0.512 ± 0.054 | 0.550 ± 0.072 |
| RA entry | 170 | 0.447 ± 0.073 | 0.442 ± 0.055 |
| RA catch | 165 | 0.697 ± 0.058 | 0.708 ± 0.041 |
| RA pull | 165 | 0.712 ± 0.032 | 0.712 ± 0.039 |
| RA push | 165 | 0.482 ± 0.065 | 0.473 ± 0.091 |

Catch and pull phases are classified best (F1 ≈ 0.68–0.71); entry and push are the hardest (F1 ≈ 0.45–0.52).

### Error analysis (`src/pair_analysis.py`)

Decomposing the 8-class macro F1 over five seeds:

| View | Macro F1 (mean ± sd) |
| --- | --- |
| 8-class (full task) | 0.588 ± 0.035 |
| Arm only (LA vs RA) | 0.672 ± 0.021 |
| Phase only (arm collapsed) | 0.610 ± 0.033 |

- **79.5% of errors pick the wrong arm**, yet arm identity alone is the easier prediction. The errors are dominated by cross-arm confusion at opposite phases — for example, `LA entry` vs `RA push` and `RA entry` vs `LA push/LP pull` — which are temporally adjacent events in the stroke cycle.
- Per-class F1 correlates with the median distance to the nearest opposite-arm event (**Spearman ρ = +0.76, p = 0.029, n = 8**): the closer a phase sits to an opposite-arm event, the harder it is to classify.

## Interpretation

The model comfortably beats chance and learns the well-separated phases (catch, pull), but it struggles with the temporally adjacent cross-arm boundary events (entry vs the other arm's push). This suggests the remaining error is a **temporal-context** problem rather than a feature problem: 25 samples (0.25 s) may not carry enough history to disambiguate phases that sit close together in the stroke cycle. This is consistent with the intuition behind sequence models, motivating the next steps below.

## Limitations

- **Subject leakage in the split.** The 80/20 split stratifies pooled windows, so windows from the same swimmer can appear in both train and test. Reported figures are therefore optimistic for subject-independent generalisation; leave-one-swimmer-out evaluation is the next step.
- **Small cohort.** Six participants and ~1,300 windows bound how far conclusions generalise to new swimmers.
- **Manual labels.** Event boundaries come from human video review, which carries some labelling noise.

## Next steps

- Evaluate with **leave-one-swimmer-out** cross-validation to estimate subject-independent performance.
- Add a **sequence model** (e.g. a lightweight Transformer encoder over the CNN features) to model stroke-phase ordering, which the error analysis points toward.
- Explore **data augmentation** and per-arm/phase class weighting for the underperforming entry and push classes.
- Visualise attention/activations for interpretability of which signal windows drive each phase.

## Reproducibility

The private data must be present locally under `Dat/` (not committed). Dependencies: Python with `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`; R with `dplyr`, `lubridate`, `ggplot2`, `patchwork`, `RSEIS`.

```bash
# 1. Process each participant's raw data
Rscript Processing/AC.R     # repeat for IR, JD, JH, KV, RW

# 2. Build labelled event windows
python src/build_windows.py

# 3. Train the 7-channel model and print a classification report
python src/train_cnn.py

# 4. Feature ablation: 7-channel vs 3-channel over 5 seeds
python src/train_accel_only.py

# 5. Error analysis: arm vs phase confusion
python src/pair_analysis.py
```

## Repository structure

```text
Processing/<ID>.R        per-participant signal-processing scripts
src/build_windows.py     window extraction
src/train_cnn.py         7-channel CNN training
src/train_accel_only.py  3-channel baseline and model comparison
src/pair_analysis.py     error analysis (arm vs phase)
Dat/                     raw/processed data and artefacts (git-ignored, private)
```

## What this project demonstrates

- End-to-end **time-series signal processing** (filtering, differentiation, synchronisation).
- **Feature engineering evaluated honestly** — in this case showing the simpler model wins.
- **Deep learning** model design, training, and normalisation without test leakage.
- **Rigorous evaluation**: multi-seed reproducibility, per-class metrics, and error decomposition.
- Clear, honest reporting of **results, limitations, and next steps**.