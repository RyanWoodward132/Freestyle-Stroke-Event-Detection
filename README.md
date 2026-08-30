# FSED-Pilot

## Overview

Pilot experiment of [Freestyle-Stroke-Event-Detection](https://github.com/RyanWoodward132/Freestyle-Stroke-Event-Detection).

The goal of this project is to build a Convolutional Neural Network that classifies three stroke events in elite level freestyle (Entry, Pull, Push) from 3-axis accelerometer data and derived features.

## Status

| Stage | Status |
| --- | --- |
| Data collection | Complete |
| Data processing | Complete |
| Model building and training | Complete |
| Model evaluation | Pending |

## Data Collection

10 participants wore an Inertial Measurement Unit (IMU) on their upper back's while performing 100m of maximal effort freestyle. The accelerometer recorded movement along three axis - Vertical, Lateral, and Sagittal - at 100Hz and stroke events were manually labelled by reviewing synchronised video recordings.

For each participant an `accel` and `events` file were created:

- `accel`: Contains acceleration measurements across the three axis
- `events`: Contains stroke event timestamps corresponding to events occuring in `accel`

Due to privacy concerns these data and video files will not be made public.

## Data Processing

Each participant's raw data is processed in `Processing/<ID>.R` (R) using the same pipeline:

1. **Load** raw accelerometer (`Dat/Raw/Accel/<ID>.csv`) and event timestamps (`Dat/Raw/Events/<ID>.csv`).
2. **Convert timestamps** to elapsed seconds from the first recording.
3. **Identify the swim start** by locating the vertical acceleration peak in a known time window and computing the midpoint between the min and max.
4. **Sync event timestamps** to the accelerometer time axis using the detected start offset.
5. **Filter** each axis with a 2nd-order Butterworth low-pass filter (20 Hz cutoff, 100 Hz sampling rate).
6. **Compute jerk** (time derivative of acceleration) per axis and jerk magnitude (`sqrt(JerkV^2 + JerkL^2 + JerkS^2)`).
7. **Remove rows with missing values** (from the jerk derivative).
8. **Export** processed accel and events CSVs to `Dat/Processed/`.

Processed data is available for all 6 participants (AC, IR, JD, JH, KV, RW).

## Model Building

A 1D CNN is trained on sliding windows extracted from the processed accelerometer data.

### Window extraction (`src/build_windows.py`)

- Extracts 25-sample (0.25 s at 100 Hz) windows starting at each stroke event.
- Uses 7 features: Vertical, Lateral, Sagittal, JerkV, JerkL, JerkS, JerkMag.
- Labels each window by stroke type: LA/RA (left/right arm) × entry/catch/pull/push (8 classes).
- Concatenates windows across all 6 participants (1313 total windows).
- Saves to `Dat/Windows/windows.npz`.

### Training (`src/train_cnn.py`)

- **Architecture**: 3-layer 1D ConvNet (Conv1d → BatchNorm → ReLU) × 3, followed by AdaptiveAvgPool1d, Dropout (0.3), and a Linear output layer.
  - Conv1d: 7→64 (kernel 5), 64→128 (kernel 5), 128→128 (kernel 3)
  - Output: 8 classes
- **Training**: 120 epochs, AdamW optimizer (lr=3e-3, weight_decay=1e-4), cosine annealing LR schedule, stratified 80/20 train/test split.
- **Normalization**: Per-channel z-score using train-set statistics (saved with the model).
- **Output**: Model checkpoint saved to `Dat/Windows/cnn.pt` (state dict, normalization stats, class labels).

## Model Evaluation

Section will be updated once this stage is completed
