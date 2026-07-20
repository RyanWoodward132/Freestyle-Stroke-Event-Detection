# FSED-Pilot
## Overview
Pilot experiment of [Freestyle-Stroke-Event-Detection](https://github.com/RyanWoodward132/CNN-Stroke-Event-Detection).

The goal of this project is to build a Convolutional Neural Network that classifies three stroke events in elite level freestyle (Entry, Pull, Push) from 3-axis accelerometer data and derived features.

## Status
| Stage | Status |
|---|---|
| Data collection | Complete |
| Data processing | In progress |
| Model building and training | Pending |
| Model evaluation | Pending |

## Data Collection
10 participants wore an Inertial Measurement Unit (IMU) on their upper back's while performing 100m of maximal effort freestyle. The accelerometer recorded movement along three axis - Vertical, Lateral, and Sagittal - at 100Hz and stroke events were manually labelled by reviewing synchronised video recordings.

For each participant an `accel` and `events` file were created:

- `accel`: Contains acceleration measurements across the three axis
- `events`: Contains stroke event timestamps corresponding to events occuring in `accel`

## Data Processing
Section will be updated once this stage is completed

## Model Building
Section will be updated once this stage is completed

## Model Evaluation
Section will be updated once this stage is completed