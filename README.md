# Freestyle Stroke Event Detection

Freestyle stroke-event detection from a single wearable accelerometer.

## Overview

This project classifies the stroke phases of elite freestyle swimmers from a single three-axis accelerometer worn on the upper back. For each stroke event the model distinguishes eight classes: left or right arm, and entry, catch, pull or push.

Two models are compared. A compact 1D CNN classifies a single 0.25 s window around each event and reaches 0.61 macro F1. A lightweight transformer that sees the surrounding nine events reaches 0.76, and its gains land on the same phases the CNN keeps confusing. That pattern has an explanation: about four in five CNN errors pick the wrong arm at a neighbouring phase, which is precisely the kind of mistake that temporal context can fix.

## Data

Ten competitive swimmers performed a 100 m maximal-effort freestyle swim wearing an inertial measurement unit (IMU) on the upper back. The accelerometer recorded three axes (vertical, lateral, sagittal) at 100 Hz, and stroke events were labelled by hand from synchronised video.

Six participants (`AC`, `IR`, `JD`, `JH`, `KV`, `RW`) were carried through the full pipeline and used for modelling, giving 1,313 labelled event windows (159 to 169 per class, roughly balanced).

Raw accelerometer data and videos are withheld for privacy and are not part of this repository. The committed code only documents the complete processing and modelling pipeline.

## Processing pipeline

Per-participant R scripts in `Processing/<ID>.R` run an identical pipeline:

1. Load raw accelerometer data and event timestamps.
2. Convert timestamps to elapsed seconds from the first recording.
3. Detect the swim start from the vertical-acceleration peak in a known time window, then sync event timestamps to the accelerometer time axis.
4. Filter each axis with a 2nd-order Butterworth low-pass filter (20 Hz cutoff, 100 Hz sampling rate).
5. Compute jerk (time derivative of acceleration) per axis, plus jerk magnitude.
6. Drop rows with missing values (introduced by the derivative) and export processed CSVs.

`src/build_windows.py` then extracts a 25-sample window (0.25 s at 100 Hz) anchored at each stroke event, labels it by arm and phase, and saves everything to `Dat/Windows/windows.npz`. The file holds seven channels (three acceleration, three jerk, jerk magnitude) plus each event's synced timestamp. The models use the acceleration channels only; the timestamps are what the extended model needs to order events, because the event CSVs are not stored in time order.

## Base model

The base model is a compact 1D CNN over the three acceleration channels:

```text
Conv1d(3, 64, k=5) -> BatchNorm -> ReLU
Conv1d(64, 128, k=5) -> BatchNorm -> ReLU -> MaxPool1d(2)
Conv1d(128, 128, k=3) -> BatchNorm -> ReLU
AdaptiveAvgPool1d(1) -> Flatten -> Dropout(0.3) -> Linear(128, 8)
```

Training: 120 epochs, AdamW (`lr=3e-3`, `weight_decay=1e-4`), cosine-annealing schedule, stratified 80/20 train/test split held constant per seed, per-channel z-score normalisation using training-split statistics only.

A 7-channel variant adding the jerk features was evaluated first and did no better (0.588 vs 0.602 macro F1 over five seeds, with deltas smaller than the seed-to-seed spread), so the 3-channel model is the default and the baseline for everything below.

## Extended model

The CNN's errors suggested the remaining problem was context, not features, so the extended model keeps the CNN as a per-event encoder and adds a transformer over the event sequence.

Sequence construction: within each swimmer, events are ordered by their synced timestamps, and each event becomes the centre of a sequence of nine consecutive events: the event to be classified (the **centre event**) plus the four before it and the four after it as context, just over one full stroke cycle (eight events). Sequences at the ends of a recording replicate the edge events, with an attention mask over the padding. Building sequences over the full event stream keeps the context spacing uniform: in both train and test, the four events either side of a centre event are its true temporal neighbours, so the model is trained and tested on the same kind of sequence.

```text
per event: CNN conv stack above (no classifier head)     => 128-d embedding
sequence:  + learned positional embedding (9 positions)
           -> TransformerEncoder(d_model=128, 4 heads, 2 layers, ff=256, dropout=0.1)
           -> centre token -> Dropout(0.3) -> Linear(128, 8)
```

The training recipe is unchanged. The model is evaluated two ways:

- **Pooled stratified 80/20**, identical seeds to the base model. Sequences are built over all events and assigned to train or test by their centre event. Test events can appear as unlabelled context inside training sequences; labels never cross the split. This is the same class of leakage the pooled regime already accepts for individual windows.
- **Leave-one-swimmer-out (LOSO)**: six folds, each holding out one swimmer entirely, so nothing from the test swimmer, context included, is seen in training.

## Results

Pooled 80/20, paired runs over five seeds (`src/train_transformer.py`):

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| CNN (base) | 0.610 ± 0.036 | 0.611 ± 0.036 |
| Transformer (extended) | **0.763 ± 0.021** | **0.764 ± 0.022** |

Paired delta in macro F1: **+0.154 ± 0.018**. Per-class F1 improves on every class, most of all the boundary classes:

| Class | CNN F1 | Transformer F1 |
| --- | --- | --- |
| LA entry | 0.557 ± 0.059 | **0.678 ± 0.036** |
| LA catch | 0.704 ± 0.065 | **0.815 ± 0.049** |
| LA pull | 0.721 ± 0.023 | **0.830 ± 0.019** |
| LA push | 0.562 ± 0.051 | **0.732 ± 0.034** |
| RA entry | 0.450 ± 0.054 | **0.672 ± 0.067** |
| RA catch | 0.712 ± 0.043 | **0.880 ± 0.026** |
| RA pull | 0.698 ± 0.036 | **0.865 ± 0.067** |
| RA push | 0.480 ± 0.104 | **0.641 ± 0.053** |

LOSO, six folds with five seeds each (mean over seeds per fold):

| Test swimmer | CNN acc | Transformer acc | CNN macro F1 | Transformer macro F1 |
| --- | --- | --- | --- | --- |
| AC | 0.223 | **0.308** | 0.185 | **0.271** |
| IR | **0.195** | 0.177 | **0.173** | 0.168 |
| JD | 0.446 | **0.475** | 0.427 | **0.455** |
| JH | 0.326 | **0.382** | 0.304 | **0.370** |
| KV | **0.364** | 0.310 | **0.321** | 0.294 |
| RW | 0.244 | **0.259** | 0.211 | **0.251** |
| **Overall (mean ± sd)** | 0.300 ± 0.093 | **0.318 ± 0.115** | 0.270 ± 0.095 | **0.301 ± 0.110** |

## Interpretation

The CNN's errors are mostly not random. Decomposing its 8-class macro F1 (0.588 over five seeds): predicting only the arm scores 0.672, and predicting only the phase scores 0.610. Yet 79.5% of errors pick the wrong arm, usually at a neighbouring phase, such as LA entry confused with RA push. Per-class F1 also tracks the median distance to the nearest opposite-arm event (Spearman ρ = +0.76, p = 0.029, n = 8): the closer a phase sits to an opposite-arm event, the harder it is to classify. Catch and pull are well separated and classified best; entry and push sit next to the other arm's events and are the hardest.

This is why the transformer helps where it does. Under the pooled split its biggest gains are RA entry (0.450 to 0.672) and LA entry (0.557 to 0.678), the two weakest CNN classes, followed by the push phases. The model is using stroke-cycle order to tell apart events that look alike in a 0.25 s window.

The LOSO numbers keep this honest. Both models lose roughly half their pooled performance when the test swimmer is unseen, which says the pooled figures lean heavily on swimmer-specific patterns rather than general stroke structure. The transformer still finishes ahead on four of six folds (+0.031 ± 0.081 macro F1), with its gains again concentrated on entry and push, but at six swimmers that delta is inside the fold-to-fold noise. Treat 0.764 as an upper bound and 0.301 as a lower bound on what the extended model would do on a new swimmer.

## Limitations

- **Subject leakage in the pooled split.** Windows from the same swimmer appear in both train and test, so the pooled figures are optimistic for subject-independent generalisation. The LOSO results above are the more realistic estimate.
- **Small cohort.** Six participants and about 1,300 windows bound how far any of this generalises, and they make the LOSO folds noisy.
- **Manual labels.** Event boundaries come from human video review, which carries labelling noise.
- **Events are given, not detected.** Both models classify windows around known event timestamps. Running on a raw stream would need an event-detection step in front, and the extended model additionally needs the surrounding events located before it can classify.

## Next steps

- Grow the cohort. More swimmers would tighten the LOSO estimate and matter more than any model tuning.
- Add an event-detection front end so the pipeline runs on raw accelerometer streams rather than hand-marked events.
- Explore data augmentation and per-class weighting for the entry and push phases.
- Visualise the transformer's attention to check whether it attends to the adjacent opposite-arm events the error analysis implicates.
