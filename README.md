# FSED-Pilot

Freestyle stroke-event detection from a single wearable accelerometer.

Pilot experiment for [Freestyle-Stroke-Event-Detection](https://github.com/RyanWoodward132/Freestyle-Stroke-Event-Detection).

## Overview

This project teaches a model to recognise what a swimmer is doing at any moment of the freestyle stroke, using only a small motion sensor worn on the upper back. Every stroke event falls into one of eight classes: left or right arm, and one of four phases (entry, catch, pull, push).

Two models are compared. The first looks at a 0.25-second snippet of motion around each event and scores 0.61. The second also sees the four events before and the four after, and scores 0.76. The reason context helps so much is simple: when the first model gets something wrong, it usually has the right shape of movement but the wrong arm, and you can only tell those apart if you know where you are in the stroke cycle.

Scores below are accuracy and macro F1. Both run from 0 to 1; macro F1 treats all eight classes equally, even the harder ones. Random guessing scores 0.125. The ± values show how much results vary between runs.

## Data

Ten competitive swimmers swam a 100 m maximal-effort freestyle wearing an inertial measurement unit (a small sensor recording acceleration in three directions, 100 times per second) on the upper back. Each stroke event was labelled by hand from synchronised video.

Six swimmers (`AC`, `IR`, `JD`, `JH`, `KV`, `RW`) were carried through the full pipeline and used for modelling, giving 1,313 labelled events, roughly balanced across the eight classes (159 to 169 each).

The raw data and videos are private and not part of this repository. The code here documents the full pipeline.

## Processing pipeline

Each swimmer's raw data goes through the same steps (one R script per swimmer in `Processing/`):

1. Load the raw accelerometer data and the event timestamps.
2. Convert timestamps to seconds from the start of the recording.
3. Find the moment the swim starts (a clear spike in vertical acceleration) and line the event labels up with the sensor timeline.
4. Smooth each axis with a low-pass filter to remove high-frequency noise.
5. Compute jerk (how quickly acceleration changes) for each axis.
6. Drop incomplete rows and export the cleaned files.

`src/build_windows.py` then cuts a 0.25-second window (25 samples) of the signal at each event, attaches its class label, and saves everything to `Dat/Windows/windows.npz`. It also stores when each event happened, because the event files are not stored in time order and the second model needs events in the order they actually occurred.

## Base model

The base model is a small neural network (a 1D convolutional network, or CNN) that looks at one window at a time and learns which patterns of movement belong to which class:

```text
Conv1d(3, 64, k=5) -> BatchNorm -> ReLU
Conv1d(64, 128, k=5) -> BatchNorm -> ReLU -> MaxPool1d(2)
Conv1d(128, 128, k=3) -> BatchNorm -> ReLU
AdaptiveAvgPool1d(1) -> Flatten -> Dropout(0.3) -> Linear(128, 8)
```

It is trained on 80% of the windows and tested on the remaining 20%, with the mix of classes kept the same in both halves. Windows are standardised using statistics from the training half only, so the model never gets a sneak peek at the test data.

We first tested a version that also used the jerk channels as extra inputs. It did no better (0.588 vs 0.602 macro F1 over five runs), so the simpler three-channel model is the baseline for everything below.

## Extended model

The CNN's mistakes suggested the problem was missing context rather than missing features, so the extended model shows the CNN the surrounding events as well. Each event is placed in the middle of a sequence of nine consecutive events from that swimmer: the event to classify (the centre event) plus the four that happened just before it and the four just after. That covers a little over one full stroke cycle. The context events are raw signal only; the model is never told their labels.

```text
per event: the CNN above (without its final layer)     => one 128-number summary per event
sequence:  + learned position information (slots 1 to 9)
           -> TransformerEncoder(d_model=128, 4 heads, 2 layers, ff=256, dropout=0.1)
           -> read out the middle slot -> Dropout(0.3) -> Linear(128, 8)
```

The second stage is a transformer, a sequence model that can learn what tends to come before and after each phase. Both models are trained with identical settings and identical data splits, so any difference in score comes from the context, not the setup. Because sequences are built from the full event stream, the events either side are always the ones that really happened just before and after, in both training and testing.

The extended model is evaluated two ways:

- **Shared 80/20 split**: the same split as the base model. A sequence belongs to whichever half its centre event landed in, so events from one half can appear as unlabelled context in the other. This is the same level of information sharing the base model already has (windows from the same swimmer appear in both halves).
- **Leave-one-swimmer-out**: train on five swimmers, test on the sixth, and rotate until every swimmer has been the test case. Nothing from the test swimmer is seen during training. This is the harder, more realistic test of whether the model works on someone new.

## Results

Shared 80/20 split, both models trained on the same splits over five runs:

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| CNN (base) | 0.610 ± 0.036 | 0.611 ± 0.036 |
| Transformer (extended) | **0.763 ± 0.021** | **0.764 ± 0.022** |

That is a gain of +0.154 macro F1, and every class improves. The biggest gains are on the classes the CNN found hardest:

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

Leave-one-swimmer-out, five runs per swimmer:

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

The CNN's mistakes are not random. When it is wrong, about four times out of five it has picked the wrong arm at a nearby phase, for example confusing a left-arm entry with a right-arm push. These are events that happen close together in the stroke cycle and look similar in a quarter-second window. Phases that sit further apart, like catch and pull, are classified well; the boundary phases, entry and push, are the hardest.

This is exactly where the context helps. The transformer's largest gains are on the entry and push classes, the ones that need stroke-cycle order to tell apart. RA entry, the CNN's weakest class at 0.450, rises to 0.672.

The leave-one-swimmer-out results keep the headline number honest. On an unseen swimmer both models drop to around 0.30, which means a large part of the 0.76 comes from recognising individual swimmers' movement styles rather than universal stroke structure. The transformer still beats the CNN for four of the six swimmers, and again mainly on the entry and push classes, but with only six swimmers that edge is within the noise. The fairest reading: on a new swimmer, the extended model's true score likely sits somewhere between 0.30 and 0.76, and pinning that down needs more data.

## Limitations

- **The 80/20 split shares swimmers.** Windows from the same swimmer appear in both halves, so the 0.76 figure is optimistic for use on someone new. The leave-one-swimmer-out results are the more realistic estimate.
- **Small cohort.** Six swimmers and about 1,300 labelled events can only take conclusions so far, and it makes the per-swimmer results noisy.
- **Hand-made labels.** Event times come from human video review, which is never perfectly precise.
- **Events are given, not found.** Both models classify around known event times. Using this in practice would first require a step that finds stroke events in a raw recording.

## Next steps

- Collect data from more swimmers. This would tighten the leave-one-swimmer-out estimate more than any model tuning would.
- Add a step that finds stroke events automatically, so the pipeline can run on raw recordings instead of hand-marked events.
- Try data augmentation and class weighting for the entry and push phases.
- Look at what the transformer attends to, to check whether it is really using the neighbouring events we think it is.
