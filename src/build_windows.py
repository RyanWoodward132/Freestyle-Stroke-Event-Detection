"""Build 25-sample (0.25 s @ 100 Hz) windows starting at each stroke event."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

DAT = Path(os.environ.get("FSED_DAT", Path(__file__).resolve().parents[1] / "Dat"))
WIN = 25          # 0.25 s at 100 Hz
FS = 100
CHANNELS = ["Vertical", "Lateral", "Sagittal", "JerkV", "JerkL", "JerkS", "JerkMag"]
CLASSES = [f"{a} {e}" for a in ("LA", "RA") for e in ("entry", "catch", "pull", "push")]


def label_of(event: str) -> str:
    """'LA push (upsweep)' -> 'LA push'. Empty for Start / Break water / blanks."""
    parts = str(event).split()
    return f"{parts[0]} {parts[1]}" if len(parts) >= 2 and parts[0] in ("LA", "RA") else ""


def participant(pid: str):
    acc = pd.read_csv(DAT / "Processed/Accel" / f"{pid}.csv", index_col=0)
    ev = pd.read_csv(DAT / "Processed/Events" / f"{pid}.csv", index_col=0).dropna(subset=["Synced"])
    ev = ev.drop_duplicates(["Event", "Synced"])   # a few exact repeats in every file
    assert ev["Synced"].nunique() > 1, f"{pid}: events collapsed to one timestamp -- rerun Processing/{pid}.R"
    t = ev["Synced"].to_numpy()

    labels = np.array([label_of(e) or "" for e in ev["Event"]])
    keep = labels != ""

    time = acc["Time"].to_numpy()
    assert np.allclose(np.diff(time), 1 / FS), f"{pid}: gaps in the time grid -- windows would span them"
    idx = np.searchsorted(time, t[keep] - 0.5 / FS)          # event time -> first sample
    labels = labels[keep]

    ok = idx <= len(time) - WIN                               # window must fit
    idx, labels = idx[ok], labels[ok]
    assert np.abs(time[idx] - t[keep][ok]).max() < 1 / FS, f"{pid}: event/sample misalignment"

    feat = acc[CHANNELS].to_numpy(np.float32)
    X = sliding_window_view(feat, WIN, axis=0)[idx]           # (n, channels, WIN)
    y = np.array([CLASSES.index(l) for l in labels], np.int64)
    return X, y, np.full(len(y), pid), (~ok).sum()


def main():
    pids = sorted(p.stem for p in (DAT / "Processed/Accel").glob("*.csv"))
    Xs, ys, gs = [], [], []
    for pid in pids:
        X, y, g, dropped = participant(pid)
        print(f"{pid}: {len(y):4d} windows" + (f"  ({dropped} past end of recording)" if dropped else ""))
        Xs.append(X), ys.append(y), gs.append(g)

    X, y, groups = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)
    assert X.shape[1:] == (len(CHANNELS), WIN)

    out = DAT / "Windows/windows.npz"
    np.savez_compressed(out, X=X, y=y, groups=groups, classes=CLASSES, channels=CHANNELS)
    print(f"\n{X.shape} -> {out}")
    for i, c in enumerate(CLASSES):
        print(f"  {c:9s} {(y == i).sum()}")


if __name__ == "__main__":
    main()
