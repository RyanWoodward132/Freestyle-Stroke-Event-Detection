"""Where does the 8-class F1 go? Split arm errors from phase errors, over N seeds."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

from train_cnn import DAT, load, run

SEEDS = range(5)
classes = load()[2]
ARM = np.array([c.split()[0] == "RA" for c in classes], int)      # 0 = LA, 1 = RA
PHASE = np.array([["entry", "catch", "pull", "push"].index(c.split()[1]) for c in classes])


def contralateral_gap():
    """Median seconds from each event to the nearest event of the *other* arm."""
    def lab(e):
        p = str(e).split()
        return f"{p[0]} {p[1]}" if len(p) >= 2 and p[0] in ("LA", "RA") else ""

    t = {c: [] for c in classes}
    for f in sorted((DAT / "Processed/Events").glob("*.csv")):
        ev = pd.read_csv(f, index_col=0).dropna(subset=["Synced"]).drop_duplicates(["Event", "Synced"])
        for e, s in zip(ev["Event"], ev["Synced"]):
            if lab(e):
                t[lab(e)].append(s)
    t = {c: np.sort(np.array(v)) for c, v in t.items()}

    gap = []
    for i, a in enumerate(classes):
        best = np.inf
        for j, b in enumerate(classes):
            if ARM[i] == ARM[j]:
                continue
            k = np.clip(np.searchsorted(t[b], t[a]), 1, len(t[b]) - 1)
            d = np.minimum(np.abs(t[a] - t[b][k - 1]), np.abs(t[a] - t[b][k]))
            best = min(best, np.median(d))
        gap.append(best)
    return np.array(gap)


rows, per_class, wrong_arm = [], [], []
for s in SEEDS:
    y, p = run(s)
    per_class.append(f1_score(y, p, average=None, labels=range(len(classes)), zero_division=0))
    rows.append(dict(
        total=f1_score(y, p, average="macro", zero_division=0),
        arm=f1_score(ARM[y], ARM[p], average="macro", zero_division=0),
        phase=f1_score(PHASE[y], PHASE[p], average="macro", zero_division=0),
    ))
    err = y != p
    wrong_arm.append(((ARM[y] != ARM[p]) & err).sum() / err.sum())
    print(f"seed {s}: macro F1 {rows[-1]['total']:.3f}")

per_class = np.array(per_class)
m = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
sd = {k: np.std([r[k] for r in rows]) for k in rows[0]}

print(f"\n{len(list(SEEDS))} seeds, macro F1 (mean +/- sd)\n")
print(f"  8-class total               {m['total']:.3f} +/- {sd['total']:.3f}")
print(f"  arm only   (LA vs RA)       {m['arm']:.3f} +/- {sd['arm']:.3f}")
print(f"  phase only (arm collapsed)  {m['phase']:.3f} +/- {sd['phase']:.3f}")
print(f"\n  errors that pick the wrong arm: {np.mean(wrong_arm):.1%}")

gap = contralateral_gap()
f1m = per_class.mean(0)
rho, p_val = spearmanr(gap, f1m)
print(f"\nper-class F1 vs distance to nearest opposite-arm event:\n")
for i in np.argsort(gap):
    print(f"  {classes[i]:9s} gap {gap[i]:.3f}s   F1 {f1m[i]:.3f} +/- {per_class[:, i].std():.3f}")
print(f"\n  Spearman rho = {rho:+.3f}  (p = {p_val:.3f}, n = 8)")
