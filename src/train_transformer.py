"""LOSO evaluation: 3-channel CNN baseline vs a transformer over event sequences.

Each labelled event becomes the centre of a length-K sequence of consecutive
events (Synced-time order) from the same swimmer; both models classify the
centre event. Folds are leave-one-swimmer-out, so no window or context from
the test swimmer is seen in training. 6 folds x 5 seeds for each model.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # MPS lacks the padding-mask check op

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from torch import nn

from train_cnn import BATCH, DAT, EPOCHS

K = 9                       # one full stroke cycle (8 events) of context
CENTER = K // 2
SEEDS = range(5)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = DAT / "Windows/transformer.pt"


def load():
    d = np.load(DAT / "Windows/windows.npz", allow_pickle=False)
    return d["X"][:, :3].astype(np.float32), d["y"], d["groups"], d["t"], [str(c) for c in d["classes"]]


def sequence_index(groups, t, keep=None):
    """(n, K) event indices + (n, K) padding mask; sequence j classifies its centre event.

    With `keep` (bool over events), sequences use only kept events: context and
    centre never cross the kept/dropped boundary (used to sequence within a split).
    """
    pos = np.arange(-CENTER, CENTER + 1)
    rows, masks = [], []
    for pid in np.unique(groups):
        idx = np.flatnonzero(groups == pid)
        idx = idx[np.argsort(t[idx], kind="stable")]      # CSV rows are not chronological
        if keep is not None:
            idx = idx[keep[idx]]
        for i in range(len(idx)):
            rows.append(idx[np.clip(i + pos, 0, len(idx) - 1)])
            masks.append((i + pos < 0) | (i + pos >= len(idx)))
    return np.array(rows), np.array(masks)


def cnn_trunk():
    """The train_accel_only conv stack minus the classifier head -> 128-d embedding."""
    return nn.Sequential(
        nn.Conv1d(3, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
        nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(),
    )


class CNN(nn.Module):
    """Single-window baseline, identical to train_accel_only.build_model(3, 8)."""

    def __init__(self, n_classes):
        super().__init__()
        self.trunk = cnn_trunk()
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):                                # (B, 3, WIN)
        return self.head(self.trunk(x))


class EventTransformer(nn.Module):
    """CNN embedding per event + 2-layer transformer encoder; centre token is classified."""

    def __init__(self, n_classes):
        super().__init__()
        self.trunk = cnn_trunk()
        self.pos = nn.Parameter(torch.zeros(1, K, 128))
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x, pad):                           # (B, K, 3, WIN), pad True = ignore
        B, K_ = x.shape[:2]
        z = self.trunk(x.reshape(B * K_, *x.shape[2:])).reshape(B, K_, -1)
        z = self.encoder(z + self.pos, src_key_padding_mask=pad)
        return self.head(z[:, CENTER])


def fit(model, feats, labels):
    model.to(DEVICE)
    feats = [torch.as_tensor(f).to(DEVICE) for f in feats]
    ytr = torch.as_tensor(labels).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        model.train()
        for i in torch.randperm(len(ytr), device=DEVICE).split(BATCH):
            opt.zero_grad()
            loss_fn(model(*[f[i] for f in feats]), ytr[i]).backward()
            opt.step()
        sched.step()


@torch.no_grad()
def predict(model, feats):
    model.eval()
    feats = [torch.as_tensor(f).to(DEVICE) for f in feats]
    out = [model(*[f[i:i + 512] for f in feats]).argmax(1)
           for i in range(0, len(feats[0]), 512)]
    return torch.cat(out).cpu().numpy()


def run():
    X, y, groups, t, classes = load()
    seq_of, seq_mask = sequence_index(groups, t)
    seq_y = y[seq_of[:, CENTER]]
    assert len(seq_y) == len(y), "one sequence per event"
    logo = LeaveOneGroupOut()

    rows = []           # (fold, pid, seed, model, acc, macro_f1, per_class_f1)
    for fold, (tr, te) in enumerate(logo.split(X, y, groups)):
        pid = np.unique(groups[te])[0]
        mu = X[tr].mean((0, 2), keepdims=True)          # train swimmers only
        sd = X[tr].std((0, 2), keepdims=True)
        Xn = (X - mu) / sd
        S, M = Xn[seq_of], seq_mask                     # gather sequences after normalising
        tr_s, te_s = np.isin(seq_of[:, CENTER], tr), np.isin(seq_of[:, CENTER], te)

        for seed in SEEDS:
            torch.manual_seed(seed)
            cnn = CNN(len(classes))
            fit(cnn, (Xn[tr],), y[tr])
            p_cnn = predict(cnn, (Xn[te],))

            torch.manual_seed(seed)
            tf = EventTransformer(len(classes))
            fit(tf, (S[tr_s], M[tr_s]), seq_y[tr_s])
            p_tf = predict(tf, (S[te_s], M[te_s]))

            for name, yt, yp in (("cnn", y[te], p_cnn), ("transformer", seq_y[te_s], p_tf)):
                rows.append((fold, pid, seed, name, accuracy_score(yt, yp),
                             f1_score(yt, yp, average="macro"),
                             f1_score(yt, yp, average=None, zero_division=0)))  # type: ignore[call-overload]
            print(f"fold {fold} ({pid}) seed {seed}: "
                  f"cnn acc={rows[-2][4]:.3f}  transformer acc={rows[-1][4]:.3f}", flush=True)
    return rows, classes, (X, y, groups, seq_of, seq_mask, seq_y)


def report(rows, classes):
    import pandas as pd
    df = pd.DataFrame([r[:6] for r in rows],
                      columns=["fold", "pid", "seed", "model", "acc", "f1"])
    piv = df.pivot_table(index=["pid"], columns="model", values=["acc", "f1"])
    print("\nPer-fold (mean over seeds):")
    print(piv.round(3).to_string())
    print("\nOverall (mean +/- sd over fold x seed):")
    for m in df.model.unique():
        g = df[df.model == m]
        print(f"  {m:16s} acc {g.acc.mean():.3f} +/- {g.acc.std():.3f}"
              f"   macro F1 {g.f1.mean():.3f} +/- {g.f1.std():.3f}")
    d = df.pivot_table(index=["fold", "seed"], columns="model", values="f1")
    for m in df.model.unique():
        if m == "cnn":
            continue
        delta = d[m] - d["cnn"]
        print(f"  delta macro F1 ({m} - cnn): {delta.mean():+.3f} +/- {delta.std():.3f}")
    print("\nPer-class F1 (mean +/- sd):")
    for i, c in enumerate(classes):
        for m in df.model.unique():
            v = np.array([r[6][i] for r in rows if r[3] == m])
            print(f"  {c:9s} {m:16s} {v.mean():.3f} +/- {v.std():.3f}")
    return df, piv, d


def save_final(data, classes):
    """One transformer trained on all swimmers, for the checkpoint."""
    X, y, _, seq_of, seq_mask, seq_y = data
    mu, sd = X.mean((0, 2), keepdims=True), X.std((0, 2), keepdims=True)
    Xn = (X - mu) / sd
    torch.manual_seed(0)
    model = EventTransformer(len(classes))
    fit(model, (Xn[seq_of], seq_mask), seq_y)
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                "classes": classes, "K": K}, OUT)
    print(f"\nfinal model (all swimmers) -> {OUT}")


def run_pooled():
    """Stratified 80/20 split (same seeds as train_accel_only), sequences built
    within each split and swimmer so no context crosses the train/test boundary."""
    from sklearn.model_selection import train_test_split
    X, y, groups, t, classes = load()
    rows = []
    for seed in SEEDS:
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
        in_tr = np.isin(idx, tr)
        mu = X[tr].mean((0, 2), keepdims=True)          # train windows only
        sd = X[tr].std((0, 2), keepdims=True)
        Xn = (X - mu) / sd
        seq_tr, mask_tr = sequence_index(groups, t, in_tr)
        seq_te, mask_te = sequence_index(groups, t, ~in_tr)
        assert sorted(seq_tr[:, CENTER]) == sorted(tr)  # one sequence per event, no straddling
        assert sorted(seq_te[:, CENTER]) == sorted(te)

        # Full-context variant: sequences over ALL events (spacing 1), split by
        # centre event only. Test events can appear as unlabelled context in
        # train sequences -- same leakage class the pooled regime already accepts.
        seq_all, mask_all = sequence_index(groups, t)
        tr_a = np.isin(seq_all[:, CENTER], tr)
        te_a = np.isin(seq_all[:, CENTER], te)

        torch.manual_seed(seed)
        cnn = CNN(len(classes))
        fit(cnn, (Xn[tr],), y[tr])
        p_cnn = predict(cnn, (Xn[te],))

        torch.manual_seed(seed)
        tf = EventTransformer(len(classes))
        fit(tf, (Xn[seq_tr], mask_tr), y[seq_tr[:, CENTER]])
        p_tf = predict(tf, (Xn[seq_te], mask_te))

        torch.manual_seed(seed)
        tfc = EventTransformer(len(classes))
        fit(tfc, (Xn[seq_all][tr_a], mask_all[tr_a]), y[seq_all[tr_a, CENTER]])
        p_tfc = predict(tfc, (Xn[seq_all][te_a], mask_all[te_a]))

        for name, yt, yp in (("cnn", y[te], p_cnn),
                             ("transformer", y[seq_te[:, CENTER]], p_tf),
                             ("transformer_ctx", y[seq_all[te_a, CENTER]], p_tfc)):
            rows.append((seed, "pooled", seed, name, accuracy_score(yt, yp),
                         f1_score(yt, yp, average="macro"),
                         f1_score(yt, yp, average=None, zero_division=0)))  # type: ignore[call-overload]
        print(f"seed {seed}: " + "  ".join(f"{r[3]} acc={r[4]:.3f}" for r in rows[-3:]), flush=True)
    return rows, classes


if __name__ == "__main__":
    import sys
    if "loso" in sys.argv:
        rows, classes, data = run()
        report(rows, classes)
        save_final(data, classes)
    else:
        rows, classes = run_pooled()
        report(rows, classes)
