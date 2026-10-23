"""Train the accel-only CNN and compare it to the 7-channel model over several
seeds. Each seed uses the identical stratified 80/20 split for both models."""
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn

from train_cnn import BATCH, DAT, EPOCHS, load, run

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

SEEDS = range(5)
ACCEL_CHANNELS = ["Vertical", "Lateral", "Sagittal"]
OUT = DAT / "Windows/cnn_accel.pt"
REPORT = Path(__file__).resolve().parents[1] / "Model_Comparison_Report.md"


def build_model(in_channels, n_classes):
    return nn.Sequential(
        nn.Conv1d(in_channels, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
        nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        nn.Dropout(0.3), nn.Linear(128, n_classes),
    )


def run_accel(seed=0, save=None):
    """Train the accel-only model on the same seed split as train_cnn.run. Returns (y_true, y_pred)."""
    X, y, classes = load()
    classes = [str(c) for c in classes]
    Xa = X[:, :3, :]                                   # Vertical, Lateral, Sagittal only

    torch.manual_seed(seed)
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)

    mu = Xa[tr_idx].mean((0, 2), keepdims=True)
    sd = Xa[tr_idx].std((0, 2), keepdims=True)
    Xn = (Xa - mu) / sd

    model = build_model(Xa.shape[1], len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    Xtr_t, ytr_t = torch.tensor(Xn[tr_idx]), torch.tensor(y[tr_idx])

    for _ in range(EPOCHS):
        model.train()
        for i in torch.randperm(len(ytr_t)).split(BATCH):
            opt.zero_grad()
            loss_fn(model(Xtr_t[i]), ytr_t[i]).backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xn[te_idx])).argmax(1).numpy()

    if save:
        torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                    "classes": classes, "channels": ACCEL_CHANNELS, "test_idx": te_idx}, save)
    return y[te_idx], pred


def ms(x):
    return f"{np.mean(x):.3f} ± {np.std(x):.3f}"


def confusion_str(cm, classes):
    w = max(len(c) for c in classes)
    header = (" " * (w + 3)) + " ".join(f"{c:>{w}}" for c in classes)
    rows = [header]
    for i, c in enumerate(classes):
        rows.append(f"{c:>{w}}  " + " ".join(f"{v:>{w}}" for v in cm[i]))
    return "\n".join(rows)


def main():
    classes = [str(c) for c in load()[2]]
    n = len(classes)

    acc7, acc3, macro7, macro3 = [], [], [], []
    pcls7 = np.zeros((len(SEEDS), n))
    pcls3 = np.zeros((len(SEEDS), n))
    cm7 = np.zeros((n, n), int)
    cm3 = np.zeros((n, n), int)
    seed_rows = []

    for i, s in enumerate(SEEDS):
        y7, p7 = run(s)                                                        # 7-channel
        y3, p3 = run_accel(s, save=OUT if s == 0 else None)                    # 3-channel
        assert np.array_equal(y7, y3), f"split mismatch at seed {s}"           # models share the test set

        acc7.append(accuracy_score(y7, p7))
        acc3.append(accuracy_score(y3, p3))
        macro7.append(f1_score(y7, p7, average="macro"))
        macro3.append(f1_score(y3, p3, average="macro"))
        pcls7[i] = f1_score(y7, p7, labels=range(n), average=None)  # pyright: ignore[reportArgumentType]
        pcls3[i] = f1_score(y3, p3, labels=range(n), average=None)  # pyright: ignore[reportArgumentType]
        cm7 += confusion_matrix(y7, p7, labels=range(n))
        cm3 += confusion_matrix(y3, p3, labels=range(n))
        seed_rows.append((s, acc7[-1], acc3[-1], macro7[-1], macro3[-1]))
        print(f"seed {s}: 7ch acc {acc7[-1]:.3f} macro {macro7[-1]:.3f}  |  "
              f"3ch acc {acc3[-1]:.3f} macro {macro3[-1]:.3f}")

    acc7, acc3 = np.array(acc7), np.array(acc3)
    macro7, macro3 = np.array(macro7), np.array(macro3)

    lines = [
        "# Model Comparison: Raw Acceleration vs Acceleration + Jerk Features",
        "",
        "Two 1D CNNs with identical architecture and training procedure were trained",
        f"and evaluated over {len(SEEDS)} seeds (0-{max(SEEDS)}). For each seed both models",
        "use the identical stratified 80/20 train/test split, so they are evaluated on",
        "the same test windows. The only difference is the input channels: the current",
        "model uses 7 (acceleration + jerk), the accel-only model uses 3 (acceleration).",
        "",
        "Positive delta = jerk features improve the metric (7ch - 3ch).",
        "",
        "## Overall performance (mean ± sd across seeds)",
        "",
        "| Metric | Current (7ch) | Accel-only (3ch) | Delta |",
        "| --- | --- | --- | --- |",
        f"| Accuracy | {ms(acc7)} | {ms(acc3)} | {ms(acc7 - acc3)} |",
        f"| Macro F1 | {ms(macro7)} | {ms(macro3)} | {ms(macro7 - macro3)} |",
        "",
        "## Per-seed results",
        "",
        "| Seed | Acc (7ch) | Acc (3ch) | Macro F1 (7ch) | Macro F1 (3ch) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s, a7, a3, m7, m3 in seed_rows:
        lines.append(f"| {s} | {a7:.3f} | {a3:.3f} | {m7:.3f} | {m3:.3f} |")

    lines += [
        "",
        "## Per-class F1 (mean ± sd across seeds)",
        "",
        "| Class | Support (total) | F1 (7ch) | F1 (3ch) | Delta |",
        "| --- | --- | --- | --- | --- |",
    ]
    support = cm7.sum(axis=1)
    for c in range(n):
        lines.append(
            f"| {classes[c]} | {int(support[c])} | {ms(pcls7[:, c])} | {ms(pcls3[:, c])} "
            f"| {ms(pcls7[:, c] - pcls3[:, c])} |"
        )

    lines += [
        "",
        "## Confusion matrices (summed across seeds, rows = true, columns = predicted)",
        "",
        "### Current model (7 channels)",
        "",
        "```",
        confusion_str(cm7, classes),
        "```",
        "",
        "### Accel-only model (3 channels)",
        "",
        "```",
        confusion_str(cm3, classes),
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines))
    print(f"report -> {REPORT}")


if __name__ == "__main__":
    main()
