"""1D CNN over the 25-sample event windows. Stratified 80/20 split."""
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn

DAT = Path(os.environ.get("FSED_DAT", Path(__file__).resolve().parents[1] / "Dat"))
EPOCHS, BATCH = 120, 64


def load():
    d = np.load(DAT / "Windows/windows.npz", allow_pickle=False)
    return d["X"], d["y"], list(d["classes"])


def run(seed=0, save=None):
    """Train one model. Returns (y_true, y_pred) on the held-out 20%."""
    X, y, classes = load()
    torch.manual_seed(seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    mu = Xtr.mean((0, 2), keepdims=True)                 # per-channel, train only
    sd = Xtr.std((0, 2), keepdims=True)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    model = nn.Sequential(
        nn.Conv1d(X.shape[1], 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
        nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        nn.Dropout(0.3), nn.Linear(128, len(classes)),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)

    for _ in range(EPOCHS):
        model.train()
        for i in torch.randperm(len(ytr_t)).split(BATCH):
            opt.zero_grad()
            loss_fn(model(Xtr_t[i]), ytr_t[i]).backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte)).argmax(1).numpy()
    if save:
        torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd, "classes": classes}, save)
    return yte, pred


if __name__ == "__main__":
    classes = load()[2]
    yte, pred = run(0, save=DAT / "Windows/cnn.pt")
    print(f"test {len(yte)}  (per class: {np.bincount(yte)})")
    print(classification_report(yte, pred, target_names=classes, digits=3, zero_division=0))
    print("confusion (rows = true):\n", confusion_matrix(yte, pred))
