"""The reactor model's *given* building blocks — the parts the previous engineer
finished and that the notebook treats as a black box until it opens them.

The notebook (HW3, basin-temperature) asks the manager to FINISH an architecture.
The pieces that are already done live here:

  * `VisionEncoder` — the "do-not-touch" thermal-camera encoder. It's a small
    conv net; in the story its weights were trained elsewhere, so we must keep it
    **frozen**. (Here it's just deterministically random — freezing it is still a
    real, checkable requirement.)
  * `SpectroNet` — a generic, off-the-shelf 1-D signal module. Its own `forward`
    does *classification* (it ends in a head that outputs class logits), which is
    NOT what we need: we want the **features** before that head. The notebook
    extracts them by nesting/slicing the module — a core PyTorch skill.
  * data plumbing (`make_loaders`) and the regression train/eval loop
    (`run_epoch`) — the model itself is assembled by the student in the notebook.

Students are told not to read this file top to bottom; the notebook reveals the
relevant slice (via `basin_viz.show_source`) exactly when a task needs it.
"""
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# feature widths the two branches produce — handy named constants
THERMAL_FEATURES = 32
SIGNAL_FEATURES = 32


# ===========================================================================
#  The thermal-camera encoder — keep it FROZEN
# ===========================================================================
class VisionEncoder(nn.Module):
    """A small image encoder standing in for a pretrained thermal-feature extractor.

    Maps one batch of single-channel frames `(B, 1, H, W)` to a `(B, 32)` feature
    vector. The weights are seeded so the "pretrained" features are identical every
    run. In the story these weights are precious and must not change — the notebook
    keeps the module frozen during training.
    """

    def __init__(self, out_dim=THERMAL_FEATURES, seed=0):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(4)        # -> (B, 8, 4, 4)
        self.proj = nn.Linear(8 * 4 * 4, out_dim)
        # deterministic "pretrained" weights, independent of the global RNG
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.empty_like(p).uniform_(-0.3, 0.3, generator=g))

    def forward(self, frames):                     # frames: (B, 1, H, W)
        x = torch.relu(self.conv(frames))
        x = self.pool(x).flatten(1)
        return self.proj(x)                        # (B, out_dim)


# ===========================================================================
#  The off-the-shelf 1-D signal module (SpectroNet)
# ===========================================================================
class SpectroNet(nn.Module):
    """A generic 1-D convolutional signal network — the kind you'd `import` from a
    signal-processing library and drop in.

    Architecture, in order:
        stem        Conv1d(in -> 16, k=5)  + ReLU
        block1      Conv1d(16 -> 32, k=3)  + ReLU
        pool        global average over time  -> (B, 32)   <-- the FEATURES we want
        classifier  Linear(32 -> n_classes)   -> (B, n_classes)

    Its `forward` runs the whole stack, so it outputs **class logits** — it was
    built to label a signal's "regime", not to hand back features. We only want
    everything up to (and including) `pool`; the final `classifier` is in the way.
    The children are registered in order (stem → block1 → pool → classifier), so a
    module rebuilt from those children *minus the last one* stops exactly at the
    `(B, 32)` features — that's the task in the notebook.
    """

    def __init__(self, in_channels=4, n_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2), nn.ReLU())
        self.block1 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU())
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())   # -> (B, 32)
        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x):                          # x: (B, in_channels, T)
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool(x)
        return self.classifier(x)                  # (B, n_classes) — NOT what we need


# ===========================================================================
#  Data plumbing
# ===========================================================================
def make_loaders(bundle, batch_size=128, val_frac=0.2, seed=0):
    """Split the bundle into train/val DataLoaders of (X_sensor, X_thermal, y)."""
    n = len(bundle["y"])
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).numpy()
    n_val = int(n * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_sig = torch.as_tensor(bundle["X_sensor"], dtype=torch.float32)
    X_th = torch.as_tensor(bundle["X_thermal"], dtype=torch.float32)
    y = torch.as_tensor(bundle["y"], dtype=torch.float32)

    def subset(idx, shuffle):
        ds = TensorDataset(X_sig[idx], X_th[idx], y[idx])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return subset(train_idx, True), subset(val_idx, False)


# ===========================================================================
#  Regression train / eval loop
# ===========================================================================
def run_epoch(model, loader, optimizer=None, device="cpu"):
    """One pass over `loader`. Train if `optimizer` is given, else evaluate.

    Returns (mean_mse_loss, rmse). The target is standardized, so a model that
    always predicts the mean scores RMSE ≈ 1.0 — our honest baseline.
    """
    train = optimizer is not None
    model.train(train)
    loss_fn = nn.MSELoss()
    total_loss, total_se, total = 0.0, 0.0, 0

    for x_sig, x_th, y in loader:
        x_sig, x_th, y = x_sig.to(device), x_th.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            pred = model(x_sig, x_th)
            loss = loss_fn(pred, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_se += ((pred.detach() - y) ** 2).sum().item()
        total += y.size(0)

    return total_loss / total, (total_se / total) ** 0.5


def predict_all(model, loader, device="cpu"):
    """Run the model over a loader; return (predictions, targets) as 1-D arrays."""
    model.eval()
    preds, ys = [], []
    with torch.inference_mode():
        for x_sig, x_th, y in loader:
            p = model(x_sig.to(device), x_th.to(device))
            preds.append(p.cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(preds), np.concatenate(ys)


def n_trainable(model):
    """How many parameters will actually be updated (requires_grad=True)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
#  Device helpers
# ===========================================================================
def available_devices():
    devs = [("cpu", "CPU — always present")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devs.append((f"cuda:{i}", torch.cuda.get_device_name(i)))
    elif torch.backends.mps.is_available():
        devs.append(("mps", "Apple Metal GPU"))
    return devs


def pick_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
