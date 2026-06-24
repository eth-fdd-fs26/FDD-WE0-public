"""The engineering team's training code — imported, not shown, until we open it.

This module is the "black box" the participants inherit. The story: a previous
notebook trained a churn model with PyTorch; now the team is stuck and the
manager (you) opens the hood. The code here intentionally contains the realistic
mistakes the notebook diagnoses one by one:

  * the model & batches are never moved to the GPU (`.to(device)` missing);
  * a `very_costly_operation` runs *inside* the Dataset, starving the GPU;
  * mixed precision is enabled the wrong way (fp16 cast, no GradScaler);
  * checkpoints save only the model weights (optimizer / epoch / scaler lost);
  * a wrong `.view()` on the embedding stack silently scrambles the batch.

Each defect is behind a flag so the notebook can flip from "broken" to "fixed"
and show the difference. Participants are told NOT to read this file top-to-bottom;
the notebook reveals the relevant slice (via `inspect.getsource`) at each step.
"""
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ===========================================================================
#  The Dataset the engineers wrote
# ===========================================================================
def _very_costly_operation(row_num, row_cat):
    """A stand-in for "expensive preprocessing the team left in the data path".

    Pretend this is some heavy featurization (a giant python loop, a per-row
    similarity search, a regex parse…). It returns the inputs unchanged — the
    point is only that it BURNS TIME on the CPU for every single sample, every
    epoch.
    """
    acc = 0.0
    for _ in range(4000):            # pointless busy-work ~ a costly transform
        acc += np.sin(acc + 1.234) ** 2
    return row_num, row_cat


class ChurnDataset(Dataset):
    """Wraps the churn arrays as a PyTorch Dataset.

    `heavy=True` reproduces the team's mistake: the costly op runs per sample.
    `heavy=False` is the fixed version (preprocessing done once, up front).
    """

    def __init__(self, X_num, X_cat, y, heavy=True):
        self.X_num = torch.as_tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.as_tensor(X_cat, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.heavy = heavy

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x_num = self.X_num[i]
        x_cat = self.X_cat[i]
        if self.heavy:                       # 🐌 the bottleneck the notebook finds
            x_num, x_cat = _very_costly_operation(x_num, x_cat)
        return x_num, x_cat, self.y[i]


def make_loaders(bundle, batch_size=256, val_frac=0.2, heavy=True, seed=0):
    """Split the bundle into train/val DataLoaders (val is never `heavy`)."""
    n = len(bundle["y"])
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).numpy()
    n_val = int(n * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def subset(idx, heavy_flag):
        return ChurnDataset(bundle["X_num"][idx], bundle["X_cat"][idx],
                            bundle["y"][idx], heavy=heavy_flag)

    train_ds = subset(train_idx, heavy)
    val_ds = subset(val_idx, False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ===========================================================================
#  The model the engineers wrote
# ===========================================================================
def flatten_embeddings(embs, batch_size):
    """The team's step that flattens the per-column embeddings into one feature
    vector per row, before the MLP head.

    `embs` is a list with one (batch, emb_dim) tensor per categorical column. We
    stack them into a single tensor and reshape to (batch, n_cat * emb_dim).
    """
    stacked = torch.stack(embs, dim=0)                  # one tensor per column
    n_cat, _, emb_dim = stacked.shape
    return stacked.view(batch_size, n_cat * emb_dim)    # flatten to (batch, features)


class ChurnModel(nn.Module):
    """Embeds the categorical columns, concatenates the numeric ones, MLP head.

    `buggy=True` keeps the team's reshape mistake: the per-column embeddings are
    stacked on the WRONG axis and then flattened with `.view(batch, -1)`. The
    shapes line up (so nothing crashes), but each row of the flattened tensor
    mixes embeddings from *different customers* — the labels no longer match the
    features, so the loss can never really go down. `buggy=False` flattens with
    einops, which makes the axis order explicit and the bug impossible to miss.
    """

    def __init__(self, n_numeric, cat_cardinalities, emb_dim=8, hidden=64,
                 buggy=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_cat = len(cat_cardinalities)
        self.buggy = buggy
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, emb_dim) for card in cat_cardinalities])
        in_dim = n_numeric + self.n_cat * emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def _flatten_embeddings(self, x_cat):
        batch = x_cat.shape[0]
        embs = [emb(x_cat[:, j]) for j, emb in enumerate(self.embeddings)]
        if self.buggy:
            return flatten_embeddings(embs, batch)      # the team's version
        # fixed: rearrange names the axes, so batch stays batch.
        from einops import rearrange
        stacked = torch.stack(embs, dim=1)          # (batch, n_cat, emb_dim)
        return rearrange(stacked, "b n d -> b (n d)")

    def forward(self, x_num, x_cat):
        e = self._flatten_embeddings(x_cat)
        x = torch.cat([x_num, e], dim=1)
        return self.net(x)


# ===========================================================================
#  Training / evaluation utilities
# ===========================================================================
def run_epoch(model, loader, optimizer=None, device="cpu",
              use_amp=False, amp_dtype=torch.float16, scaler=None):
    """One pass over `loader`. Train if `optimizer` is given, else evaluate.

    Returns (mean_loss, accuracy). When `use_amp` is True the forward runs under
    autocast in `amp_dtype`; `scaler` (a torch.cuda.amp.GradScaler) is used only
    if provided — the notebook shows what happens with and without it.
    """
    train = optimizer is not None
    model.train(train)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0

    for x_num, x_cat, y in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.split(":")[0],
                                dtype=amp_dtype, enabled=use_amp):
                logits = model(x_num, x_cat)
                loss = loss_fn(logits, y)

            if train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, total_correct / total


def time_epoch(model, loader, optimizer, device="cpu", **kw):
    """Run one training epoch and return its wall-clock seconds (for throughput)."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_epoch(model, loader, optimizer=optimizer, device=device, **kw)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter() - t0


# ===========================================================================
#  Checkpointing
# ===========================================================================
def save_checkpoint_broken(path, model):
    """What the team did: save only the weights. Reloading loses the optimizer
    state, the epoch counter, the AMP scaler and the RNG — so 'resume' silently
    restarts those, and the loss jumps."""
    torch.save(model.state_dict(), path)


def save_checkpoint(path, model, optimizer, epoch, scaler=None, extra=None):
    """The full picture: everything that carries state across epochs."""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, map_location="cpu"):
    """Restore everything `save_checkpoint` wrote. Returns the epoch to resume at."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if ckpt.get("torch_rng") is not None:
        # `map_location` may have moved the RNG tensor onto the GPU; set_rng_state
        # needs a CPU uint8 (ByteTensor), so force it back.
        torch.set_rng_state(ckpt["torch_rng"].to("cpu", torch.uint8))
    if ckpt.get("numpy_rng") is not None:
        np.random.set_state(ckpt["numpy_rng"])
    return ckpt.get("epoch", 0)


# ===========================================================================
#  Device helpers
# ===========================================================================
def available_devices():
    """List the devices PyTorch can see, with a short human label each."""
    devs = [("cpu", "CPU — always present; treated as ~infinite RAM")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            devs.append((f"cuda:{i}", f"{name} — {vram:.0f} GB VRAM"))
    elif torch.backends.mps.is_available():
        devs.append(("mps", "Apple Metal GPU"))
    return devs


def pick_device():
    """The device we'd actually train on: first GPU if any, else CPU."""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
