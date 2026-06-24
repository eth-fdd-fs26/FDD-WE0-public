"""Part 2 — WRONG: a layer size is flipped (the intern's fusion-width mistake).

The fusion concatenates two 32-feature branches into 64, but the regression head
was hard-coded to expect 32. forward() throws a size-mismatch the moment a real
batch flows through it → "Network won't run".
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
from einops import rearrange, reduce

for _rel in ("../../../homework/helpers", "../../../../exercises"):
    _cand = os.path.abspath(os.path.join(os.path.dirname(__file__), _rel))
    if os.path.exists(os.path.join(_cand, "basin_lab.py")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break
import basin_lab  # noqa: E402


def thermal_forward(encoder, images):
    n_frames = images.shape[1]
    x = rearrange(images, "b f h w c -> (b f) c h w")
    feat = encoder(x)
    feat = rearrange(feat, "(b f) d -> b f d", f=n_frames)
    return reduce(feat, "b f d -> b d", "mean")


class MLPCore(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin1 = nn.Linear(d, d)
        self.lin2 = nn.Linear(d, d)

    def forward(self, x):
        h = torch.relu(self.lin1(x))
        h = self.lin2(h)
        return torch.relu(h + x)


class BasinNet(nn.Module):
    def __init__(self, signal_extractor, encoder):
        super().__init__()
        self.signal_extractor = signal_extractor
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        fusion_dim = basin_lab.SIGNAL_FEATURES + basin_lab.THERMAL_FEATURES   # 64
        self.core = MLPCore(fusion_dim)
        self.head = nn.Linear(basin_lab.SIGNAL_FEATURES, 1)   # ← should be Linear(64, 1)

    def forward(self, x_sig, x_th):
        s = self.signal_extractor(x_sig)
        t = thermal_forward(self.encoder, x_th)
        x = torch.cat([s, t], dim=1)            # (B, 64)
        return self.head(self.core(x)).squeeze(-1)   # head expects 32 → boom


def build_model():
    spectro = basin_lab.SpectroNet(in_channels=4)
    signal_extractor = nn.Sequential(*list(spectro.children())[:-1])
    return BasinNet(signal_extractor, basin_lab.VisionEncoder())


class BasinSolution:
    def build_model(self):
        return build_model()


def get_solution():
    return BasinSolution()
