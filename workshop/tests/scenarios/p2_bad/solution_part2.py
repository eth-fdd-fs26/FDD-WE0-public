"""Part 2 — BAD: the "do-not-touch" vision encoder was left trainable.

The shapes are all correct and the model runs, but the freeze was forgotten — the
encoder's weights would be updated during training. The wiring check fails at
"Vision encoder isn't frozen".
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
        # ← the freeze loop (requires_grad_(False)) was forgotten
        fusion_dim = basin_lab.SIGNAL_FEATURES + basin_lab.THERMAL_FEATURES
        self.core = MLPCore(fusion_dim)
        self.head = nn.Linear(fusion_dim, 1)

    def forward(self, x_sig, x_th):
        s = self.signal_extractor(x_sig)
        t = thermal_forward(self.encoder, x_th)
        x = torch.cat([s, t], dim=1)
        return self.head(self.core(x)).squeeze(-1)


def build_model():
    spectro = basin_lab.SpectroNet(in_channels=4)
    signal_extractor = nn.Sequential(*list(spectro.children())[:-1])
    return BasinNet(signal_extractor, basin_lab.VisionEncoder())


class BasinSolution:
    def build_model(self):
        return build_model()


def get_solution():
    return BasinSolution()
