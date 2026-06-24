"""Reference solution — Part 2: Basin-temperature predictor architecture (HW2).

The HW2 homework is about *finishing a half-built multimodal architecture*:
fuse a 1-D **signal branch** (`SpectroNet` with its classifier sliced off) with a
**frozen thermal vision encoder**, push the concatenated features through a
**residual MLP core**, and end in a **regression head**. This file is the
correct, un-buggy assembly those fixes converge to.

Part 2 is graded **structurally** — no training, no weights. The engine builds
the model returned by ``build_model()`` and checks shapes, the frozen encoder,
and that every trainable block is wired into ``forward`` (a single
forward/backward). So this module ships only the architecture; ``build_model()``
returns a fresh, untrained model.

The real course helpers (``basin_lab.py``) are reused as-is for the two given
building blocks (``SpectroNet``, ``VisionEncoder``) and the feature-width
constants.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
from einops import rearrange, reduce

# Make the course's basin helper importable (new home first, legacy fallback).
for _rel in ("../../homework/helpers", "../../../exercises"):
    _cand = os.path.abspath(os.path.join(os.path.dirname(__file__), _rel))
    if os.path.exists(os.path.join(_cand, "basin_lab.py")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break
import basin_lab  # noqa: E402  (path injected above)


def thermal_forward(encoder, images):
    """`(B, frames, H, W, 1)` channels-last → encode every frame → mean over frames → `(B, 32)`."""
    n_frames = images.shape[1]
    x = rearrange(images, "b f h w c -> (b f) c h w")   # fold frames into batch, channels first
    feat = encoder(x)                                   # (B*frames, 32)
    feat = rearrange(feat, "(b f) d -> b f d", f=n_frames)
    return reduce(feat, "b f d -> b d", "mean")         # one vector per sample


class MLPCore(nn.Module):
    """Residual block: two width-preserving Linears + a skip connection."""

    def __init__(self, d):
        super().__init__()
        self.lin1 = nn.Linear(d, d)
        self.lin2 = nn.Linear(d, d)

    def forward(self, x):
        h = torch.relu(self.lin1(x))
        h = self.lin2(h)
        return torch.relu(h + x)                        # skip: output width == input width


class BasinNet(nn.Module):
    """Signal branch + frozen thermal encoder → fuse → residual core → regression head."""

    def __init__(self, signal_extractor, encoder):
        super().__init__()
        self.signal_extractor = signal_extractor
        self.encoder = encoder

        # The vision encoder is "pretrained, do-not-touch" — keep it frozen.
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        fusion_dim = basin_lab.SIGNAL_FEATURES + basin_lab.THERMAL_FEATURES   # 32 + 32 = 64
        self.core = MLPCore(fusion_dim)
        self.head = nn.Linear(fusion_dim, 1)            # regression head → one temperature

    def forward(self, x_sig, x_th):
        s = self.signal_extractor(x_sig)                # (B, 32)
        t = thermal_forward(self.encoder, x_th)         # (B, 32)
        x = torch.cat([s, t], dim=1)                    # (B, 64)
        return self.head(self.core(x)).squeeze(-1)      # (B,) one temperature per sample


def build_model():
    """Assemble a fresh BasinNet from the given building blocks (untrained)."""
    spectro = basin_lab.SpectroNet(in_channels=4)
    signal_extractor = nn.Sequential(*list(spectro.children())[:-1])   # drop the classifier
    encoder = basin_lab.VisionEncoder()
    return BasinNet(signal_extractor, encoder)


class BasinSolution:
    def build_model(self):
        return build_model()


def get_solution() -> BasinSolution:
    return BasinSolution()
