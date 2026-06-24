"""Part 2 adapter — Basin-temperature predictor (an architecture exercise).

HW2 asks the manager to *finish a half-built multimodal model*: a 1-D signal
branch (SpectroNet with its classifier dropped) fused with a frozen thermal
vision encoder, through a residual MLP core and a regression head. The bugs the
exercise plants — a leftover classifier, a broken reshape, a dropped skip, a
flipped layer size, a missing head, an un-frozen encoder, an ignored branch —
are all things we can catch **without training and without any saved weights**.

So this adapter never trains. It builds the student's model from
``solution.build_model()`` and grades it *structurally*:

  * ``basin.arch``   — shape probes: the signal branch yields features (not
    logits), the residual core preserves its width, and the whole model ends in
    one temperature per sample (the regression head + correct fusion width).
  * ``basin.wiring`` — the vision encoder is frozen, the thermal branch actually
    influences the output (perturb it and watch), and a single forward/backward
    shows gradients reaching every trainable block (so no branch is dead).

The plant melts down at the first failed check. The hidden thresholds live here,
not in the solution.
"""
from __future__ import annotations

import os
import sys

# Reuse the real HW2 building blocks + data (new home first, legacy fallback).
for _rel in ("../../homework/helpers", "../../../exercises"):
    _cand = os.path.abspath(os.path.join(os.path.dirname(__file__), _rel))
    if os.path.exists(os.path.join(_cand, "basin_lab.py")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break
import basin_lab  # noqa: E402  (path injected above)

from . import exploded, ok, running  # noqa: E402

PART = 2
SIG = basin_lab.SIGNAL_FEATURES       # signal branch features (32)
TH = basin_lab.THERMAL_FEATURES       # thermal branch features (32)
FUSION = SIG + TH                     # fused width after the cat (64)
SENSITIVITY_MIN = 1e-5                 # a used branch must move the output by at least this


def _check(name, detail):
    return {"name": name, "ok": True, "detail": detail}


def _trainable(module):
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def run(solution):
    import torch

    # ---- 0. assemble the student's architecture ----------------------------
    yield running(PART, "basin.arch", "Inspecting the architecture",
                  "Building the backup predictor and tracing shapes through every block — the "
                  "signal branch, the thermal branch, the residual core and the regression head.")
    try:
        model = solution.build_model()
        model.eval()
    except Exception as exc:  # noqa: BLE001
        yield exploded(PART, "basin.arch", "Model won't assemble",
                       f"build_model() raised {type(exc).__name__}: {exc}")
        return

    for attr in ("signal_extractor", "encoder", "core", "head"):
        if not hasattr(model, attr):
            yield exploded(PART, "basin.arch", "Model is missing a block",
                           f"The assembled model has no `{attr}` — expected submodules "
                           "signal_extractor, encoder, core and head.")
            return

    checks = []

    # ---- 1. signal branch: features, not class logits ----------------------
    try:
        with torch.inference_mode():
            sig_out = model.signal_extractor(torch.zeros(2, 4, 64))
    except Exception as exc:  # noqa: BLE001
        yield exploded(PART, "basin.arch", "Signal branch won't run",
                       f"signal_extractor raised {type(exc).__name__}: {exc}")
        return
    if tuple(sig_out.shape) != (2, SIG):
        yield exploded(PART, "basin.arch", "Signal branch still ends in the classifier",
                       f"It outputs {tuple(sig_out.shape)} — expected (2, {SIG}) features. "
                       "Drop SpectroNet's final classifier so the branch stops at the features.")
        return
    checks.append(_check("Signal branch → 32 features", "SpectroNet classifier dropped"))

    # ---- 2. residual core preserves its width ------------------------------
    try:
        with torch.inference_mode():
            core_out = model.core(torch.zeros(2, FUSION))
    except Exception as exc:  # noqa: BLE001
        yield exploded(PART, "basin.arch", "MLP core won't run",
                       f"core raised {type(exc).__name__}: {exc} — the skip connection needs "
                       f"matching shapes (in and out both width {FUSION}).")
        return
    if tuple(core_out.shape) != (2, FUSION):
        yield exploded(PART, "basin.arch", "Residual skip can't line up",
                       f"core maps width {FUSION} → {tuple(core_out.shape)[1]}; a skip "
                       "connection needs output width == input width.")
        return
    checks.append(_check("Residual core preserves width", f"in/out width {FUSION}"))

    # ---- 3. the whole model ends in one temperature per sample -------------
    x_sig = torch.zeros(8, 4, 64)
    x_th = torch.zeros(8, 3, 16, 16, 1)
    try:
        with torch.inference_mode():
            out = model(x_sig, x_th)
    except Exception as exc:  # noqa: BLE001
        yield exploded(PART, "basin.arch", "Network won't run",
                       f"forward() raised {type(exc).__name__}: {exc} — a layer size is "
                       "probably flipped, or the fusion width is wrong.")
        return
    flat = out.reshape(out.shape[0], -1) if out.ndim > 1 else out.reshape(-1, 1)
    if flat.shape != (8, 1):
        yield exploded(PART, "basin.arch", "No regression head",
                       f"Output shape {tuple(out.shape)} — the model must end in a single "
                       "temperature per sample. Add the Linear(64, 1) regression head.",
                       {"output_shape": list(out.shape)})
        return
    checks.append(_check("Outputs one temperature / sample", "regression head present"))

    trainable = _trainable(model)
    frozen = int(sum(p.numel() for p in model.parameters() if not p.requires_grad))
    yield ok(PART, "basin.arch", "Architecture wired correctly",
             f"signal {SIG} + thermal {TH} → fuse {FUSION} → 1 °C · "
             f"{trainable:,} trainable / {frozen:,} frozen params.",
             {"checks": checks, "trainable_params": trainable, "frozen_params": frozen,
              "total_params": trainable + frozen, "signal_features": SIG,
              "thermal_features": TH, "fusion_dim": FUSION})

    # ---- 4. wiring: frozen encoder, live branches, gradients reach all -----
    yield running(PART, "basin.wiring", "Checking the wiring",
                  "Confirming the vision encoder stays frozen, that the thermal branch actually "
                  "feeds the prediction, and that one backward pass reaches every trainable block.")
    wiring = []

    enc_trainable = _trainable(model.encoder)
    if enc_trainable != 0:
        yield exploded(PART, "basin.wiring", "Vision encoder isn't frozen",
                       f"The encoder has {enc_trainable:,} trainable params — it's the "
                       "do-not-touch block and must stay frozen (requires_grad=False).",
                       {"checks": wiring + [{"name": "Vision encoder frozen", "ok": False,
                                             "detail": f"{enc_trainable:,} trainable params"}]})
        return
    wiring.append(_check("Vision encoder frozen", "0 trainable params"))

    # the thermal branch must change the prediction when its input changes
    torch.manual_seed(0)
    a_sig = torch.randn(8, 4, 64)
    a_th = torch.randn(8, 3, 16, 16, 1)
    b_th = a_th + 1.0
    with torch.inference_mode():
        out_a = model(a_sig, a_th)
        out_thermal = model(a_sig, b_th)
    thermal_delta = float((out_a - out_thermal).abs().max())
    if thermal_delta < SENSITIVITY_MIN:
        yield exploded(PART, "basin.wiring", "Thermal branch is ignored",
                       "Changing the thermal-camera input doesn't change the prediction — the "
                       "thermal branch isn't fused into forward (did you forget to cat it?).",
                       {"checks": wiring + [{"name": "Both branches feed the output", "ok": False,
                                             "detail": "thermal input has no effect"}]})
        return
    wiring.append(_check("Both branches feed the output",
                         f"thermal moves output by {thermal_delta:.2f}"))

    # one forward/backward: every trainable block must receive a gradient
    model.zero_grad(set_to_none=True)
    out = model(a_sig, a_th)
    loss = (out ** 2).mean()
    loss.backward()

    def _grad_norm(module):
        tot = 0.0
        for p in module.parameters():
            if p.requires_grad and p.grad is not None:
                tot += float(p.grad.detach().pow(2).sum())
        return tot ** 0.5

    for block_name, block in (("signal extractor", model.signal_extractor),
                              ("MLP core", model.core),
                              ("regression head", model.head)):
        if _grad_norm(block) <= 0.0:
            yield exploded(PART, "basin.wiring", "A block isn't wired into the output",
                           f"The {block_name} receives no gradient — it isn't connected to "
                           "forward()'s output, so training could never update it.",
                           {"checks": wiring + [{"name": "Gradients reach every block",
                                                 "ok": False, "detail": f"{block_name} is dead"}]})
            return
    wiring.append(_check("Gradients reach every block", "signal · core · head all live"))

    yield ok(PART, "basin.wiring", "Backup predictor is sound",
             "Encoder frozen, both branches live, every trainable block wired — the model is "
             "ready to train and watch the basin.",
             {"checks": wiring})
