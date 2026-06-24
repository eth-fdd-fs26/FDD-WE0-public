"""Part adapters.

Each ``part{N}_*`` module exposes ``run(solution)`` — a generator of
:class:`~backend.contracts.StepEvent`s that drives one subsystem of the day,
runs the HIDDEN grading checks, and yields an EXPLODED event the moment a check
fails. The checks live here (not in the solution) so they can't be gamed.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..contracts import EXPLODED, INFO, OK, RUNNING, StepEvent


def running(part: int, step_id: str, title: str, message: str = "",
            payload: Dict[str, Any] | None = None) -> StepEvent:
    return StepEvent(part, step_id, title, RUNNING, message, payload or {})


def ok(part: int, step_id: str, title: str, message: str = "",
       payload: Dict[str, Any] | None = None) -> StepEvent:
    return StepEvent(part, step_id, title, OK, message, payload or {})


def exploded(part: int, step_id: str, title: str, message: str,
             payload: Dict[str, Any] | None = None) -> StepEvent:
    return StepEvent(part, step_id, title, EXPLODED, message, payload or {})


def info(step_id: str, title: str, message: str = "",
         payload: Dict[str, Any] | None = None) -> StepEvent:
    return StepEvent(0, step_id, title, INFO, message, payload or {})


def downsample(values: List[float], k: int = 120) -> List[float]:
    """Evenly thin a long list down to ~k points for plotting."""
    arr = np.asarray(values, dtype=float)
    if len(arr) <= k:
        return [float(v) for v in arr]
    idx = np.linspace(0, len(arr) - 1, k).round().astype(int)
    return [float(v) for v in arr[idx]]
