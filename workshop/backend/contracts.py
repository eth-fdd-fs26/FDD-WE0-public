"""The contract between the core engine and each homework solution.

Every part of the homework produces a Python module that, once dropped into
``workshop/solutions/`` as ``solution_part{N}.py``, exposes a single factory::

    def get_solution():
        '''Return an object implementing the Part-N protocol below.'''
        ...

The hidden grading tests live in ``backend/parts/part{N}_*.py`` — NOT here — so
a solution can never see (or fake) the checks it must pass.

The engine streams :class:`StepEvent`s to the front end over a WebSocket; each
event carries a small ``payload`` dict that the matching React panel renders.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Status vocabulary used on the wire.
RUNNING = "running"   # a step has started (front end shows the spinner / animation)
OK = "ok"             # a step passed its hidden checks
EXPLODED = "exploded"  # a step FAILED → the plant melts down and the day stops
INFO = "info"         # narrative beat (shift start, dawn, victory) — no pass/fail


@dataclass
class StepEvent:
    """One beat of the day, streamed to the front end as JSON."""

    part: int               # 0 = narrative, 1 = pump, 2 = basin, 3 = waste
    step_id: str            # stable id, e.g. "pump.clean"
    title: str              # human title shown on the panel
    status: str             # one of RUNNING / OK / EXPLODED / INFO
    message: str = ""       # one-line plain-language summary
    payload: Dict[str, Any] = field(default_factory=dict)  # data the panel draws

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Per-part protocols. A solution only has to implement its own part.

@runtime_checkable
class Part1Solution(Protocol):
    """Coolant-pump failure prediction (data cleaning + a classifier)."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a tidy frame: no NaNs, no duplicates, plausible sensor ranges."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return a 0/1 array (Failure within the next hour) — one per input row."""


@runtime_checkable
class Part2Solution(Protocol):
    """Basin-temperature predictor — a half-built multimodal PyTorch architecture
    the student must *finish* (HW2).

    Part 2 is an **architecture** exercise, so the engine grades it structurally
    (no training, no weights): it builds a fresh model and probes its shapes,
    checks the vision encoder is frozen, and runs a single forward/backward to
    confirm every trainable block is wired in. The solution therefore only has to
    hand back the *architecture* via ``build_model()``.
    """

    def build_model(self) -> Any:
        """Return a fresh (untrained) ``nn.Module`` assembling the predictor.

        The returned model must expose the submodules ``signal_extractor``,
        ``encoder``, ``core`` and ``head``, and define
        ``forward(x_sig, x_th)`` that returns one temperature per sample, where
        ``x_sig`` is ``(B, 4, 64)`` and ``x_th`` is ``(B, 3, 16, 16, 1)``.
        """


@runtime_checkable
class Part3Solution(Protocol):
    """Waste-machine clustering (repair the log, choose #machines, configure, route)."""

    def fill_gaps(self, history_df: pd.DataFrame, gappy_df: pd.DataFrame) -> pd.DataFrame:
        """Train on the complete ``history_df`` and return ``gappy_df`` with its
        missing ``quantity_kg`` and ``critical`` cells filled in (no NaNs left)."""

    def build_machine_plan(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return at least ``{"k", "prep", "km", "X", "data"}`` (see the reference impl)."""

    def route(self, plan: Dict[str, Any], new_df: pd.DataFrame) -> np.ndarray:
        """Return the machine id (0..k-1) each row of ``new_df`` is sent to."""


class SolutionError(Exception):
    """Raised by an adapter when a solution crashes or violates the contract.

    The engine turns this into an EXPLODED event rather than a 500.
    """
