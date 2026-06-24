"""Discover which solution to run for each part.

Priority: a student file dropped into ``workshop/solutions/solution_part{N}.py``
wins; otherwise we fall back to the bundled ``backend/reference`` solution.
Either way the module must expose ``get_solution()`` returning an object that
implements the matching protocol in :mod:`backend.contracts`.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from typing import Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.abspath(os.path.join(_HERE, "..", "solutions"))

STUDENT = "student"
REFERENCE = "reference"


def _student_path(part: int) -> str:
    return os.path.join(SOLUTIONS_DIR, f"solution_part{part}.py")


def fallback_allowed() -> bool:
    """Whether a missing student solution may fall back to the bundled reference.

    Off by default (strict): a part with no ``solutions/solution_part{N}.py``
    melts down. Set ``NCM_ALLOW_FALLBACK=1`` (the launcher's ``--allow-fallback``
    flag) to run the reference instead so the day plays through regardless.
    """
    return os.environ.get("NCM_ALLOW_FALLBACK", "0").strip().lower() in ("1", "true", "yes", "on")


def solution_source(part: int) -> str:
    """Return ``"student"`` if a drop-in file exists for this part, else ``"reference"``."""
    return STUDENT if os.path.isfile(_student_path(part)) else REFERENCE


def _load_student_module(part: int):
    path = _student_path(part)
    name = f"student_solution_part{part}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    # Let the student file `import solution_partN`-style siblings if needed.
    if SOLUTIONS_DIR not in os.sys.path:
        os.sys.path.insert(0, SOLUTIONS_DIR)
    spec.loader.exec_module(module)
    return module


def load_solution(part: int) -> Tuple[object, str]:
    """Return ``(solution_object, source)`` for the given part (1..3)."""
    source = solution_source(part)
    if source == STUDENT:
        module = _load_student_module(part)
    else:
        module = importlib.import_module(f".reference.solution_part{part}", __package__)
    if not hasattr(module, "get_solution"):
        raise AttributeError(
            f"solution_part{part} must define get_solution() returning the solution object")
    return module.get_solution(), source
