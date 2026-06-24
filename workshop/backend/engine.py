"""The day engine.

``DayEngine.run()`` is an async generator that plays the shift one
:class:`StepEvent` at a time: shift start → pump → basin → waste → dawn. The
**first** failed check yields an EXPLODED event followed by a meltdown beat, and
the day stops there (instant-explode game model).

Pacing is deliberate: ``running`` beats pass quickly (something is happening),
while ``ok`` / ``info`` results **linger** so you can read what just happened.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .contracts import EXPLODED, INFO, OK, StepEvent
from .loader import REFERENCE, fallback_allowed, load_solution, solution_source
from .parts import exploded as _exploded
from .parts import info
from .parts import part1_pump, part2_basin, part3_waste

_PARTS = [
    (1, "🛠️ Coolant pump", part1_pump),
    (2, "🌡️ Basin temperature", part2_basin),
    (3, "♻️ Waste machines", part3_waste),
]

# A one-line briefing shown when each subsystem comes online (the "why").
PART_INTRO = {
    1: ("The coolant pump is the core's lifeline — if it seizes, the core overheats. "
        "Its logs are a mess, so we clean them, then predict a failure in the next hour."),
    2: ("The main basin must never boil. We forecast its temperature from the plant's "
        "sensors so the overheat alarm can fire *before* it's too late."),
    3: ("Tonight's waste must be handled by as few machines as possible, each tuned to its "
        "waste. We cluster the waste to choose how many machines to run, configure them, "
        "then route the night's arrivals."),
}


class DayEngine:
    def __init__(self, pace: float = 1.4, allow_fallback: bool | None = None) -> None:
        # base seconds between streamed events — paces the on-screen animation
        self.pace = pace
        # None → read the NCM_ALLOW_FALLBACK env (set by launch.py --allow-fallback).
        self.allow_fallback = fallback_allowed() if allow_fallback is None else allow_fallback

    def _dwell(self, ev: StepEvent) -> float:
        # let results sit longer than the brief "…working" beats
        if ev.status in (OK, INFO):
            return self.pace * 1.7
        if ev.status == EXPLODED:
            return self.pace * 1.2
        return self.pace * 0.85

    async def run(self) -> AsyncIterator[StepEvent]:
        start = info("day.start", "☀️ 06:00 — Shift start",
                     "You have the conn, Manager. Three subsystems must hold until dawn, "
                     "each powered by one of your homework solutions. Watch the log.")
        yield start
        await asyncio.sleep(self._dwell(start))

        for part, label, module in _PARTS:
            # Strict mode (default): a missing student file is a meltdown, not a
            # silent fall-back to the reference. --allow-fallback re-enables it.
            if solution_source(part) == REFERENCE and not self.allow_fallback:
                print(f"[NCM] Part {part} — {label}: STRICT — no "
                      f"solutions/solution_part{part}.py → MELTDOWN "
                      "(relaunch with --allow-fallback to run the reference).", flush=True)
                ev = _exploded(part, f"part{part}.source",
                               f"{label}: no solution file",
                               f"Strict mode — solutions/solution_part{part}.py is missing "
                               "and reference fall-back is OFF. Export this part's homework into "
                               f"workshop/solutions/, or relaunch with --allow-fallback.",
                               {"part": part, "source": "missing"})
                yield ev
                async for m in self._meltdown(part, label):
                    yield m
                return

            # Pick the student's solution if present, else the reference.
            try:
                solution, source = load_solution(part)
            except Exception as exc:  # noqa: BLE001
                ev = _exploded(part, f"part{part}.load", f"{label} — solution failed to load",
                               f"Could not load solution_part{part}: "
                               f"{type(exc).__name__}: {exc}", {"source": "student"})
                yield ev
                async for m in self._meltdown(part, label):
                    yield m
                return

            # A clear, standalone line stating exactly which code drives this part.
            if source == "student":
                src_detail = f"solutions/solution_part{part}.py"
                src_title = f"📄 {label}: running YOUR solution file"
            else:
                src_detail = f"backend/reference/solution_part{part}.py"
                src_title = f"📦 {label}: running the built-in REFERENCE"
            print(f"[NCM] Part {part} — {label}: source={source.upper()} ({src_detail})",
                  flush=True)
            source_note = StepEvent(part, f"part{part}.source", src_title, INFO,
                                    f"Source = {source.upper()} → {src_detail}",
                                    {"part": part, "source": source})
            yield source_note
            await asyncio.sleep(self._dwell(source_note))

            briefing = info(f"part{part}.start", f"{label} online",
                            PART_INTRO[part],
                            {"part": part, "source": source})
            yield briefing
            await asyncio.sleep(self._dwell(briefing))

            blew_up = False
            for ev in module.run(solution):
                ev.payload.setdefault("source", source)
                yield ev
                await asyncio.sleep(self._dwell(ev))
                if ev.status == EXPLODED:
                    blew_up = True
                    break
            if blew_up:
                async for m in self._meltdown(part, label):
                    yield m
                return

        yield info("day.victory", "🏁 06:00 — Dawn. You made it.",
                   "Every subsystem held through the night. The plant survives another day. ☢️✅")

    async def _meltdown(self, part: int, label: str) -> AsyncIterator[StepEvent]:
        await asyncio.sleep(self.pace)
        yield info("day.meltdown", "☢️ MELTDOWN",
                   f"{label} failed its safety check and the core breached. "
                   "Day over — review the report below.",
                   {"failed_part": part})


async def collect_events(pace: float = 0.0):
    """Run a whole day headlessly and return the list of events (for tests)."""
    return [ev async for ev in DayEngine(pace=pace).run()]
