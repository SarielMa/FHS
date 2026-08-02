#!/usr/bin/env python3
"""One shared counter for the generation token budget, in a module with a single identity.

WHY ITS OWN FILE
    The counter first lived in run_fintagging_grounding_baseline.py. That file runs as `__main__`,
    while ags_frozen_grounding.py imports it by name -- so Python held TWO copies of the module and
    two copies of the counter. main() armed the `__main__` copy; the frozen family's generation
    calls incremented the imported copy. The gate then reported `generation_calls: 0` for a run that
    had generated normally, i.e. the instrumentation was blind exactly where it mattered.

    A module imported only by name has one identity, so both paths share this state.

WHAT IT IS FOR
    Truncation is invisible in the metrics that already exist: a truncated STRUCTURED output fails
    to parse and shows up in the parse rate, but a truncated FREE-TEXT query parses fine and is
    silently shortened. Counting calls that reach the cap is the only way to see both.
"""

from __future__ import annotations

from typing import Any

_STATE: dict[str, int] = {"calls": 0, "at_cap": 0, "cap": 0}


def arm(cap: int) -> None:
    """Record the cap this run was given. Call once, before any generation."""
    _STATE["cap"] = int(cap)


def note(completion_tokens: int) -> None:
    """Count one generation call, and whether it reached the cap."""
    cap = _STATE["cap"]
    if not cap:
        return
    _STATE["calls"] += 1
    if completion_tokens >= cap - 1:
        _STATE["at_cap"] += 1


def report() -> dict[str, Any]:
    calls = _STATE["calls"]
    return {
        "generation_calls": calls,
        "calls_at_token_cap": _STATE["at_cap"],
        "fraction_at_token_cap": (_STATE["at_cap"] / calls) if calls else None,
        "token_cap": _STATE["cap"] or None,
    }
