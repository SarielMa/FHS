#!/usr/bin/env python3
"""Minimal AGS rendering helpers vendored for the CodiEsp transfer experiment."""

from __future__ import annotations

from typing import Any

from run_fintagging_grounding_baseline import DIMENSIONS, normalize_space, tokenize


def dimensions_lower(dimensions: dict[str, Any]) -> dict[str, Any]:
    return {
        dimension.lower(): None
        if normalize_space(dimensions.get(dimension, "")).lower() == "unresolved"
        else normalize_space(dimensions.get(dimension, ""))
        for dimension in DIMENSIONS
    }


def render_definition(hypothesis: dict[str, Any]) -> str:
    query = normalize_space(hypothesis.get("retrieval_query", ""))
    if query:
        return query
    dims = dimensions_lower(hypothesis.get("dimensions", {}))
    resolved = [str(value) for value in dims.values() if value]
    return normalize_space(" ".join(resolved))


def render_label(hypothesis: dict[str, Any]) -> str:
    dims = dimensions_lower(hypothesis.get("dimensions", {}))
    values = [dims.get(dimension) for dimension in ("family", "role", "event", "qualifier", "scope", "temporal")]
    tokens: list[str] = []
    for value in values:
        if value:
            tokens.extend(tokenize(value))
    return normalize_space(" ".join(tokens))
