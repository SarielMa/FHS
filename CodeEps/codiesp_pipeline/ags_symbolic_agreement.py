#!/usr/bin/env python3
"""Shared symbolic normalization and agreement for AGS experiments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_fintagging_grounding_baseline import (
    DIMENSIONS,
    SCRIPT_DIR,
    normalize_space,
    normalize_tag,
    raw_tag,
    tokenize,
)


DEFAULT_NORMALIZATION_MAP = SCRIPT_DIR / "ags_symbolic_normalization_map.yaml"
UNRESOLVED_VALUES = {
    "",
    "na",
    "n/a",
    "none",
    "null",
    "unknown",
    "unsupported",
    "unresolved",
    "not specified",
    "not applicable",
}
DIMENSION_ORDER = tuple(dimension.lower() for dimension in DIMENSIONS)
SYMBOLIC_DIMENSIONS = ("family", "role", "event", "qualifier", "scope", "temporal", "aggregation")
CONTROLLED_DIMENSIONS = {"family", "qualifier", "scope", "temporal", "aggregation"}
VERDICT_SUPPORT = "support"
VERDICT_CONTRADICT = "contradict"
VERDICT_UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class AgreementResult:
    score: float
    matched: int
    evaluated: int
    verdicts: list[dict[str, Any]]


def load_normalization_map(path: Path | str = DEFAULT_NORMALIZATION_MAP) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def map_version(path: Path | str = DEFAULT_NORMALIZATION_MAP) -> str:
    return str(load_normalization_map(path).get("version", "unknown"))


def is_unresolved(value: Any) -> bool:
    text = normalize_space(value).lower()
    return text in UNRESOLVED_VALUES


def normalize_dimension_name(value: Any) -> str:
    text = normalize_space(value).strip().lower()
    aliases = {
        "family": "family",
        "accounting_family": "family",
        "role": "role",
        "event": "event",
        "state": "event",
        "qualifier": "qualifier",
        "modifier": "qualifier",
        "aggregation": "aggregation",
        "scope": "scope",
        "dimensional_scope": "scope",
        "temporal": "temporal",
        "time": "temporal",
        "period": "temporal",
    }
    return aliases.get(text.replace(" ", "_"), text)


def canonical_hypothesis_dimensions(dimensions: dict[str, Any] | None) -> dict[str, Any]:
    source = dimensions or {}
    result: dict[str, Any] = {dimension: None for dimension in DIMENSION_ORDER}
    for key, value in source.items():
        dimension = normalize_dimension_name(key)
        if dimension in result:
            result[dimension] = None if is_unresolved(value) else normalize_space(value)
        elif dimension == "aggregation":
            result[dimension] = None if is_unresolved(value) else normalize_space(value)
    return result


def _phrase_tokens(value: Any) -> set[str]:
    return set(tokenize(normalize_space(value)))


def _category_matches(text: str, text_tokens: set[str], phrase: str) -> bool:
    phrase_text = normalize_space(phrase).lower()
    if not phrase_text:
        return False
    phrase_tokens = _phrase_tokens(phrase_text)
    return phrase_text in text or bool(phrase_tokens and phrase_tokens <= text_tokens)


def normalize_dimension_value(
    dimension: str,
    value: Any,
    normalization_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dimension = normalize_dimension_name(dimension)
    raw = None if is_unresolved(value) else normalize_space(value)
    if raw is None:
        return {"raw": normalize_space(value), "resolved": False, "categories": [], "tokens": []}

    normalization_map = normalization_map or load_normalization_map()
    vocab = normalization_map.get("dimensions", {}).get(dimension, {}) or {}
    text = raw.lower()
    text_tokens = _phrase_tokens(text)
    categories = [
        category
        for category, phrases in vocab.items()
        if any(_category_matches(text, text_tokens, phrase) for phrase in phrases)
    ]
    return {
        "raw": raw,
        "resolved": True,
        "categories": sorted(set(categories)),
        "tokens": sorted(text_tokens),
    }


def normalize_hypothesis_dimensions(
    dimensions: dict[str, Any] | None,
    normalization_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    canonical = canonical_hypothesis_dimensions(dimensions)
    return {
        dimension: normalize_dimension_value(dimension, canonical.get(dimension), normalization_map)
        for dimension in canonical
    }


def candidate_text(candidate: dict[str, Any]) -> str:
    parts = [
        raw_tag(candidate.get("tag", "")),
        normalize_space(candidate.get("standard_label", "")),
        normalize_space(candidate.get("documentation", "")),
        normalize_space(candidate.get("retrieval_text", "")),
        normalize_space(candidate.get("type", "")),
        normalize_space(candidate.get("datatype", "")),
    ]
    return normalize_space(" ".join(part for part in parts if part))


def _metadata_value(candidate: dict[str, Any], aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in candidate and normalize_space(candidate.get(alias)):
            return normalize_space(candidate.get(alias))
    return None


def parse_candidate_symbolic_profile(
    candidate: dict[str, Any],
    normalization_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalization_map = normalization_map or load_normalization_map()
    text = candidate_text(candidate)
    text_lc = text.lower()
    tokens = sorted(set(tokenize(text)))
    token_set = set(tokens)
    dimension_values: dict[str, dict[str, Any]] = {}
    for dimension, vocab in normalization_map.get("dimensions", {}).items():
        categories = [
            category
            for category, phrases in (vocab or {}).items()
            if any(_category_matches(text_lc, token_set, phrase) for phrase in phrases)
        ]
        dimension_values[dimension] = {
            "categories": sorted(set(categories)),
            "tokens": tokens,
        }

    aliases = normalization_map.get("metadata_aliases", {})
    metadata = {
        canonical: _metadata_value(candidate, list(alias_values))
        for canonical, alias_values in aliases.items()
    }
    metadata = {key: value for key, value in metadata.items() if value}
    return {
        "tag": normalize_tag(candidate.get("tag", "")),
        "label": normalize_space(candidate.get("standard_label", "")),
        "tokens": tokens,
        "dimensions": dimension_values,
        "metadata": metadata,
    }


def _token_overlap_score(hyp_tokens: list[str], candidate_tokens: list[str]) -> float | None:
    hyp = set(hyp_tokens)
    cand = set(candidate_tokens)
    if not hyp:
        return None
    if not cand:
        return None
    return len(hyp & cand) / len(hyp)


def dimension_agreement_verdict(
    candidate: dict[str, Any],
    dimension: str,
    value: Any,
    normalization_map: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalization_map = normalization_map or load_normalization_map()
    dimension = normalize_dimension_name(dimension)
    normalized = normalize_dimension_value(dimension, value, normalization_map)
    if not normalized["resolved"]:
        return {
            "dimension": dimension,
            "raw_hypothesis_value": normalize_space(value),
            "normalized_hypothesis_value": normalized,
            "candidate_values": {},
            "verdict": VERDICT_UNRESOLVED,
            "matched": None,
            "source_layer": "symbolic",
            "confidence": 0.0,
            "reason": "hypothesis_unresolved",
        }

    profile = profile or parse_candidate_symbolic_profile(candidate, normalization_map)
    candidate_dim = profile["dimensions"].get(dimension, {})
    candidate_categories = set(candidate_dim.get("categories", []))
    hypothesis_categories = set(normalized.get("categories", []))
    if hypothesis_categories or candidate_categories:
        if not candidate_categories:
            matched = False
            confidence = 0.35
            reason = "candidate_missing_controlled_category"
        else:
            matched = bool(hypothesis_categories & candidate_categories)
            confidence = 0.85
            reason = "controlled_category_intersection"
    else:
        overlap = _token_overlap_score(normalized.get("tokens", []), profile.get("tokens", []))
        if overlap is None:
            matched = None
            confidence = 0.0
            reason = "no_comparable_tokens"
        else:
            matched = overlap >= 0.5
            confidence = min(0.75, max(0.2, overlap))
            reason = "token_overlap"

    if matched is None:
        verdict = VERDICT_UNRESOLVED
    else:
        verdict = VERDICT_SUPPORT if matched else VERDICT_CONTRADICT
    return {
        "dimension": dimension,
        "raw_hypothesis_value": normalized["raw"],
        "normalized_hypothesis_value": normalized,
        "candidate_values": {
            "categories": sorted(candidate_categories),
            "tokens": profile.get("tokens", []),
            "metadata": profile.get("metadata", {}),
            "label": profile.get("label", ""),
        },
        "verdict": verdict,
        "matched": matched,
        "source_layer": "symbolic",
        "confidence": round(float(confidence), 6),
        "reason": reason,
    }


def agree(
    candidate: dict[str, Any],
    hypothesis_dimensions: dict[str, Any],
    normalization_map: dict[str, Any] | None = None,
) -> AgreementResult:
    normalization_map = normalization_map or load_normalization_map()
    profile = parse_candidate_symbolic_profile(candidate, normalization_map)
    return agree_with_profile(candidate, hypothesis_dimensions, normalization_map, profile)


def agree_with_profile(
    candidate: dict[str, Any],
    hypothesis_dimensions: dict[str, Any],
    normalization_map: dict[str, Any],
    profile: dict[str, Any],
) -> AgreementResult:
    canonical = canonical_hypothesis_dimensions(hypothesis_dimensions)
    verdicts = [
        dimension_agreement_verdict(candidate, dimension, value, normalization_map, profile)
        for dimension, value in canonical.items()
        if not is_unresolved(value)
    ]
    evaluated = sum(1 for verdict in verdicts if verdict["matched"] is not None)
    matched = sum(1 for verdict in verdicts if verdict["matched"] is True)
    score = matched / evaluated if evaluated else 0.0
    return AgreementResult(score=round(score, 6), matched=matched, evaluated=evaluated, verdicts=verdicts)


def mean_agreement(
    candidates: list[dict[str, Any]],
    hypothesis_dimensions: dict[str, Any],
    top_m: int = 10,
    normalization_map: dict[str, Any] | None = None,
) -> float:
    selected = candidates[:top_m]
    if not selected:
        return 0.0
    scores = [agree(candidate, hypothesis_dimensions, normalization_map).score for candidate in selected]
    return round(sum(scores) / len(scores), 6)


def consensus_agreement(
    candidate: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    normalization_map: dict[str, Any] | None = None,
) -> float:
    if not hypotheses:
        return 0.0
    normalization_map = normalization_map or load_normalization_map()
    profile = parse_candidate_symbolic_profile(candidate, normalization_map)
    scores = [
        agree_with_profile(candidate, hypothesis.get("dimensions", hypothesis), normalization_map, profile).score
        for hypothesis in hypotheses
    ]
    return round(sum(scores) / len(scores), 6)


def symbolic_feedback_from_candidates(
    hypothesis_dimensions: dict[str, Any],
    candidates: list[dict[str, Any]],
    top_m: int = 10,
    normalization_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalization_map = normalization_map or load_normalization_map()
    selected = candidates[:top_m]
    canonical = canonical_hypothesis_dimensions(hypothesis_dimensions)
    verdict_records: list[dict[str, Any]] = []
    supported: list[str] = []
    contradicted: list[str] = []
    unresolved: list[str] = []

    for dimension, value in canonical.items():
        if is_unresolved(value):
            unresolved.append(dimension.upper())
            verdict_records.append(
                {
                    "dimension": dimension.upper(),
                    "verdict": VERDICT_UNRESOLVED,
                    "source_layer": "symbolic",
                    "confidence": 0.0,
                    "candidate_support_fraction": 0.0,
                    "reason": "hypothesis_unresolved",
                }
            )
            continue

        candidate_verdicts = [
            dimension_agreement_verdict(candidate, dimension, value, normalization_map)
            for candidate in selected
        ]
        comparable = [verdict for verdict in candidate_verdicts if verdict["matched"] is not None]
        if not comparable:
            unresolved.append(dimension.upper())
            support_fraction = 0.0
            confidence = 0.0
            verdict_name = VERDICT_UNRESOLVED
        else:
            support_fraction = sum(verdict["matched"] is True for verdict in comparable) / len(comparable)
            if support_fraction >= 0.6:
                supported.append(dimension.upper())
                verdict_name = VERDICT_SUPPORT
            elif support_fraction <= 0.25:
                contradicted.append(dimension.upper())
                verdict_name = VERDICT_CONTRADICT
            else:
                unresolved.append(dimension.upper())
                verdict_name = VERDICT_UNRESOLVED
            confidence = abs(support_fraction - 0.5) * 2.0

        verdict_records.append(
            {
                "dimension": dimension.upper(),
                "verdict": verdict_name,
                "source_layer": "symbolic",
                "confidence": round(confidence, 6),
                "candidate_support_fraction": round(support_fraction, 6),
                "candidate_verdicts": candidate_verdicts,
            }
        )

    structural_dims = {"FAMILY", "ROLE", "EVENT"}
    structural_contradictions = [dim for dim in contradicted if dim in structural_dims]
    mismatch_flag = bool(structural_contradictions and len(structural_contradictions) >= 2)
    return {
        "supported_dimensions": supported,
        "contradicted_dimensions": contradicted,
        "unresolved_dimensions": unresolved,
        "dimension_verdicts": verdict_records,
        "structural_mismatch": {
            "is_mismatch": mismatch_flag,
            "reason": "symbolic_contradictions=" + ",".join(structural_contradictions)
            if mismatch_flag
            else "",
        },
        "source_layer": "symbolic",
        "top_m": top_m,
    }


def merge_feedback_layers(symbolic_feedback: dict[str, Any], llm_feedback: dict[str, Any]) -> dict[str, Any]:
    verdict_by_dim: dict[str, dict[str, Any]] = {}
    for verdict in symbolic_feedback.get("dimension_verdicts", []):
        verdict_by_dim[str(verdict.get("dimension", "")).upper()] = dict(verdict)

    for field, verdict_name in (
        ("supported_dimensions", VERDICT_SUPPORT),
        ("contradicted_dimensions", VERDICT_CONTRADICT),
        ("unresolved_dimensions", VERDICT_UNRESOLVED),
    ):
        for raw_dim in llm_feedback.get(field, []):
            dim = normalize_dimension_name(raw_dim).upper()
            existing = verdict_by_dim.get(dim)
            if existing is None or existing.get("verdict") == VERDICT_UNRESOLVED:
                verdict_by_dim[dim] = {
                    "dimension": dim,
                    "verdict": verdict_name,
                    "source_layer": "llm",
                    "confidence": 0.6,
                }
            elif existing.get("verdict") != verdict_name and verdict_name != VERDICT_UNRESOLVED:
                existing["llm_disagrees"] = True
                existing["llm_verdict"] = verdict_name
                existing["confidence"] = round(max(0.1, float(existing.get("confidence", 0.5)) - 0.2), 6)

    supported = sorted(dim for dim, verdict in verdict_by_dim.items() if verdict.get("verdict") == VERDICT_SUPPORT)
    contradicted = sorted(dim for dim, verdict in verdict_by_dim.items() if verdict.get("verdict") == VERDICT_CONTRADICT)
    unresolved = sorted(dim for dim in {key.upper() for key in DIMENSION_ORDER} if dim not in set(supported) | set(contradicted))
    llm_mismatch = llm_feedback.get("structural_mismatch", {}) if isinstance(llm_feedback.get("structural_mismatch"), dict) else {}
    symbolic_mismatch = symbolic_feedback.get("structural_mismatch", {})
    return {
        "supported_dimensions": supported,
        "contradicted_dimensions": contradicted,
        "unresolved_dimensions": unresolved,
        "dimension_verdicts": [verdict_by_dim[key] for key in sorted(verdict_by_dim)],
        "structural_mismatch": {
            "is_mismatch": bool(symbolic_mismatch.get("is_mismatch") or llm_mismatch.get("is_mismatch")),
            "reason": normalize_space(
                " ".join(
                    str(reason)
                    for reason in (
                        symbolic_mismatch.get("reason"),
                        llm_mismatch.get("reason"),
                    )
                    if reason
                )
            ),
        },
        "source_layer": "symbolic+llm",
    }


def utility_from_rank(rank: int | None) -> float:
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / math.log2(rank + 1.0)
