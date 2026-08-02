#!/usr/bin/env python3
"""One scoring path for every AGS grounding configuration.

Both the configuration ablation and the stage decomposition call
`score_configuration`, so a stage row and a config row with the same flags are
the identical computation and cannot diverge.

Flags:
    rendering        def | lab | dual
    hypotheses_used  1 | 2 | 3
    aggregation      none | rrf | selection | selection_union | oracle
    score_variant    sum | mean
    rerank_beta      0.0 | 0.05

`sum` reproduces the RRF the pipeline implements: scores are summed across the
fused lists, so the retrieval-score scale grows with the number of lists. `mean`
divides by the number of lists, putting every configuration on one scale so
rerank_beta carries the same weight everywhere. They are identical when only one
list is fused.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from ags_symbolic_agreement import (
    agree_with_profile,
    mean_agreement,
    parse_candidate_symbolic_profile,
)
from run_fintagging_grounding_baseline import Example, first_gold_rank, normalize_tag


AGGREGATIONS = ("none", "rrf", "selection", "selection_union", "oracle")
SCORE_VARIANTS = ("sum", "mean")
SCORE_SCALINGS = ("raw", "range_normalized")
RENDERINGS = ("def", "lab", "dual")


@dataclass
class FactContext:
    """Everything about one fact that scoring needs, parsed once and reused."""

    example: Example
    records: list[dict[str, Any]]
    hypotheses: list[dict[str, Any]]
    normalization_map: dict[str, Any]
    agreement_top_m: int
    candidate_lists: list[list[str]] = field(init=False)
    candidate_by_tag: dict[str, dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self.candidate_lists = [
            [normalize_tag(tag) for tag in record["candidate_ids"]] for record in self.records
        ]
        self.candidate_by_tag = {}
        for record in self.records:
            for candidate in record["candidates"]:
                self.candidate_by_tag.setdefault(normalize_tag(candidate["tag"]), candidate)

    @cached_property
    def agreement(self) -> dict[str, list[float]]:
        """Per-candidate agreement score against each hypothesis, computed on demand."""
        profiles = {
            tag: parse_candidate_symbolic_profile(candidate, self.normalization_map)
            for tag, candidate in self.candidate_by_tag.items()
        }
        return {
            tag: [
                agree_with_profile(
                    self.candidate_by_tag[tag],
                    hypothesis.get("dimensions", hypothesis),
                    self.normalization_map,
                    profiles[tag],
                ).score
                for hypothesis in self.hypotheses
            ]
            for tag in self.candidate_by_tag
        }

    @cached_property
    def selection_scores(self) -> list[float]:
        return [
            mean_agreement(
                record["candidates"],
                self.hypotheses[idx].get("dimensions", self.hypotheses[idx]),
                self.agreement_top_m,
                self.normalization_map,
            )
            for idx, record in enumerate(self.records)
        ]

    @cached_property
    def selected_idx_by_j(self) -> dict[int, int]:
        return {
            j: max(range(j), key=lambda idx: (self.selection_scores[idx], -idx))
            for j in range(1, len(self.records) + 1)
        }

    def consensus_over(self, hypotheses_used: int) -> dict[str, float]:
        return {
            tag: round(sum(scores[:hypotheses_used]) / hypotheses_used, 6)
            for tag, scores in self.agreement.items()
        }


@dataclass
class ConfigurationResult:
    gold_rank: int | None
    gold_rank_before_rerank: int | None
    selected_hypothesis_idx: int | None
    lists_fused: int


def rrf_scores(
    candidate_lists: list[list[str]],
    kappa: float,
) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    first_list: dict[str, int] = {}
    for list_idx, tags in enumerate(candidate_lists):
        for rank, tag in enumerate(tags, start=1):
            scores[tag] += 1.0 / (kappa + rank)
            if tag not in best_rank or rank < best_rank[tag]:
                best_rank[tag] = rank
            first_list.setdefault(tag, list_idx)
    return scores, best_rank, first_list


def order_from_scores(
    scores: dict[str, float],
    best_rank: dict[str, int],
    first_list: dict[str, int],
) -> list[str]:
    return sorted(
        scores,
        key=lambda tag: (-scores[tag], first_list.get(tag, 10**9), best_rank.get(tag, 10**9), tag),
    )


def selection_union_order(selected_tags: list[str], other_lists: list[list[str]]) -> list[str]:
    """Selected hypothesis at the head, remaining union appended by best rank elsewhere."""
    order = list(selected_tags)
    seen = set(order)
    remainder_best: dict[str, int] = {}
    for tags in other_lists:
        for rank, tag in enumerate(tags, start=1):
            if tag in seen:
                continue
            if tag not in remainder_best or rank < remainder_best[tag]:
                remainder_best[tag] = rank
    order.extend(sorted(remainder_best, key=lambda tag: (remainder_best[tag], tag)))
    return order


def apply_rerank(
    order: list[str],
    scores: dict[str, float] | None,
    lists_fused: int,
    consensus: dict[str, float],
    kappa: float,
    beta: float,
    top_k: int,
    score_variant: str,
    score_scaling: str = "raw",
) -> list[str]:
    """Rerank by retrieval score + beta * consensus_agreement.

    `raw` mirrors rerank_consensus_base as the pipeline implements it: beta is added to an
    unnormalized RRF score whose scale grows with the number of fused lists, so beta's
    effective strength depends on J. `range_normalized` min-max scales the retrieval score
    to [0, 1] over the fact's own candidate pool first, so beta is expressed in units of
    the observed retrieval-score range and means the same thing at every J.
    """
    if score_scaling not in SCORE_SCALINGS:
        raise ValueError(f"Unsupported score_scaling={score_scaling}")

    retrieval: list[float] = []
    for rank, tag in enumerate(order, start=1):
        score = scores[tag] if scores is not None else 1.0 / (kappa + rank)
        if score_variant == "mean":
            score /= lists_fused
        retrieval.append(score)

    if score_scaling == "range_normalized" and retrieval:
        low = min(retrieval)
        high = max(retrieval)
        span = high - low
        retrieval = [(value - low) / span if span > 0 else 1.0 for value in retrieval]

    rescored = [
        (score + beta * consensus.get(tag, 0.0), rank, tag)
        for rank, (tag, score) in enumerate(zip(order, retrieval, strict=True), start=1)
    ]
    rescored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [tag for _, _, tag in rescored[:top_k]]


def score_configuration(
    context: FactContext,
    *,
    hypotheses_used: int,
    aggregation: str,
    score_variant: str,
    rerank_beta: float,
    rrf_kappa: float,
    top_k: int,
    score_scaling: str = "raw",
) -> ConfigurationResult:
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Unsupported aggregation={aggregation}")
    if score_variant not in SCORE_VARIANTS:
        raise ValueError(f"Unsupported score_variant={score_variant}")
    if not 1 <= hypotheses_used <= len(context.candidate_lists):
        raise ValueError(f"hypotheses_used={hypotheses_used} outside the logged hypotheses")

    lists = context.candidate_lists[:hypotheses_used]
    gold = context.example.gold_tags
    selected_idx: int | None = None
    scores: dict[str, float] | None = None
    lists_fused = 1

    if aggregation == "oracle":
        observed = [rank for rank in (first_gold_rank(tags, gold) for tags in lists) if rank is not None]
        oracle_rank = min(observed) if observed else None
        return ConfigurationResult(oracle_rank, oracle_rank, None, len(lists))

    if aggregation == "none":
        if hypotheses_used != 1:
            raise ValueError("aggregation=none requires hypotheses_used=1")
        order = lists[0]
    elif aggregation == "rrf":
        scores, best_rank, first_list = rrf_scores(lists, rrf_kappa)
        order = order_from_scores(scores, best_rank, first_list)
        lists_fused = len(lists)
    elif aggregation == "selection":
        selected_idx = context.selected_idx_by_j[hypotheses_used]
        order = lists[selected_idx]
    else:  # selection_union
        selected_idx = context.selected_idx_by_j[hypotheses_used]
        order = selection_union_order(
            lists[selected_idx],
            [tags for idx, tags in enumerate(lists) if idx != selected_idx],
        )

    before = first_gold_rank(order[:top_k], gold)
    if rerank_beta == 0.0:
        return ConfigurationResult(before, before, selected_idx, lists_fused)

    reranked = apply_rerank(
        order,
        scores,
        lists_fused,
        context.consensus_over(hypotheses_used),
        rrf_kappa,
        rerank_beta,
        top_k,
        score_variant,
        score_scaling,
    )
    return ConfigurationResult(first_gold_rank(reranked, gold), before, selected_idx, lists_fused)
