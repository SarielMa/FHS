#!/usr/bin/env python3
"""frozen_ags: the frozen AGS single-pass grounding method, wired for the FullTagging
evaluation pipeline.

This implements the method spec in comparing_methods/ags_method_implementation.md at its
frozen configuration (spec section 9.2): J=2, K=200, kappa=60, beta=0.6, w_cov=1.0,
sum-RRF, range-normalized, dual rendering for table and definition-only for text. That
configuration is the one validated offline in the AGS analysis runs (it is Task-A's best
range-normalized cell), so "frozen" names the committed, gated config rather than a
tunable one.

Every scoring primitive is reused rather than re-derived, per the spec's repeated
instruction that measured behavior depends on how BM25 and the coverage/agree terms are
scaled against one another:

  tokenizer / label coverage / BM25 : run_fintagging_grounding_baseline (TaxonomyRetriever)
  def / lab rendering               : run_ags_component_validation
  sum-RRF fusion                    : run_fintagging_grounding_baseline.fuse_round_candidates
  dimension agreement / consensus   : ags_configuration_scoring.FactContext (+ ags_symbolic_agreement)

The pipeline entry point is build_frozen_ags_method_record, which produces one
bm25_candidates.jsonl record whose `candidates` field is the AGS-reranked ranking. The
method makes no LLM call after hypothesis generation, so the FullTagging listwise rerank
stage is disabled for this mode (see the shell wiring) and the evaluator scores the
`candidates` ordering directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ags_configuration_scoring import FactContext
from ags_symbolic_agreement import DIMENSIONS as SYMBOLIC_DIMENSIONS  # noqa: F401  (parity check)
from run_ags_component_validation import render_definition, render_label
from run_fintagging_grounding_baseline import (
    Example,
    TaxonomyRetriever,
    build_direct_query,
    build_operator_initial_messages,
    build_prompt_under_query_budget,
    finalize_candidate_record,
    fuse_round_candidates,
    llm_call_record,
    metric_row,
    normalize_space,
    normalize_tag,
    parse_hypothesis,
    retrieval_query_from_grounding,
    retrieve_candidates,
    tokenize,
)


FROZEN_AGS_QUERY_MODE = "frozen_ags"
# "One-pass grounding (structured)": AGS with J=1. Same generator prompt and schema, same
# renderer, same index/filter/K/kappa/w_cov -- the ONLY differences are that one hypothesis is
# drawn instead of two and, because a single hypothesis has nothing to reach consensus with,
# beta=0 so the rerank term vanishes and the output is ranked purely by the range-normalized
# fused score. It isolates the structured representation from the ensemble machinery.
#
# Decoding is greedy (temperature 0). frozen_ags samples at 0.8 precisely because it needs J
# *independent* draws; with J=1 there is nothing to decorrelate, so sampling noise would only
# add variance to a baseline whose whole job is to be a clean reference point.
ONE_PASS_STRUCTURED_QUERY_MODE = "one_pass_structured"

# Concepts whose own label must self-retrieve at rank 1 only when the coverage term is
# active and correctly scaled (spec 9.2, assertion 2). Short generic labels that lose to
# long compounds repeating the term without the |label|-normalized coverage direction.
COVERAGE_REGRESSION_LABELS = (
    "Assets",
    "Liabilities",
    "Revenues",
    "Goodwill",
    "Depreciation",
    "RegulatoryAssetsCurrent",
)


@dataclass(frozen=True)
class FrozenAgsConfig:
    """The frozen configuration. These values are asserted at startup, not tuned.

    `variant` selects WHICH frozen configuration is being asserted. Both variants share this
    class, and therefore share every code path below, because the one-pass-structured baseline
    is defined as "AGS with J=1" -- reusing the class is what makes that literally true rather
    than a claim about two similar implementations.
    """

    hypotheses: int = 2  # J
    top_k: int = 200  # K
    rrf_kappa: int = 60  # kappa
    rerank_beta: float = 0.6  # beta
    label_coverage_weight: float = 1.0  # w_cov
    temperature: float = 0.8
    agreement_top_m: int = 10
    dual_rendering_modalities: tuple[str, ...] = ("table",)  # text uses def only
    variant: str = FROZEN_AGS_QUERY_MODE


# Each variant is frozen just as hard as the original; they differ only in the constants the
# assertion demands. Adding a variant here must never loosen an existing one.
_FROZEN_VARIANTS: dict[str, dict[str, Any]] = {
    FROZEN_AGS_QUERY_MODE: {
        "hypotheses": 2,
        "rerank_beta": 0.6,
        "rrf_kappa": 60,
        "label_coverage_weight": 1.0,
        "top_k": 200,
    },
    ONE_PASS_STRUCTURED_QUERY_MODE: {
        "hypotheses": 1,
        "rerank_beta": 0.0,
        "rrf_kappa": 60,
        "label_coverage_weight": 1.0,
        "top_k": 200,
    },
}


def one_pass_structured_config() -> FrozenAgsConfig:
    """AGS with J=1: one greedy hypothesis, no consensus rerank, everything else identical."""
    return FrozenAgsConfig(
        hypotheses=1,
        rerank_beta=0.0,
        temperature=0.0,  # greedy, single sample
        variant=ONE_PASS_STRUCTURED_QUERY_MODE,
    )


def _assert_frozen(cfg: FrozenAgsConfig) -> None:
    """Spec 9.2 assertion 4: config frozen."""
    expected = _FROZEN_VARIANTS.get(cfg.variant)
    if expected is None:
        raise AssertionError(
            f"unknown frozen config variant {cfg.variant!r}; expected one of "
            f"{sorted(_FROZEN_VARIANTS)}"
        )
    drifted = {
        field: (getattr(cfg, field), value)
        for field, value in expected.items()
        if getattr(cfg, field) != value
    }
    if drifted:
        detail = ", ".join(f"{field}={got!r} (expected {want!r})" for field, (got, want) in sorted(drifted.items()))
        raise AssertionError(f"{cfg.variant} config drifted: {detail}")
    if cfg.variant == ONE_PASS_STRUCTURED_QUERY_MODE and cfg.temperature != 0.0:
        raise AssertionError(
            "one_pass_structured decodes greedily (temperature 0); got "
            f"temperature={cfg.temperature}"
        )


def range_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max to [0, 1] over the candidate pool (spec section 8).

    All-tied collapses to zero so agree() alone orders the pool, rather than to one; this
    is the documented edge case and differs deliberately from a retrieval-only default.
    """
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high - low < 1e-12:
        return {tag: 0.0 for tag in scores}
    span = high - low
    return {tag: (value - low) / span for tag, value in scores.items()}


def frozen_ags_startup_assertions(
    retriever: TaxonomyRetriever,
    taxonomy: list[Any],
    normalization_map: dict[str, Any],
    cfg: FrozenAgsConfig,
    self_retrieval_sample: int = 200,
    self_retrieval_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Fail loudly before processing anything (spec section 9.2).

    The label self-retrieval sweep (assertion 1) is capped at `self_retrieval_sample`
    concepts for a bounded startup cost; the six coverage-regression labels (assertion 2)
    are always checked in full. Assumes the retriever already carries w_cov = 1.0.

    Assertion 1's purpose is to catch a renderer/index tokenizer mismatch, which makes
    nearly every concept fail to retrieve itself. On the real enriched taxonomy a small
    residue of near-synonym labels ("...Sale Of Property" vs "...Sale Of Properties") lose
    rank 1 to a fuller-coverage sibling; that is a property of the index, not a mismatch,
    so the assertion trips only when the failure RATE exceeds `self_retrieval_tolerance`.
    """
    _assert_frozen(cfg)

    if retriever.label_coverage_weight <= 0.0:
        raise AssertionError(
            "frozen_ags requires the retriever's label_coverage_weight > 0 (w_cov=1.0); "
            f"got {retriever.label_coverage_weight}"
        )

    # Assertion 3: vocabulary present where required, empty where required.
    vocab = normalization_map.get("dimensions", {})
    for dimension in ("family", "qualifier", "scope", "temporal"):
        if not (vocab.get(dimension) or {}):
            raise AssertionError(f"frozen_ags vocabulary missing categories for '{dimension}'")
    for dimension in ("role", "event"):
        if vocab.get(dimension):
            raise AssertionError(
                f"frozen_ags expects no controlled vocabulary for '{dimension}' (token branch only)"
            )

    # Assertion 1: tokenizer self-retrieval. A renderer/index tokenization mismatch shows
    # up as a concept failing to retrieve itself from its own label.
    self_retrieval_failures: list[str] = []
    checked = 0
    for concept in taxonomy:
        if checked >= self_retrieval_sample:
            break
        label = getattr(concept, "standard_label", "") or getattr(concept, "raw_tag", "")
        if not label:
            continue
        checked += 1
        ranked = retrieve_candidates(retriever, label, concept.entity_type, 10)
        if not ranked or normalize_tag(ranked[0]["tag"]) != normalize_tag(concept.tag):
            self_retrieval_failures.append(concept.tag)

    # Assertion 2: coverage regression on six short/compound-prone labels.
    by_label = {
        (getattr(concept, "standard_label", "") or "").replace(" ", ""): concept
        for concept in taxonomy
    }
    by_raw = {normalize_tag(concept.tag).split(":")[-1]: concept for concept in taxonomy}
    coverage_failures: list[str] = []
    coverage_checked: list[str] = []
    for label in COVERAGE_REGRESSION_LABELS:
        concept = by_label.get(label) or by_raw.get(label)
        if concept is None:
            continue
        coverage_checked.append(concept.tag)
        query_label = getattr(concept, "standard_label", "") or label
        ranked = retrieve_candidates(retriever, query_label, concept.entity_type, 10)
        if not ranked or normalize_tag(ranked[0]["tag"]) != normalize_tag(concept.tag):
            coverage_failures.append(concept.tag)

    self_retrieval_failure_rate = len(self_retrieval_failures) / checked if checked else 0.0
    report = {
        "self_retrieval_checked": checked,
        "self_retrieval_failure_count": len(self_retrieval_failures),
        "self_retrieval_failure_rate": round(self_retrieval_failure_rate, 4),
        "self_retrieval_tolerance": self_retrieval_tolerance,
        "self_retrieval_failures": self_retrieval_failures[:20],
        "coverage_regression_checked": coverage_checked,
        "coverage_regression_failures": coverage_failures,
        "vocabulary_ok": True,
        "config_frozen_ok": True,
    }
    if self_retrieval_failure_rate > self_retrieval_tolerance:
        raise AssertionError(
            f"frozen_ags tokenizer self-retrieval failed for {len(self_retrieval_failures)}/"
            f"{checked} concept(s) (rate {self_retrieval_failure_rate:.3f} > "
            f"{self_retrieval_tolerance}); this rate indicates a renderer/index tokenizer "
            f"mismatch rather than near-synonym collisions. Examples: {self_retrieval_failures[:5]}"
        )
    if coverage_failures:
        raise AssertionError(
            f"frozen_ags coverage regression failed for {coverage_failures}; the coverage "
            "term is inactive or mis-scaled"
        )
    return report


def sample_hypotheses(
    generator: Any,
    args: Any,
    example: Example,
    cfg: FrozenAgsConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """J independent structured samples at the frozen temperature (spec section 4).

    Reuses the pipeline's operator-initial prompt (which asks for the six dimensions and a
    compact retrieval query) and parse_hypothesis. A sample that fails to parse is retried
    once and then dropped; if none survive, the serialized instance is used as the query
    and the fallback is logged.
    """
    fallback = build_direct_query(example)
    prompt, prompt_tokens, used_context_chars = build_prompt_under_query_budget(
        generator.tokenizer,
        lambda ctx_chars: build_operator_initial_messages(example, ctx_chars),
        context_max_chars=args.query_context_max_chars,
        max_input_tokens=args.query_max_input_tokens,
    )

    hypotheses: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    original_temperature = generator.args.query_temperature
    original_top_p = generator.args.query_top_p
    generator.args.query_temperature = cfg.temperature
    # At temperature 0 the decode is greedy and a nucleus cutoff is meaningless; pin top_p to
    # 1.0 so a stray --frozen-ags-top-p on the command line cannot make "greedy" ambiguous.
    generator.args.query_top_p = 1.0 if cfg.temperature == 0.0 else getattr(args, "frozen_ags_top_p", 1.0)
    try:
        for sample_idx in range(cfg.hypotheses):
            raw_output = generator.generate_one(prompt)
            hypothesis, parse_ok = parse_hypothesis(raw_output, fallback)
            if not parse_ok:
                # Retry once, then drop.
                raw_output = generator.generate_one(prompt)
                hypothesis, parse_ok = parse_hypothesis(raw_output, fallback)
            calls.append(
                llm_call_record(
                    f"{cfg.variant}_hypothesis",
                    raw_output=raw_output,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=generator.count_text_tokens(raw_output),
                    parse_ok=parse_ok,
                    backend=generator.backend,
                    model_name=generator.model_name,
                    extra_fields={
                        "sample_idx": sample_idx,
                        "used_context_max_chars": used_context_chars,
                    },
                )
            )
            if parse_ok:
                hypothesis["hypothesis_idx"] = len(hypotheses)
                hypotheses.append(hypothesis)
    finally:
        generator.args.query_temperature = original_temperature
        generator.args.query_top_p = original_top_p

    used_fallback = False
    if not hypotheses:
        used_fallback = True
        hypotheses.append(
            {
                "dimensions": {dimension: "UNRESOLVED" for dimension in SYMBOLIC_DIMENSIONS},
                "operators": ["direct_label"],
                "retrieval_query": fallback,
                "hypothesis_idx": 0,
            }
        )
    return hypotheses, calls, used_fallback


def frozen_ags_rankings(
    retriever: TaxonomyRetriever,
    example: Example,
    hypotheses: list[dict[str, Any]],
    cfg: FrozenAgsConfig,
) -> list[dict[str, Any]]:
    """Render + retrieve (spec 9 steps 2-3). One flat ranking list per (hypothesis,
    rendering): definition always, label additionally for dual-rendering modalities when
    render_label resolves. The retriever must already carry w_cov = 1.0."""
    dual = example.input_type in cfg.dual_rendering_modalities
    rounds: list[dict[str, Any]] = []
    round_idx = 0
    for hypothesis in hypotheses:
        hypothesis_idx = int(hypothesis.get("hypothesis_idx", 0))
        q_def = render_definition(hypothesis)
        query_def = retrieval_query_from_grounding(example, q_def)
        round_idx += 1
        rounds.append(
            {
                "round": round_idx,
                "hypothesis_idx": hypothesis_idx,
                "rendering": "def",
                "grounding": normalize_space(q_def),
                "query": query_def,
                "candidates": retrieve_candidates(retriever, query_def, example.entity_type, cfg.top_k),
                "label_render_skipped": False,
            }
        )
        if not dual:
            continue
        q_lab = render_label(hypothesis)
        if not q_lab:
            rounds[-1]["label_render_skipped"] = True
            continue
        query_lab = retrieval_query_from_grounding(example, q_lab)
        round_idx += 1
        rounds.append(
            {
                "round": round_idx,
                "hypothesis_idx": hypothesis_idx,
                "rendering": "lab",
                "grounding": normalize_space(q_lab),
                "query": query_lab,
                "candidates": retrieve_candidates(retriever, query_lab, example.entity_type, cfg.top_k),
                "label_render_skipped": False,
            }
        )
    return rounds


def frozen_ags_rerank(
    rounds: list[dict[str, Any]],
    example: Example,
    hypotheses: list[dict[str, Any]],
    normalization_map: dict[str, Any],
    cfg: FrozenAgsConfig,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Fuse + rerank over the FULL pool (spec 9 steps 4-6).

    Flat sum-RRF over every rendering ranking, range-normalize the fused score, then add
    beta * agree_consensus(concept, hyps). Agreement/consensus come from the validated
    FactContext, so the agree() semantics match the offline AGS analysis exactly.
    """
    fused = fuse_round_candidates(rounds, cfg.top_k, cfg.rrf_kappa)
    if not fused:
        return [], {}

    rrf_by_tag = {normalize_tag(candidate["tag"]): float(candidate.get("rrf_score", 0.0)) for candidate in fused}
    normed = range_normalize(rrf_by_tag)

    # FactContext computes agree(concept, h) per candidate against each hypothesis and
    # averages; the full fused pool is exposed through the round candidate dicts so every
    # rerank target has an agreement value. candidate_by_tag is the union over all rounds,
    # so consensus covers every fused tag regardless of which rendering surfaced it.
    agreement_records = [
        {
            "candidate_ids": [normalize_tag(candidate["tag"]) for candidate in round_record["candidates"]],
            "candidates": round_record["candidates"],
        }
        for round_record in rounds
    ]
    context = FactContext(
        example=example,
        records=agreement_records,
        hypotheses=hypotheses,
        normalization_map=normalization_map,
        agreement_top_m=cfg.agreement_top_m,
    )
    consensus = context.consensus_over(len(hypotheses))

    final_scores = {
        tag: normed.get(tag, 0.0) + cfg.rerank_beta * consensus.get(tag, 0.0) for tag in normed
    }
    order = sorted(final_scores, key=lambda tag: (-final_scores[tag], tag))[: cfg.top_k]

    candidate_by_tag = {normalize_tag(candidate["tag"]): candidate for candidate in fused}
    reranked: list[dict[str, Any]] = []
    for rank, tag in enumerate(order, start=1):
        candidate = dict(candidate_by_tag[tag])
        candidate["rank"] = rank
        candidate["frozen_ags_rrf_score"] = round(rrf_by_tag.get(tag, 0.0), 8)
        candidate["frozen_ags_rrf_normalized"] = round(normed.get(tag, 0.0), 6)
        candidate["frozen_ags_agree_consensus"] = round(consensus.get(tag, 0.0), 6)
        candidate["frozen_ags_final_score"] = round(final_scores[tag], 6)
        reranked.append(candidate)
    diagnostics = {
        "rrf_normalized": {tag: round(value, 6) for tag, value in normed.items()},
        "agree_consensus": {tag: round(value, 6) for tag, value in consensus.items()},
    }
    return reranked, diagnostics


def build_frozen_ags_method_record(
    args: Any,
    generator: Any,
    retriever: TaxonomyRetriever,
    example: Example,
    normalization_map: dict[str, Any],
    cfg: FrozenAgsConfig | None = None,
) -> dict[str, Any]:
    """Top-level pipeline entry: produce one candidate record for `example`.

    The record's `candidates` (and `final_candidates`) hold the AGS-reranked ranking, so
    the FullTagging evaluator scores the method's own output with no downstream listwise
    rerank required.
    """
    cfg = cfg or FrozenAgsConfig()
    _assert_frozen(cfg)
    start_time = time.monotonic()

    hypotheses, calls, used_fallback = sample_hypotheses(generator, args, example, cfg)
    rounds = frozen_ags_rankings(retriever, example, hypotheses, cfg)
    reranked, _ = frozen_ags_rerank(rounds, example, hypotheses, normalization_map, cfg)

    prompt_tokens = sum(int(call.get("prompt_tokens", 0) or 0) for call in calls)
    completion_tokens = sum(int(call.get("completion_tokens", 0) or 0) for call in calls)

    # finalize_candidate_record re-fuses `rounds` with a plain sum-RRF; for frozen_ags the
    # authoritative ranking is the reranked pool, so overwrite the ranking-derived fields.
    # The extra_fields keys stay `frozen_ags_*` for BOTH variants on purpose: every downstream
    # reader (the T28 replay, the Table 5 ablation loaders, compute_ags_*) keys off those names,
    # and a one-pass-structured trace is meant to be readable by all of them unchanged. The
    # `variant` entry below is what distinguishes the two, and record["query_mode"] carries it too.
    record = finalize_candidate_record(
        example,
        query_mode=cfg.variant,
        rounds=rounds,
        top_k=cfg.top_k,
        rrf_kappa=cfg.rrf_kappa,
        total_llm_calls=len(calls),
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        wall_time=time.monotonic() - start_time,
        extra_fields={
            "frozen_ags_config": {
                "variant": cfg.variant,
                "hypotheses": cfg.hypotheses,
                "top_k": cfg.top_k,
                "rrf_kappa": cfg.rrf_kappa,
                "rerank_beta": cfg.rerank_beta,
                "label_coverage_weight": cfg.label_coverage_weight,
                "temperature": cfg.temperature,
                "dual_rendering_modalities": list(cfg.dual_rendering_modalities),
            },
            "frozen_ags_hypotheses": hypotheses,
            "frozen_ags_used_fallback": used_fallback,
            "frozen_ags_llm_calls": calls,
        },
    )

    candidate_tags = [candidate["tag"] for candidate in reranked]
    retrieval_metrics = metric_row(candidate_tags, example.gold_tags, (10, 50, cfg.top_k))
    record["candidates"] = reranked
    record["final_candidates"] = reranked
    record["candidate_union_tags"] = candidate_tags
    record["retrieval_metrics"] = retrieval_metrics
    record["gold_rank"] = retrieval_metrics.get("rank")
    record["search_coverage"] = any(
        normalize_tag(tag) in {normalize_tag(candidate) for candidate in candidate_tags}
        for tag in example.gold_tags
    )
    record["total_retrieval_calls"] = len(rounds)
    return record
