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
bm25_candidates.jsonl record whose `candidates` field is the AGS-reranked ranking. For
the full frozen_ags arm, the internal six-dimensional FHS verifier is the final LLM
reranker, so the generic FullTagging listwise rerank stage is disabled for this mode and
the evaluator scores the `candidates` ordering directly.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, replace
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
    format_candidate_for_prompt,
    fuse_round_candidates,
    llm_call_record,
    metric_row,
    normalize_space,
    normalize_tag,
    parse_json_object,
    parse_hypothesis,
    retrieval_query_from_grounding,
    retrieve_candidates,
    serialize_evidence,
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

FHS_VERIFIER_DIMENSIONS = ("FAMILY", "ROLE", "EVENT", "QUALIFIER", "SCOPE", "TEMPORAL")
VERDICT_TO_SUPPORT = {"support": 1.0, "contradict": 0.0, "unresolved": 0.0}
FHS_J1_QUERY_MODE = "fhs_j1"
FHS_NO_VERIFIER_QUERY_MODE = "fhs_no_verifier"
FHS_J3_WCOV0_QUERY_MODE = "fhs_j3_wcov0"
FHS_J3_WCOV1_QUERY_MODE = "fhs_j3_wcov1"
FHS_J4_WCOV0_QUERY_MODE = "fhs_j4_wcov0"
FHS_J4_WCOV1_QUERY_MODE = "fhs_j4_wcov1"


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
    label_coverage_ablation: bool = False


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
    FHS_J1_QUERY_MODE: {
        "hypotheses": 1,
        "rerank_beta": 0.6,
        "rrf_kappa": 60,
        "label_coverage_weight": 1.0,
        "top_k": 200,
    },
    FHS_NO_VERIFIER_QUERY_MODE: {
        "hypotheses": 2,
        "rerank_beta": 0.0,
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
    FHS_J3_WCOV0_QUERY_MODE: {
        "hypotheses": 3,
        "rerank_beta": 0.6,
        "rrf_kappa": 60,
        "label_coverage_weight": 0.0,
        "top_k": 200,
    },
    FHS_J3_WCOV1_QUERY_MODE: {
        "hypotheses": 3,
        "rerank_beta": 0.6,
        "rrf_kappa": 60,
        "label_coverage_weight": 1.0,
        "top_k": 200,
    },
    FHS_J4_WCOV0_QUERY_MODE: {
        "hypotheses": 4,
        "rerank_beta": 0.6,
        "rrf_kappa": 60,
        "label_coverage_weight": 0.0,
        "top_k": 200,
    },
    FHS_J4_WCOV1_QUERY_MODE: {
        "hypotheses": 4,
        "rerank_beta": 0.6,
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


def fhs_j1_config() -> FrozenAgsConfig:
    """FHS ablation with one structured hypothesis and the candidate-level verifier retained."""
    return FrozenAgsConfig(
        hypotheses=1,
        rerank_beta=0.6,
        temperature=0.8,
        variant=FHS_J1_QUERY_MODE,
    )


def fhs_no_verifier_config() -> FrozenAgsConfig:
    """FHS ablation with two hypotheses but no candidate-level verifier rerank term."""
    return FrozenAgsConfig(
        hypotheses=2,
        rerank_beta=0.0,
        temperature=0.8,
        variant=FHS_NO_VERIFIER_QUERY_MODE,
    )


def fhs_j3_wcov0_config() -> FrozenAgsConfig:
    """FHS J=3 sweep cell with label coverage disabled."""
    return FrozenAgsConfig(
        hypotheses=3,
        rerank_beta=0.6,
        label_coverage_weight=0.0,
        temperature=0.8,
        variant=FHS_J3_WCOV0_QUERY_MODE,
    )


def fhs_j3_wcov1_config() -> FrozenAgsConfig:
    """FHS J=3 sweep cell with label coverage enabled."""
    return FrozenAgsConfig(
        hypotheses=3,
        rerank_beta=0.6,
        label_coverage_weight=1.0,
        temperature=0.8,
        variant=FHS_J3_WCOV1_QUERY_MODE,
    )


def fhs_j4_wcov0_config() -> FrozenAgsConfig:
    """FHS J=4 sweep cell with label coverage disabled."""
    return FrozenAgsConfig(
        hypotheses=4,
        rerank_beta=0.6,
        label_coverage_weight=0.0,
        temperature=0.8,
        variant=FHS_J4_WCOV0_QUERY_MODE,
    )


def fhs_j4_wcov1_config() -> FrozenAgsConfig:
    """FHS J=4 sweep cell with label coverage enabled."""
    return FrozenAgsConfig(
        hypotheses=4,
        rerank_beta=0.6,
        label_coverage_weight=1.0,
        temperature=0.8,
        variant=FHS_J4_WCOV1_QUERY_MODE,
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
        and not (field == "label_coverage_weight" and cfg.label_coverage_ablation)
    }
    if drifted:
        detail = ", ".join(f"{field}={got!r} (expected {want!r})" for field, (got, want) in sorted(drifted.items()))
        raise AssertionError(f"{cfg.variant} config drifted: {detail}")
    if cfg.variant == ONE_PASS_STRUCTURED_QUERY_MODE and cfg.temperature != 0.0:
        raise AssertionError(
            "one_pass_structured decodes greedily (temperature 0); got "
            f"temperature={cfg.temperature}"
        )


def with_label_coverage_weight(cfg: FrozenAgsConfig, label_coverage_weight: float) -> FrozenAgsConfig:
    """Keep the frozen variant fixed while changing only w_cov for an ablation cell."""
    return replace(
        cfg,
        label_coverage_weight=label_coverage_weight,
        label_coverage_ablation=True,
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

    coverage_enabled = cfg.label_coverage_weight > 0.0
    if coverage_enabled and retriever.label_coverage_weight <= 0.0:
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
    if coverage_enabled:
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


def fused_only_rerank(
    rounds: list[dict[str, Any]],
    cfg: FrozenAgsConfig,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    fused = fuse_round_candidates(rounds, cfg.top_k, cfg.rrf_kappa)
    if not fused:
        return [], {}
    rrf_by_tag = {normalize_tag(candidate["tag"]): float(candidate.get("rrf_score", 0.0)) for candidate in fused}
    normed = range_normalize(rrf_by_tag)
    order = sorted(normed, key=lambda tag: (-normed[tag], tag))[: cfg.top_k]
    candidate_by_tag = {normalize_tag(candidate["tag"]): candidate for candidate in fused}
    reranked: list[dict[str, Any]] = []
    for rank, tag in enumerate(order, start=1):
        candidate = dict(candidate_by_tag[tag])
        candidate["rank"] = rank
        candidate["frozen_ags_rrf_score"] = round(rrf_by_tag.get(tag, 0.0), 8)
        candidate["frozen_ags_rrf_normalized"] = round(normed.get(tag, 0.0), 6)
        candidate["frozen_ags_verifier_support"] = 0.0
        candidate["frozen_ags_final_score"] = round(normed.get(tag, 0.0), 6)
        reranked.append(candidate)
    return reranked, {
        "rrf_normalized": {tag: round(value, 6) for tag, value in normed.items()},
        "fhs_verifier_support": {},
    }


def build_fhs_verifier_messages(
    example: Example,
    hypothesis: dict[str, Any],
    candidates: list[dict[str, Any]],
    context_max_chars: int,
    doc_max_chars: int,
) -> list[dict[str, str]]:
    evidence = serialize_evidence(example, context_max_chars)
    candidate_text = "\n\n".join(format_candidate_for_prompt(candidate, doc_max_chars) for candidate in candidates)
    dims = hypothesis.get("dimensions", {})
    shown = {
        dimension: normalize_space(dims.get(dimension, dims.get(dimension.lower(), "UNRESOLVED")))
        for dimension in FHS_VERIFIER_DIMENSIONS
    }
    dimension_list = ", ".join(FHS_VERIFIER_DIMENSIONS)
    schema_fields = ", ".join(f'"{dimension}": "support|contradict|unresolved"' for dimension in FHS_VERIFIER_DIMENSIONS)
    user = f"""Judge how well each candidate ICD-10-CM concept matches the structured clinical hypothesis.

Judge all six dimensions: {dimension_list}.
Use "support" when the candidate supports the hypothesis dimension, "contradict" when it conflicts, and "unresolved" when the candidate text does not let you decide.
Return one verdict entry for each candidate. Do not choose the answer directly; score each candidate independently against the hypothesis.

Hypothesis:
{shown}

Evidence:
{evidence}

Candidates:
{candidate_text}

Return JSON only with this schema:
{{"verdicts": [{{"tag": "A00.0", {schema_fields}, "confidence": 0.0}}]}}"""
    return [
        {"role": "system", "content": "You verify ICD-10-CM grounding hypotheses against candidate diagnosis concepts on six dimensions."},
        {"role": "user", "content": user},
    ]


def _clean_llm_json_text(raw_output: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL | re.IGNORECASE).strip()
    return cleaned.strip("`")


def _extract_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    depth = 0
    start: int | None = None
    in_string = False
    escape = False
    for idx, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(text[start : idx + 1])
                    start = None
    return objects


def _parse_fhs_verdict_entries(raw_entries: Any, candidate_tags: list[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_entries, list):
        return {}
    allowed = set(candidate_tags)
    by_tag: dict[str, dict[str, Any]] = {}
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        tag = normalize_tag(entry.get("tag", ""))
        if tag not in allowed:
            continue
        verdict: dict[str, Any] = {}
        for dimension in FHS_VERIFIER_DIMENSIONS:
            raw = normalize_space(entry.get(dimension, "")).lower()
            verdict[dimension] = raw if raw in VERDICT_TO_SUPPORT else "unresolved"
        try:
            verdict["confidence"] = float(entry.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            verdict["confidence"] = 0.0
        by_tag[tag] = verdict
    return by_tag


def _salvage_fhs_verdicts(raw_output: str, candidate_tags: list[str]) -> dict[str, dict[str, Any]]:
    cleaned = _clean_llm_json_text(raw_output)
    verdict_marker = cleaned.find('"verdicts"')
    search_space = cleaned[verdict_marker:] if verdict_marker >= 0 else cleaned
    array_start = search_space.find("[")
    if array_start >= 0:
        search_space = search_space[array_start + 1 :]
    entries: list[dict[str, Any]] = []
    for object_text in _extract_json_objects(search_space):
        try:
            parsed = json.loads(object_text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "tag" in parsed:
            entries.append(parsed)
    return _parse_fhs_verdict_entries(entries, candidate_tags)


def parse_fhs_verifier_output(raw_output: str, candidate_tags: list[str]) -> tuple[dict[str, dict[str, Any]], bool, str]:
    parsed, parse_ok = parse_json_object(raw_output)
    entries = parsed.get("verdicts") if isinstance(parsed, dict) else None
    by_tag = _parse_fhs_verdict_entries(entries, candidate_tags)
    if parse_ok and by_tag:
        return by_tag, True, "clean"
    salvaged = _salvage_fhs_verdicts(raw_output, candidate_tags)
    if salvaged:
        return salvaged, True, "salvaged"
    return {}, False, "failed"


def fhs_verifier_score(verdict: dict[str, Any] | None) -> float:
    if verdict is None:
        return 0.0
    values = [VERDICT_TO_SUPPORT.get(str(verdict.get(dimension, "unresolved")).lower(), 0.0) for dimension in FHS_VERIFIER_DIMENSIONS]
    return sum(values) / len(FHS_VERIFIER_DIMENSIONS)


def fhs_verifier_rerank(
    args: Any,
    generator: Any,
    rounds: list[dict[str, Any]],
    example: Example,
    hypotheses: list[dict[str, Any]],
    cfg: FrozenAgsConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    fused = fuse_round_candidates(rounds, cfg.top_k, cfg.rrf_kappa)
    if not fused:
        return [], {}, []

    rrf_by_tag = {normalize_tag(candidate["tag"]): float(candidate.get("rrf_score", 0.0)) for candidate in fused}
    normed = range_normalize(rrf_by_tag)
    fused_order = sorted(normed, key=lambda tag: (-normed[tag], tag))
    candidate_by_tag = {normalize_tag(candidate["tag"]): candidate for candidate in fused}
    window_tags = fused_order[: cfg.agreement_top_m]
    window = [candidate_by_tag[tag] for tag in window_tags]

    prompts: list[str] = []
    prompt_meta: list[tuple[int, int]] = []
    hyp_indices: list[int] = []
    for hypothesis in hypotheses:
        prompt, prompt_tokens, used_context_chars = build_prompt_under_query_budget(
            generator.tokenizer,
            lambda ctx_chars, hypothesis=hypothesis: build_fhs_verifier_messages(
                example,
                hypothesis,
                window,
                ctx_chars,
                getattr(args, "candidate_doc_max_chars", 320),
            ),
            context_max_chars=getattr(args, "query_context_max_chars", 12000),
            max_input_tokens=getattr(args, "query_max_input_tokens", 16000),
        )
        prompts.append(prompt)
        prompt_meta.append((prompt_tokens, used_context_chars))
        hyp_indices.append(int(hypothesis.get("hypothesis_idx", len(hyp_indices))))

    verifier_max_new_tokens = int(
        getattr(args, "fhs_verifier_max_new_tokens", getattr(args, "query_max_new_tokens", 512)) or 512
    )
    original_query_max_new_tokens = getattr(generator.args, "query_max_new_tokens", None)
    if original_query_max_new_tokens is not None:
        generator.args.query_max_new_tokens = verifier_max_new_tokens
    try:
        raw_outputs = generator.generate_many(prompts)
    finally:
        if original_query_max_new_tokens is not None:
            generator.args.query_max_new_tokens = original_query_max_new_tokens

    verifier_calls: list[dict[str, Any]] = []
    support_by_hypothesis: dict[int, dict[str, float]] = {}
    verdicts_by_hypothesis: dict[int, dict[str, dict[str, Any]]] = {}
    for hyp_idx, raw_output, (prompt_tokens, used_context_chars) in zip(hyp_indices, raw_outputs, prompt_meta, strict=True):
        verdicts, parse_ok, parse_mode = parse_fhs_verifier_output(raw_output, window_tags)
        verdicts_by_hypothesis[hyp_idx] = verdicts
        support_by_hypothesis[hyp_idx] = {
            tag: fhs_verifier_score(verdicts.get(tag)) for tag in window_tags
        }
        verifier_calls.append(
            llm_call_record(
                "fhs_candidate_verifier",
                raw_output=raw_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=generator.count_text_tokens(raw_output),
                parse_ok=parse_ok,
                backend=generator.backend,
                model_name=generator.model_name,
                extra_fields={
                    "hypothesis_idx": hyp_idx,
                    "used_context_max_chars": used_context_chars,
                    "candidate_window_size": len(window_tags),
                    "candidate_window_tags": window_tags,
                    "judged_dimensions": list(FHS_VERIFIER_DIMENSIONS),
                    "verifier_max_new_tokens": verifier_max_new_tokens,
                    "parse_mode": parse_mode,
                    "verdict_count": len(verdicts),
                    "missing_verdict_tags": [tag for tag in window_tags if tag not in verdicts],
                },
            )
        )

    mean_support: dict[str, float] = {}
    for tag in normed:
        values = [support_by_hypothesis.get(hyp_idx, {}).get(tag, 0.0) for hyp_idx in hyp_indices]
        mean_support[tag] = sum(values) / len(values) if values else 0.0

    final_scores = {
        tag: normed.get(tag, 0.0) + cfg.rerank_beta * mean_support.get(tag, 0.0) for tag in normed
    }
    order = sorted(final_scores, key=lambda tag: (-final_scores[tag], tag))[: cfg.top_k]

    reranked: list[dict[str, Any]] = []
    for rank, tag in enumerate(order, start=1):
        candidate = dict(candidate_by_tag[tag])
        candidate["rank"] = rank
        candidate["frozen_ags_rrf_score"] = round(rrf_by_tag.get(tag, 0.0), 8)
        candidate["frozen_ags_rrf_normalized"] = round(normed.get(tag, 0.0), 6)
        candidate["frozen_ags_verifier_support"] = round(mean_support.get(tag, 0.0), 6)
        candidate["frozen_ags_final_score"] = round(final_scores[tag], 6)
        reranked.append(candidate)

    diagnostics = {
        "rrf_normalized": {tag: round(value, 6) for tag, value in normed.items()},
        "fhs_verifier_support": {tag: round(value, 6) for tag, value in mean_support.items() if value},
        "fhs_verifier_window_tags": window_tags,
        "fhs_verifier_dimensions": list(FHS_VERIFIER_DIMENSIONS),
        "fhs_verifier_max_new_tokens": verifier_max_new_tokens,
        "fhs_verifier_verdicts": verdicts_by_hypothesis,
    }
    return reranked, diagnostics, verifier_calls


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
    verifier_calls: list[dict[str, Any]] = []
    if cfg.rerank_beta > 0.0:
        reranked, rerank_diagnostics, verifier_calls = fhs_verifier_rerank(
            args, generator, rounds, example, hypotheses, cfg
        )
    else:
        reranked, rerank_diagnostics = fused_only_rerank(rounds, cfg)

    all_calls = calls + verifier_calls
    prompt_tokens = sum(int(call.get("prompt_tokens", 0) or 0) for call in all_calls)
    completion_tokens = sum(int(call.get("completion_tokens", 0) or 0) for call in all_calls)

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
        total_llm_calls=len(all_calls),
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
            "frozen_ags_verifier_calls": verifier_calls,
            "frozen_ags_rerank_diagnostics": rerank_diagnostics,
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
