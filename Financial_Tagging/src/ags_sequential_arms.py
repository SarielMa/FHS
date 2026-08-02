#!/usr/bin/env python3
"""ags_seq / ags_seq_random: the two sequential negative-control arms (ags_seq_arms_spec.md).

Both arms start from AGS's round one and then run up to B=4 rounds of

    FEEDBACK -> CONTROLLER -> REVISE -> RENDER -> RETRIEVE -> re-consolidate

The single difference between them is directive selection: `ags_seq` samples
theta_o ~ N(mu_o, nu^2 Sigma_o) per operator on the slate and takes argmax psi^T theta_o,
`ags_seq_random` draws uniformly from the same slate. Both compute and update the
posteriors, so "the posteriors moved" and "consulting them helped" stay separable.

Everything before the loop is the frozen AGS code path, unchanged and imported rather than
restated: `sample_hypotheses` for the J=2 hypotheses, `frozen_ags_rankings` for the
modality-conditional rendering and K=200 retrieval, and the same sum-RRF ->
range-normalize -> +beta*consensus consolidation. `assert_round_one_parity` re-derives
round one through the frozen entry point for every fact and fails loudly on any
divergence, which is what makes round-1-vs-full an internal comparison rather than a
comparison between two systems.

The consolidation here is a local reimplementation only because it must run several times
per fact (each round, plus the counterfactual replay, plus an untruncated pool for the
utility) and the shared `FactContext` re-parses every candidate's symbolic profile on each
call. The local version caches profiles by tag; the per-fact parity assertion against
`frozen_ags_rerank` is what keeps it honest.

Feedback is symbolic only. The spec reports the LLM verification layer as disabled
(it measured at base rate), so `AgsSeqConfig.llm_feedback_enabled` exists to be asserted
False rather than to be switched on.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ags_frozen_grounding import (
    FrozenAgsConfig,
    frozen_ags_rankings,
    frozen_ags_rerank,
    range_normalize,
    sample_hypotheses,
)
from ags_symbolic_agreement import (
    agree_with_profile,
    parse_candidate_symbolic_profile,
    symbolic_feedback_from_candidates,
    utility_from_rank,
)
from run_ags_component_validation import render_definition
from run_ags_coverage_pilot import compact_candidates
from run_fintagging_grounding_baseline import (
    DIMENSIONS,
    OPERATOR_LIBRARY,
    Example,
    TaxonomyRetriever,
    build_operator_initial_messages,
    build_operator_revision_messages,
    build_prompt_under_query_budget,
    finalize_candidate_record,
    first_gold_rank,
    fuse_round_candidates,
    llm_call_record,
    metric_row,
    neighborhood_novelty,
    normalize_space,
    normalize_tag,
    parse_hypothesis,
    tokenize,
)


AGS_SEQ_QUERY_MODE = "ags_seq"
AGS_SEQ_RANDOM_QUERY_MODE = "ags_seq_random"
AGS_SEQ_QUERY_MODES = (AGS_SEQ_QUERY_MODE, AGS_SEQ_RANDOM_QUERY_MODE)

DIMENSION_OPERATOR = {
    "FAMILY": "O_refine_family",
    "ROLE": "O_refine_role",
    "EVENT": "O_refine_event",
    "QUALIFIER": "O_refine_qualifier",
    "SCOPE": "O_refine_scope",
    "TEMPORAL": "O_refine_temporal",
}
CHANGE_STRATEGY_OPERATOR = "O_change_strategy"
PERTURB_OPERATOR = "O_perturb"
OPERATORS = tuple(DIMENSION_OPERATOR[dimension] for dimension in DIMENSIONS) + (
    CHANGE_STRATEGY_OPERATOR,
    PERTURB_OPERATOR,
)

# Directive priority when the slate exceeds L. REFINE acts on a contradicted dimension and
# is the most specific evidence the feedback stage produces; PERTURB is the least, but it
# is never crowded out (see admissible_slate) so the slate always holds one directive that
# is admissible under any feedback.
MODE_PRIORITY = {"REFINE": 0, "CHANGE_STRATEGY": 1, "BRANCH": 2, "PERTURB": 3}

# Retained psi block, measured on the 250-instance
# diagnostic logs (spec section 4). `is_text` merges into `is_table` (complementary
# indicators, rank-deficient against `bias`); `is_percent` drops against `is_monetary`
# (r = -0.92). Condition number on the diagnostic stream falls from 20,518 (11 features)
# to 1,485 on the 8 features that carry variance there -- `is_table` is constant in that
# tabular-only log and is retained on structural grounds for the two-modality test run.
DEFAULT_PSI_FEATURES = (
    "bias",
    "is_table",
    "is_monetary",
    "is_shares",
    "D_plus_count",
    "D_minus_count",
    "D_question_count",
    "structural_mismatch_g",
    "neighborhood_novelty_n",
)

STOP_BUDGET = "budget_exhausted"
STOP_NO_DIRECTIVE = "no_admissible_directive"
STOP_GATE = "novelty_gate_exhaustion"
STOP_REVISION_FAILED = "revision_parse_failed"


@dataclass(frozen=True)
class AgsSeqConfig:
    """Sequential-arm configuration. `frozen` is the AGS config and stays frozen."""

    frozen: FrozenAgsConfig = field(default_factory=FrozenAgsConfig)
    max_rounds: int = 4  # B, counting AGS's round one
    slate_limit: int = 6  # L
    feedback_top_m: int = 10  # M
    cluster_scan_depth: int = 60  # candidates scanned for cluster representatives
    # Section 3: the gate is off by default. Leaving it on reports the negative result on
    # an arm that never spent its own budget.
    novelty_gate: bool = False
    novelty_threshold: float = 0.02
    max_gate_rejections_per_round: int = 1
    reward_alpha: float = 0.5  # alpha on delta_y vs the replay contrast
    posterior_ridge: float = 1.0  # lambda
    posterior_sigma: float = 1.0  # sigma
    posterior_nu: float = 0.75  # nu, Thompson exploration scale
    posterior_forgetting: float = 0.995  # zeta
    psi_features: tuple[str, ...] = DEFAULT_PSI_FEATURES
    llm_feedback_enabled: bool = False
    seed: int = 20260724


# ---------------------------------------------------------------------------
# consolidation (AGS's, with a profile cache)
# ---------------------------------------------------------------------------

_PROFILE_CACHE: dict[str, dict[str, Any]] = {}


def reset_profile_cache() -> None:
    _PROFILE_CACHE.clear()


def _profile_for(candidate: dict[str, Any], normalization_map: dict[str, Any]) -> dict[str, Any]:
    """Symbolic profile of a taxonomy concept. Depends only on the concept's own text, so
    it is cached across facts for the life of the process."""
    tag = normalize_tag(candidate.get("tag", ""))
    profile = _PROFILE_CACHE.get(tag)
    if profile is None:
        profile = parse_candidate_symbolic_profile(candidate, normalization_map)
        _PROFILE_CACHE[tag] = profile
    return profile


def consensus_scores(
    candidates: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    normalization_map: dict[str, Any],
) -> dict[str, float]:
    """Mean agree() over the hypothesis set, per candidate.

    Rounding matches FactContext.consensus_over exactly (agree() rounds to 6, the mean
    rounds to 6); the parity assertion depends on it.
    """
    if not hypotheses:
        return {normalize_tag(candidate.get("tag", "")): 0.0 for candidate in candidates}
    scores: dict[str, float] = {}
    for candidate in candidates:
        tag = normalize_tag(candidate.get("tag", ""))
        if tag in scores:
            continue
        profile = _profile_for(candidate, normalization_map)
        per_hypothesis = [
            agree_with_profile(
                candidate,
                hypothesis.get("dimensions", hypothesis),
                normalization_map,
                profile,
            ).score
            for hypothesis in hypotheses
        ]
        scores[tag] = round(sum(per_hypothesis) / len(per_hypothesis), 6)
    return scores


def consolidate(
    rounds: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    normalization_map: dict[str, Any],
    cfg: AgsSeqConfig,
    top_k: int | None,
) -> list[dict[str, Any]]:
    """AGS's consolidation over the accumulated pool U.

    sum-RRF (kappa) over every rendering ranking -> range-normalize -> + beta*consensus.
    `top_k=None` leaves the pool untruncated, which is what the delayed-learning utility
    is defined over (spec section 4); the delivered ranking uses the frozen K.
    """
    fused = fuse_round_candidates(rounds, top_k, cfg.frozen.rrf_kappa)
    if not fused:
        return []
    rrf_by_tag = {normalize_tag(candidate["tag"]): float(candidate.get("rrf_score", 0.0)) for candidate in fused}
    normed = range_normalize(rrf_by_tag)
    consensus = consensus_scores(fused, hypotheses, normalization_map)
    final_scores = {
        tag: normed.get(tag, 0.0) + cfg.frozen.rerank_beta * consensus.get(tag, 0.0) for tag in normed
    }
    order = sorted(final_scores, key=lambda tag: (-final_scores[tag], tag))
    if top_k is not None and top_k > 0:
        order = order[:top_k]
    candidate_by_tag = {normalize_tag(candidate["tag"]): candidate for candidate in fused}
    ranked: list[dict[str, Any]] = []
    for rank, tag in enumerate(order, start=1):
        candidate = dict(candidate_by_tag[tag])
        candidate["rank"] = rank
        candidate["ags_seq_rrf_normalized"] = round(normed.get(tag, 0.0), 6)
        candidate["ags_seq_agree_consensus"] = round(consensus.get(tag, 0.0), 6)
        candidate["ags_seq_final_score"] = round(final_scores[tag], 6)
        ranked.append(candidate)
    return ranked


def assert_round_one_parity(
    rounds: list[dict[str, Any]],
    example: Example,
    hypotheses: list[dict[str, Any]],
    normalization_map: dict[str, Any],
    cfg: AgsSeqConfig,
) -> dict[str, Any]:
    """Spec section 1: round one of a sequential arm must equal AGS's final ranking.

    The reference side runs the frozen module's own rerank over the same round-one
    retrievals, so this checks the sequential pool bookkeeping and the cached
    consolidation against the shipped AGS implementation, fact by fact.
    """
    reference, _ = frozen_ags_rerank(rounds, example, hypotheses, normalization_map, cfg.frozen)
    produced = consolidate(rounds, hypotheses, normalization_map, cfg, cfg.frozen.top_k)
    reference_tags = [normalize_tag(candidate["tag"]) for candidate in reference]
    produced_tags = [normalize_tag(candidate["tag"]) for candidate in produced]
    if reference_tags != produced_tags:
        first_divergence = next(
            (
                index
                for index, (left, right) in enumerate(zip(reference_tags, produced_tags))
                if left != right
            ),
            min(len(reference_tags), len(produced_tags)),
        )
        raise AssertionError(
            "ags_seq round-one candidate set diverged from frozen AGS for fact "
            f"{example.example_idx}: {len(reference_tags)} vs {len(produced_tags)} candidates, "
            f"first divergence at rank {first_divergence + 1} "
            f"({reference_tags[first_divergence:first_divergence + 3]} vs "
            f"{produced_tags[first_divergence:first_divergence + 3]}). The round-1-vs-full "
            "column is meaningless while this holds."
        )
    return {"round1_parity_ok": True, "round1_candidates": len(produced_tags)}


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------


def cluster_representatives(
    ranking: list[dict[str, Any]],
    normalization_map: dict[str, Any],
    top_m: int,
    scan_depth: int,
) -> list[dict[str, Any]]:
    """Top-M cluster representatives under the symbolic-dimension profile (spec 2.1).

    The raw top 10 are lexical near-duplicates and carry little contrast, so candidates are
    grouped by their resolved category signature and the best-ranked member of each group
    is taken. If the scanned window holds fewer than M groups, the remaining slots fall
    back to rank order, which keeps M stable across facts.
    """
    window = ranking[: max(scan_depth, top_m)]
    representatives: list[dict[str, Any]] = []
    seen_signatures: set[tuple[Any, ...]] = set()
    for candidate in window:
        profile = _profile_for(candidate, normalization_map)
        signature = tuple(
            (dimension, tuple(profile["dimensions"].get(dimension, {}).get("categories", [])))
            for dimension in sorted(profile.get("dimensions", {}))
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        representatives.append(candidate)
        if len(representatives) >= top_m:
            return representatives
    chosen = {normalize_tag(candidate.get("tag", "")) for candidate in representatives}
    for candidate in window:
        if len(representatives) >= top_m:
            break
        if normalize_tag(candidate.get("tag", "")) not in chosen:
            representatives.append(candidate)
            chosen.add(normalize_tag(candidate.get("tag", "")))
    return representatives


def episode_feedback(
    hypothesis: dict[str, Any],
    representatives: list[dict[str, Any]],
    normalization_map: dict[str, Any],
    cfg: AgsSeqConfig,
) -> dict[str, Any]:
    """Symbolic verdicts over the neighborhood. Never sees the gold concept."""
    if cfg.llm_feedback_enabled:
        raise AssertionError(
            "ags_seq runs symbolic feedback only; the LLM verification layer measured at "
            "base rate and the paper reports it disabled (spec 2.1)."
        )
    feedback = symbolic_feedback_from_candidates(
        hypothesis.get("dimensions", {}),
        representatives,
        top_m=len(representatives) or cfg.feedback_top_m,
        normalization_map=normalization_map,
    )
    feedback["representative_tags"] = [
        normalize_tag(candidate.get("tag", "")) for candidate in representatives
    ]
    return feedback


def psi_vector(
    example: Example,
    feedback: dict[str, Any],
    novelty: float,
    feature_names: tuple[str, ...],
) -> tuple[dict[str, float], np.ndarray]:
    """Symbolic context only -- no gold, no candidate identities."""
    datatype = (example.entity_type or "").lower()
    available = {
        "bias": 1.0,
        "is_table": 1.0 if example.input_type == "table" else 0.0,
        "is_text": 1.0 if example.input_type == "text" else 0.0,
        "is_monetary": 1.0 if "monetary" in datatype else 0.0,
        "is_shares": 1.0 if "share" in datatype else 0.0,
        "is_percent": 1.0 if "percent" in datatype or "pure" in datatype else 0.0,
        "D_plus_count": float(len(feedback.get("supported_dimensions", []))),
        "D_minus_count": float(len(feedback.get("contradicted_dimensions", []))),
        "D_question_count": float(len(feedback.get("unresolved_dimensions", []))),
        "structural_mismatch_g": 1.0
        if bool((feedback.get("structural_mismatch") or {}).get("is_mismatch"))
        else 0.0,
        "neighborhood_novelty_n": float(novelty),
    }
    missing = [name for name in feature_names if name not in available]
    if missing:
        raise KeyError(f"Unknown psi feature(s) {missing}; available: {sorted(available)}")
    named = {name: available[name] for name in feature_names}
    return named, np.asarray([named[name] for name in feature_names], dtype=float)


# ---------------------------------------------------------------------------
# controller
# ---------------------------------------------------------------------------


def admissible_slate(feedback: dict[str, Any], cfg: AgsSeqConfig) -> list[dict[str, Any]]:
    """At most L atomic directives, one per operator, admissibility from feedback alone.

    REFINE         one dimension in D_minus
    BRANCH         one dimension in D_question, preserving D_plus
    CHANGESTRATEGY the interpretation operator, when g fires
    PERTURB        no dimension; re-render under stochastic decoding
    """
    supported = [normalize_space(dim).upper() for dim in feedback.get("supported_dimensions", [])]
    contradicted = [normalize_space(dim).upper() for dim in feedback.get("contradicted_dimensions", [])]
    unresolved = [normalize_space(dim).upper() for dim in feedback.get("unresolved_dimensions", [])]
    mismatch = bool((feedback.get("structural_mismatch") or {}).get("is_mismatch"))

    proposals: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        if dimension in contradicted:
            proposals.append(
                {"mode": "REFINE", "operator": DIMENSION_OPERATOR[dimension], "target_dimension": dimension}
            )
    if mismatch:
        proposals.append(
            {"mode": "CHANGE_STRATEGY", "operator": CHANGE_STRATEGY_OPERATOR, "target_dimension": ""}
        )
    for dimension in DIMENSIONS:
        if dimension in unresolved and dimension not in contradicted:
            proposals.append(
                {"mode": "BRANCH", "operator": DIMENSION_OPERATOR[dimension], "target_dimension": dimension}
            )
    proposals.append({"mode": "PERTURB", "operator": PERTURB_OPERATOR, "target_dimension": ""})

    # One directive per operator: a dimension reaching both REFINE and BRANCH keeps REFINE.
    slate: list[dict[str, Any]] = []
    claimed: set[str] = set()
    for proposal in sorted(proposals, key=lambda item: MODE_PRIORITY[item["mode"]]):
        if proposal["operator"] in claimed:
            continue
        claimed.add(proposal["operator"])
        proposal = dict(proposal)
        proposal["preserve"] = list(supported)
        slate.append(proposal)

    if len(slate) > cfg.slate_limit:
        kept = slate[: cfg.slate_limit]
        if not any(item["operator"] == PERTURB_OPERATOR for item in kept):
            kept[-1] = next(item for item in slate if item["operator"] == PERTURB_OPERATOR)
        slate = kept
    return slate


def directive_payload(item: dict[str, Any], feedback: dict[str, Any]) -> dict[str, Any]:
    """(mode, operator, target_dim, patch, preserve_set)."""
    target = item.get("target_dimension", "")
    mode = item.get("mode", "PERTURB")
    if mode == "REFINE":
        patch = f"replace the {target} reading; the retrieved neighborhood contradicts it"
    elif mode == "BRANCH":
        patch = f"commit an alternative {target} reading that the evidence supports, or leave it UNRESOLVED"
    elif mode == "CHANGE_STRATEGY":
        patch = (
            "switch the interpretation operator; the neighborhood contradicts the structural "
            "reading, not one dimension"
        )
    else:
        patch = "keep every dimension; restate the retrieval surface"
    return {
        "mode": mode,
        "operator": item.get("operator", PERTURB_OPERATOR),
        "target_dimension": target,
        "semantic_patch": patch,
        "preserve": list(item.get("preserve", feedback.get("supported_dimensions", []))),
        "rationale": "sequential controller directive",
    }


def select_directive(
    arm: str,
    slate: list[dict[str, Any]],
    bank: "SequentialPosteriorBank",
    psi: np.ndarray,
    rng: random.Random,
    context_id: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, float]]:
    """The only difference between the two arms.

    Both compute the posterior scores; `ags_seq_random` simply does not select on them.
    The runner-up is frozen here, at decision time, for the counterfactual replay.
    """
    scores = {
        item["operator"]: bank.sample_score(item["operator"], psi, exclude_context=context_id)
        for item in slate
    }
    if arm == AGS_SEQ_RANDOM_QUERY_MODE:
        selected = slate[rng.randrange(len(slate))]
    else:
        selected = max(slate, key=lambda item: (scores[item["operator"]], -MODE_PRIORITY[item["mode"]]))
    remainder = [item for item in slate if item is not selected]
    runner_up = (
        max(remainder, key=lambda item: (scores[item["operator"]], -MODE_PRIORITY[item["mode"]]))
        if remainder
        else None
    )
    return selected, runner_up, {operator: round(float(value), 8) for operator, value in scores.items()}


# ---------------------------------------------------------------------------
# revision
# ---------------------------------------------------------------------------


def enforce_directive(
    before: dict[str, Any],
    after: dict[str, Any],
    directive: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Apply only the authorized patch (spec 2.2: preserve_set is enforced, not suggested).

    REFINE / BRANCH   exactly one dimension may move: the target.
    CHANGE_STRATEGY   the interpretation operator moves; every dimension in preserve_set
                      is restored, the rest may be re-read under the new operator.
    PERTURB           no dimension moves; only the rendered surface is new.
    """
    mode = directive.get("mode", "PERTURB")
    target = normalize_space(directive.get("target_dimension", "")).upper()
    preserve = {normalize_space(dim).upper() for dim in directive.get("preserve", [])}

    before_dimensions = dict(before.get("dimensions", {}))
    after_dimensions = dict(after.get("dimensions", {}))
    if mode in {"REFINE", "BRANCH"}:
        authorized = {target} if target else set()
    elif mode == "CHANGE_STRATEGY":
        authorized = {dimension for dimension in DIMENSIONS if dimension not in preserve}
    else:
        authorized = set()

    reverted: list[str] = []
    enforced_dimensions: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        before_value = before_dimensions.get(dimension, "UNRESOLVED")
        after_value = after_dimensions.get(dimension, before_value)
        if dimension in authorized:
            enforced_dimensions[dimension] = after_value
            continue
        if normalize_space(after_value) != normalize_space(before_value):
            reverted.append(dimension)
        enforced_dimensions[dimension] = before_value

    revised = dict(after)
    revised["dimensions"] = enforced_dimensions
    if mode == "PERTURB":
        # A perturbation may not smuggle in a new operator either; only the surface moves.
        revised["operators"] = list(before.get("operators", ["direct_label"]))
    return revised, reverted


def _generate(
    generator: Any,
    prompt: str,
    temperature: float,
    top_p: float,
) -> str:
    """Generate at an explicit decoding setting, restoring the caller's afterwards."""
    original_temperature = generator.args.query_temperature
    original_top_p = generator.args.query_top_p
    generator.args.query_temperature = temperature
    generator.args.query_top_p = top_p
    try:
        return generator.generate_one(prompt)
    finally:
        generator.args.query_temperature = original_temperature
        generator.args.query_top_p = original_top_p


def revise_hypothesis(
    args: Any,
    generator: Any,
    example: Example,
    current: dict[str, Any],
    directive: dict[str, Any],
    cfg: AgsSeqConfig,
    call_label: str,
    force_deterministic: bool = False,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], bool]:
    """h_{t+1} = revise(x, h_t, directive), with the patch authorization enforced.

    PERTURB re-renders under stochastic decoding (that is the whole directive); every other
    mode is a controlled edit and runs deterministically. `force_deterministic` is the
    counterfactual replay, which the spec requires to decode deterministically whichever
    directive the runner-up carries, so the contrast is reproducible.
    """
    mode = directive.get("mode", "PERTURB")
    stochastic = mode == "PERTURB" and not force_deterministic
    if stochastic:
        message_builder = lambda ctx_chars: build_operator_initial_messages(example, ctx_chars)  # noqa: E731
    else:
        message_builder = lambda ctx_chars: build_operator_revision_messages(  # noqa: E731
            example, current, directive, ctx_chars
        )
    prompt, prompt_tokens, used_context_chars = build_prompt_under_query_budget(
        generator.tokenizer,
        message_builder,
        context_max_chars=args.query_context_max_chars,
        max_input_tokens=args.query_max_input_tokens,
    )
    raw_output = _generate(
        generator,
        prompt,
        temperature=cfg.frozen.temperature if stochastic else 0.0,
        top_p=getattr(args, "frozen_ags_top_p", 1.0) if stochastic else 1.0,
    )
    parsed, parse_ok = parse_hypothesis(raw_output, current.get("retrieval_query", ""))
    call = llm_call_record(
        call_label,
        raw_output=raw_output,
        prompt_tokens=prompt_tokens,
        completion_tokens=generator.count_text_tokens(raw_output),
        parse_ok=parse_ok,
        backend=generator.backend,
        model_name=generator.model_name,
        extra_fields={
            "used_context_max_chars": used_context_chars,
            "directive_mode": mode,
            "directive_operator": directive.get("operator"),
            "decoding": "stochastic" if stochastic else "deterministic",
        },
    )
    if not parse_ok:
        return dict(current), [], [call], False
    revised, reverted = enforce_directive(current, parsed, directive)
    if mode == "CHANGE_STRATEGY":
        alternatives = [
            operator for operator in OPERATOR_LIBRARY if operator not in set(current.get("operators", []))
        ]
        if alternatives and not set(revised.get("operators", [])) - set(current.get("operators", [])):
            revised["operators"] = [alternatives[0]]
    return revised, reverted, [call], True


def query_novelty(query: str, previous_queries: list[str]) -> float:
    """1 - max token Jaccard against previously issued queries.

    This is the appendix's pre-retrieval gate quantity. It is computed whether or not the
    gate is armed, so the gate-off run still reports what the gate would have done.
    """
    tokens = set(tokenize(query))
    if not tokens or not previous_queries:
        return 1.0
    overlap = 0.0
    for previous in previous_queries:
        other = set(tokenize(previous))
        union = tokens | other
        if union:
            overlap = max(overlap, len(tokens & other) / len(union))
    return round(1.0 - overlap, 6)


def retrieve_for_hypothesis(
    retriever: TaxonomyRetriever,
    example: Example,
    hypothesis: dict[str, Any],
    cfg: AgsSeqConfig,
    round_offset: int,
) -> list[dict[str, Any]]:
    """Render and retrieve exactly as AGS does, numbered to continue the pool."""
    rounds = frozen_ags_rankings(retriever, example, [dict(hypothesis, hypothesis_idx=0)], cfg.frozen)
    for index, round_record in enumerate(rounds, start=1):
        round_record["round"] = round_offset + index
    return rounds


# ---------------------------------------------------------------------------
# delayed learning
# ---------------------------------------------------------------------------


class SequentialPosteriorBank:
    """Per-operator ridge posterior with forgetting, over symbolic psi only.

        A_o = lambda*I + sum_i zeta^(delta_i) psi_i psi_i^T
        b_o = sum_i zeta^(delta_i) psi_i r_i
        mu_o = A_o^-1 b_o,  Sigma_o = sigma^2 A_o^-1

    Records from the current instance's source context are excluded at read time. Without
    that, the 21 facts per table in this benchmark make within-context leakage look like
    online learning. Stored records carry symbolic features, the operator and the credit --
    never a gold identity.
    """

    def __init__(self, operators: tuple[str, ...], cfg: AgsSeqConfig) -> None:
        self.operators = tuple(operators)
        self.cfg = cfg
        self.dim = len(cfg.psi_features)
        self.rng = np.random.default_rng(cfg.seed)
        self.step = 0
        self._psi: dict[str, list[np.ndarray]] = {operator: [] for operator in self.operators}
        self._reward: dict[str, list[float]] = {operator: [] for operator in self.operators}
        self._step: dict[str, list[int]] = {operator: [] for operator in self.operators}
        self._context: dict[str, list[str]] = {operator: [] for operator in self.operators}

    # -- writes ------------------------------------------------------------
    def update(self, operator: str, psi: np.ndarray, reward: float, context_id: Any) -> None:
        if operator not in self._psi:
            raise KeyError(f"Unknown operator {operator}")
        self._psi[operator].append(np.asarray(psi, dtype=float))
        self._reward[operator].append(float(reward))
        self._step[operator].append(self.step)
        self._context[operator].append(str(context_id))
        self.step += 1

    def ingest_record(self, record: dict[str, Any]) -> None:
        """Replay a completed record's credits, so --resume reconstructs the same bank."""
        context_id = record.get("context_id")
        for round_record in record.get("ags_seq_rounds", []):
            operator = round_record.get("selected_operator")
            psi_values = round_record.get("psi_values")
            reward = round_record.get("reward")
            if operator in self._psi and isinstance(psi_values, list) and reward is not None:
                self.update(operator, np.asarray(psi_values, dtype=float), float(reward), context_id)

    # -- reads -------------------------------------------------------------
    def _moments(self, operator: str, exclude_context: Any) -> tuple[np.ndarray, np.ndarray]:
        a = np.eye(self.dim) * self.cfg.posterior_ridge
        b = np.zeros(self.dim)
        rows = self._psi[operator]
        if rows:
            excluded = str(exclude_context)
            mask = [index for index, ctx in enumerate(self._context[operator]) if ctx != excluded]
            if mask:
                psi_matrix = np.stack([rows[index] for index in mask])
                rewards = np.asarray([self._reward[operator][index] for index in mask], dtype=float)
                ages = self.step - np.asarray([self._step[operator][index] for index in mask], dtype=float)
                weights = np.power(self.cfg.posterior_forgetting, ages)
                a += (psi_matrix * weights[:, None]).T @ psi_matrix
                b += (psi_matrix * (weights * rewards)[:, None]).sum(axis=0)
        return a, b

    def posterior(self, operator: str, exclude_context: Any = None) -> tuple[np.ndarray, np.ndarray]:
        a, b = self._moments(operator, exclude_context)
        inverse = np.linalg.inv(a)
        return inverse @ b, (self.cfg.posterior_sigma ** 2) * inverse

    def sample_score(self, operator: str, psi: np.ndarray, exclude_context: Any = None) -> float:
        mu, sigma = self.posterior(operator, exclude_context)
        theta = self.rng.multivariate_normal(mu, (self.cfg.posterior_nu ** 2) * sigma)
        return float(theta @ psi)

    def mean_score(self, operator: str, psi: np.ndarray, exclude_context: Any = None) -> float:
        mu, _ = self.posterior(operator, exclude_context)
        return float(mu @ psi)

    def snapshot(self) -> dict[str, Any]:
        rows = {}
        for operator in self.operators:
            a, _ = self._moments(operator, exclude_context=None)
            mu, _ = self.posterior(operator)
            rows[operator] = {
                "n_records": len(self._psi[operator]),
                "condition_number": round(float(np.linalg.cond(a)), 6),
                "mu": [round(float(value), 8) for value in mu.tolist()],
            }
        return {"step": self.step, "dimension": self.dim, "operators": rows}


# ---------------------------------------------------------------------------
# episode
# ---------------------------------------------------------------------------


def _utility(ranking: list[dict[str, Any]], gold_tags: list[str]) -> tuple[float, int | None]:
    rank = first_gold_rank([candidate["tag"] for candidate in ranking], gold_tags)
    return utility_from_rank(rank), rank


def build_ags_seq_method_record(
    args: Any,
    arm: str,
    generator: Any,
    retriever: TaxonomyRetriever,
    example: Example,
    normalization_map: dict[str, Any],
    bank: SequentialPosteriorBank,
    cfg: AgsSeqConfig | None = None,
) -> dict[str, Any]:
    """One episode: AGS round one, then up to B-1 controller rounds, then delayed learning.

    Nothing inside the loop reads `example.gold_tags`; the utilities, credits and posterior
    updates all happen after the episode terminates (spec section 4).
    """
    cfg = cfg or AgsSeqConfig()
    if arm not in AGS_SEQ_QUERY_MODES:
        raise ValueError(f"Unknown sequential arm {arm}; expected one of {AGS_SEQ_QUERY_MODES}")
    start_time = time.monotonic()
    rng = random.Random(f"{cfg.seed}:{arm}:{example.example_idx}")

    # --- round one: the frozen AGS code path, unchanged -----------------------
    hypotheses, calls, used_fallback = sample_hypotheses(generator, args, example, cfg.frozen)
    pool = frozen_ags_rankings(retriever, example, hypotheses, cfg.frozen)
    pool_hypotheses = list(hypotheses)
    parity = assert_round_one_parity(pool, example, hypotheses, normalization_map, cfg)
    ranking = consolidate(pool, pool_hypotheses, normalization_map, cfg, cfg.frozen.top_k)
    round1_ranking = ranking
    round1_union = consolidate(pool, pool_hypotheses, normalization_map, cfg, None)

    current_hypothesis = dict(hypotheses[0])
    unions = [round1_union]  # untruncated consolidated ranking after each realized round
    replay_unions: list[list[dict[str, Any]] | None] = [None]
    neighborhoods: list[dict[str, Any]] = [{"candidates": ranking}]
    issued_queries: list[str] = [round_record.get("grounding", "") for round_record in pool]
    round_records: list[dict[str, Any]] = []
    gate_rejections = 0
    stop_reason = STOP_BUDGET
    realized_rounds = 1

    for round_idx in range(2, cfg.max_rounds + 1):
        representatives = cluster_representatives(
            ranking, normalization_map, cfg.feedback_top_m, cfg.cluster_scan_depth
        )
        feedback = episode_feedback(current_hypothesis, representatives, normalization_map, cfg)
        novelty = neighborhood_novelty(ranking, neighborhoods[:-1])
        psi_named, psi = psi_vector(example, feedback, novelty, cfg.psi_features)

        slate = admissible_slate(feedback, cfg)
        if not slate:
            stop_reason = STOP_NO_DIRECTIVE
            break
        selected, runner_up, selection_scores = select_directive(
            arm, slate, bank, psi, rng, example.context_id
        )
        directive = directive_payload(selected, feedback)
        revised, reverted, revision_calls, parse_ok = revise_hypothesis(
            args, generator, example, current_hypothesis, directive, cfg, "ags_seq_revision"
        )
        calls.extend(revision_calls)
        if not parse_ok:
            stop_reason = STOP_REVISION_FAILED
            break

        # Section 3: the gate is measured either way and armed only when configured. An
        # atomic single-dimension edit barely moves the rendered query, so with the gate on
        # it fires early and the arm never spends its own budget.
        revised_query = render_definition(revised)
        revision_query_novelty = query_novelty(revised_query, issued_queries)
        gate_would_reject = revision_query_novelty < cfg.novelty_threshold
        if cfg.novelty_gate and gate_would_reject:
            gate_rejections += 1
            if gate_rejections >= cfg.max_gate_rejections_per_round:
                stop_reason = STOP_GATE
                break

        new_rounds = retrieve_for_hypothesis(retriever, example, revised, cfg, len(pool))
        selected_novelty = neighborhood_novelty(
            [candidate for round_record in new_rounds for candidate in round_record["candidates"]],
            pool,
        )

        # --- counterfactual replay, frozen at decision time -------------------
        # The slate and the runner-up directive are fixed above; the replay revision is
        # generated here because it is deterministic and gold-independent, and only its
        # utility is read after the episode ends. Nothing in this block sees the gold.
        replay_union = None
        replay_directive = None
        replay_reverted: list[str] = []
        if runner_up is not None:
            replay_directive = directive_payload(runner_up, feedback)
            replay_hypothesis, replay_reverted, replay_calls, replay_ok = revise_hypothesis(
                args,
                generator,
                example,
                current_hypothesis,
                replay_directive,
                cfg,
                "ags_seq_replay",
                force_deterministic=True,
            )
            calls.extend(replay_calls)
            if replay_ok:
                replay_rounds = retrieve_for_hypothesis(
                    retriever, example, replay_hypothesis, cfg, len(pool)
                )
                replay_union = consolidate(
                    pool + replay_rounds,
                    pool_hypotheses + [replay_hypothesis],
                    normalization_map,
                    cfg,
                    None,
                )

        pool.extend(new_rounds)
        pool_hypotheses.append(revised)
        current_hypothesis = revised
        ranking = consolidate(pool, pool_hypotheses, normalization_map, cfg, cfg.frozen.top_k)
        unions.append(consolidate(pool, pool_hypotheses, normalization_map, cfg, None))
        replay_unions.append(replay_union)
        neighborhoods.append({"candidates": ranking})
        issued_queries.append(revised_query)
        realized_rounds = round_idx

        round_records.append(
            {
                "round_idx": round_idx,
                "psi": psi_named,
                "psi_values": [round(float(value), 8) for value in psi.tolist()],
                "slate": [
                    {key: item[key] for key in ("mode", "operator", "target_dimension")} for item in slate
                ],
                "selected_operator": selected["operator"],
                "selected_mode": selected["mode"],
                "runner_up_operator": runner_up["operator"] if runner_up else None,
                "runner_up_mode": runner_up["mode"] if runner_up else None,
                "selection_scores": selection_scores,
                "directive": directive,
                "replay_directive": replay_directive,
                "enforced_reverted_dimensions": reverted,
                "replay_enforced_reverted_dimensions": replay_reverted,
                "D_plus_count": int(len(feedback.get("supported_dimensions", []))),
                "D_minus_count": int(len(feedback.get("contradicted_dimensions", []))),
                "D_question_count": int(len(feedback.get("unresolved_dimensions", []))),
                "D_plus": list(feedback.get("supported_dimensions", [])),
                "D_minus": list(feedback.get("contradicted_dimensions", [])),
                "D_question": list(feedback.get("unresolved_dimensions", [])),
                "structural_mismatch_g": bool(
                    (feedback.get("structural_mismatch") or {}).get("is_mismatch")
                ),
                "neighborhood_novelty_n": round(float(novelty), 8),
                "selected_neighborhood_novelty": round(float(selected_novelty), 8),
                "revision_query_novelty": round(float(revision_query_novelty), 8),
                "gate_would_reject": bool(gate_would_reject),
                "feedback_representative_tags": feedback.get("representative_tags", []),
                "dimension_verdicts": feedback.get("dimension_verdicts", []),
                "revised_hypothesis": revised,
                "gate_rejections_this_round": gate_rejections if cfg.novelty_gate else 0,
                "candidate_list": [candidate["tag"] for candidate in ranking],
            }
        )

    # --- delayed learning: gold is revealed only here -------------------------
    for index, round_record in enumerate(round_records, start=1):
        utility_before, rank_before = _utility(unions[index - 1], example.gold_tags)
        utility_after, rank_after = _utility(unions[index], example.gold_tags)
        delta_y = utility_after - utility_before
        replay_union = replay_unions[index]
        if replay_union is None:
            utility_replay, rank_replay = utility_before, rank_before
        else:
            utility_replay, rank_replay = _utility(replay_union, example.gold_tags)
        delta_replay = utility_after - utility_replay
        reward = cfg.reward_alpha * delta_y + (1.0 - cfg.reward_alpha) * delta_replay
        round_record.update(
            {
                "gold_in_union_before": rank_before is not None,
                "gold_in_union_after": rank_after is not None,
                "rank_before": rank_before,
                "rank_after": rank_after,
                "rank_replay": rank_replay,
                "utility_before": round(utility_before, 8),
                "utility_after": round(utility_after, 8),
                "utility_replay": round(utility_replay, 8),
                "delta_y": round(delta_y, 8),
                "delta_replay": round(delta_replay, 8),
                "reward": round(reward, 8),
            }
        )
        bank.update(round_record["selected_operator"], np.asarray(round_record["psi_values"]), reward, example.context_id)

    prompt_tokens = sum(int(call.get("prompt_tokens", 0) or 0) for call in calls)
    completion_tokens = sum(int(call.get("completion_tokens", 0) or 0) for call in calls)
    # The pool keeps full candidate dicts (the symbolic profile reads documentation text),
    # but an episode issues up to ten retrievals and the stored trace only needs the
    # ranking, so the persisted rounds carry compacted candidates.
    stored_rounds = [
        dict(round_record, candidates=compact_candidates(round_record["candidates"]))
        for round_record in pool
    ]
    record = finalize_candidate_record(
        example,
        query_mode=arm,
        rounds=stored_rounds,
        top_k=cfg.frozen.top_k,
        rrf_kappa=cfg.frozen.rrf_kappa,
        total_llm_calls=len(calls),
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        wall_time=time.monotonic() - start_time,
        extra_fields={
            "ags_seq_arm": arm,
            "ags_seq_config": {
                "max_rounds": cfg.max_rounds,
                "slate_limit": cfg.slate_limit,
                "feedback_top_m": cfg.feedback_top_m,
                "novelty_gate": cfg.novelty_gate,
                "novelty_threshold": cfg.novelty_threshold,
                "reward_alpha": cfg.reward_alpha,
                "posterior_ridge": cfg.posterior_ridge,
                "posterior_sigma": cfg.posterior_sigma,
                "posterior_nu": cfg.posterior_nu,
                "posterior_forgetting": cfg.posterior_forgetting,
                "psi_features": list(cfg.psi_features),
                "llm_feedback_enabled": cfg.llm_feedback_enabled,
                "frozen": {
                    "hypotheses": cfg.frozen.hypotheses,
                    "top_k": cfg.frozen.top_k,
                    "rrf_kappa": cfg.frozen.rrf_kappa,
                    "rerank_beta": cfg.frozen.rerank_beta,
                    "label_coverage_weight": cfg.frozen.label_coverage_weight,
                    "temperature": cfg.frozen.temperature,
                    "dual_rendering_modalities": list(cfg.frozen.dual_rendering_modalities),
                },
            },
            "ags_seq_hypotheses": pool_hypotheses,
            "ags_seq_used_fallback": used_fallback,
            "ags_seq_round1_parity": parity,
            "ags_seq_rounds": round_records,
            "ags_seq_llm_calls": calls,
        },
    )

    final_tags = [candidate["tag"] for candidate in ranking]
    round1_tags = [candidate["tag"] for candidate in round1_ranking]
    final_metrics = metric_row(final_tags, example.gold_tags, (10, 50, cfg.frozen.top_k))
    record["candidates"] = ranking
    record["final_candidates"] = ranking
    record["candidate_union_tags"] = final_tags
    record["retrieval_metrics"] = final_metrics
    record["gold_rank"] = final_metrics.get("rank")
    record["search_coverage"] = any(
        normalize_tag(tag) in {normalize_tag(candidate) for candidate in final_tags}
        for tag in example.gold_tags
    )
    record["total_retrieval_calls"] = len(pool)
    record["round1_candidates"] = round1_tags
    record["round1_retrieval_metrics"] = metric_row(
        round1_tags, example.gold_tags, (10, 50, cfg.frozen.top_k)
    )
    record["round1_rank_gold"] = record["round1_retrieval_metrics"].get("rank")
    record["final_rank_gold"] = final_metrics.get("rank")
    record["realized_rounds"] = realized_rounds
    record["stop_reason"] = stop_reason
    return record
