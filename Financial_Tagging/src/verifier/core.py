#!/usr/bin/env python3
"""Table 5 ablation core (ags_table5_ablation_spec.md section 1).

One `evaluate(fact, config, normalization_map)` function. Every row in the table is a call
to it with a different `AblationConfig` -- rationale stated in the spec: two independently
written aggregation routines already produced inconsistent stage-decomposition numbers once
in this project (Appendix O). One function, flags.

This package is deliberately independent of run_fintagging_grounding_baseline.py/.sh: those
files are live dependencies of other experiments (the AGS+Seq sequential-arm jobs run
against them at the time this was written). Everything here only *imports* pure, read-only
functions from the shared pipeline -- fusion/scoring primitives, tokenization, the symbolic
agreement layer -- and writes to its own `runs_ags_table5_ablation/` output tree. Nothing in
this package edits a file the rest of the pipeline depends on.

Nine of eleven rows are pure re-analysis of already-logged AGS candidate lists: no new
generation, no new retrieval (see ags_table5_ablation/data_prep.py for where those lists
come from). Only `llm_verifier` (row 3.9) and the index ablation (row 3.10, a separate panel
handled in run_index_ablation.py) touch anything beyond this module's own arithmetic.
"""

from __future__ import annotations
# --- resolve local packages regardless of this file's depth in the tree ---
import sys as _sys, pathlib as _pathlib
for _p in _pathlib.Path(__file__).resolve().parents:
    if (_p / "src" / "run_fintagging_grounding_baseline.py").exists():
        _sys.path.insert(0, str(_p / "src"))
        _sys.path.insert(0, str(_p / "analysis"))
        FHS_ROOT = _p
        break
# -------------------------------------------------------------------------

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from ags_frozen_grounding import range_normalize  # noqa: E402
from ags_sequential_arms import cluster_representatives, consensus_scores  # noqa: E402
from ags_symbolic_agreement import (  # noqa: E402
    canonical_hypothesis_dimensions,
    dimension_agreement_verdict,
    is_unresolved,
    parse_candidate_symbolic_profile,
    symbolic_feedback_from_candidates,
)
from run_fintagging_grounding_baseline import normalize_tag  # noqa: E402


# 20 is here because the deployed listwise reranker sees exactly the top 20, so the
# verifier/reranker interaction table is stated at that depth. Adding a depth only adds a
# recall_at_20 key; callers name the metrics they write, so existing CSV schemas are unchanged.
TOP_KS = (10, 20, 50, 200)
# THE DEPLOYED CONFIGURATION, changed 2026-07-30: the verifier is asked about, and scored over,
# every dimension the generator emits. The earlier three-dimension set (FAMILY/ROLE/EVENT) split
# the schema into "text-decidable" and "metadata-determined" -- a per-domain human judgement that
# does not transfer: in ICD-10-CM laterality and encounter type are written into the code's own
# text, so that split would be wrong there. Asking and scoring all six needs no such decision.
# Every number in the paper is regenerated under this setting; ask-3 artifacts are superseded.
LLM_VERIFIER_DIMENSIONS_DEFAULT = ("FAMILY", "ROLE", "EVENT", "QUALIFIER", "SCOPE", "TEMPORAL")

# Which verifier supplies the rerank term. "auto" preserves the original behaviour --
# hybrid when verdicts are attached, deterministic when they are not -- so every row
# written before verifier_mode existed evaluates identically.
VERIFIER_MODES = ("auto", "deterministic", "hybrid", "llm_drop", "llm_strict", "det_window",
                  "llm_neutral", "llm_varweight")


@dataclass(frozen=True)
class AblationConfig:
    """One row of Table 5. Defaults reproduce `AGS (full)` (section 3.1)."""

    name: str = "AGS (full)"
    n_hypotheses: int = 2  # J
    kept_hypothesis_idx: int | None = None  # required when n_hypotheses == 1 (section 3.2)
    renderings: tuple[str, ...] = ("def", "lab")
    lab_only_fallback: str | None = None  # None -> zero-recall (3.5a, recommended); "def" -> 3.5b
    fusion: str = "sum"  # "sum" | "mean"
    scaling: str = "range"  # "range" | "raw"
    beta: float = 0.6
    rrf_kappa: float = 60.0
    top_k: int = 200
    oracle_best_single: bool = False
    llm_verifier_top_m: int = 10
    llm_verifier_dimensions: tuple[str, ...] = LLM_VERIFIER_DIMENSIONS_DEFAULT
    # One weight per dimension, estimated ONCE over the whole corpus rather than inside each
    # K_v window. The per-window form (verifier_mode="llm_varweight") estimates from <=10 binary
    # verdicts and is measurably unstable: weight sd >= mean on all six dimensions, and the two
    # hypotheses of the same fact -- which see the SAME candidate window -- disagree on the
    # top-weighted dimension 45.3% of the time. A corpus-level weight keeps the property that
    # makes the idea transferable (no human picks the dimensions) without that noise. None keeps
    # equal weights, i.e. the deployed mean, so every existing call is unchanged.
    llm_verifier_dimension_weights: tuple[tuple[str, float], ...] | None = None
    verifier_mode: str = "auto"
    # What an LLM-only arm scores a candidate the verifier never saw. "zero" drives every
    # out-of-window candidate to the bottom of the rerank term, which conflates "remove the
    # deterministic verifier" with "add a top-K_v prior" -- the window is ~20 of 200
    # candidates, so that is a large implicit promotion. "mean" fills them with the mean of
    # that hypothesis's own judged scores, leaving them ordered by fused score among
    # themselves and neither promoted nor demoted as a block.
    llm_unjudged_fill: str = "zero"
    # Weight applied to a dimension the deterministic verifier marks CONTRADICTED. 1.0 is the
    # historical unweighted mean; below 1.0 discounts dimensions the hypothesis is probably
    # wrong on, which is the signal tab:verifier shows the symbolic layer is strongest at and
    # which the ranking path never consumed. Swept on dev like beta, never tuned on test.
    contradicted_dimension_weight: float = 1.0
    # The deployed pipeline truncates the fused pool to top_k BEFORE the consensus rerank
    # (run_fintagging_grounding_baseline.fuse_round_candidates), so its rerank can only
    # reorder within the top 200 and Recall@200 is fixed by fusion alone. This harness
    # originally reranked the whole fused pool and truncated afterwards, which lets consensus
    # pull a concept from outside the fused top 200 into it -- worth +0.0128 Recall@200 and
    # the reason the two disagreed. False keeps the original behaviour so the pre-existing
    # Table 5 rows are unchanged; True reproduces the deployed pipeline.
    truncate_pool_to_top_k: bool = False
    # (fact_id, hyp_idx, tag) -> {"FAMILY": bool|None, "ROLE": bool|None, "EVENT": bool|None};
    # populated by run_llm_verifier.py for row 3.9, absent (None) for every other row.
    llm_verifier_verdicts: dict[tuple[int, int, str], dict[str, Any]] | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {
            key: value
            for key, value in vars(self).items()
            if key != "llm_verifier_verdicts"
        }
        out["llm_verifier_active"] = self.llm_verifier_verdicts is not None
        return out


@dataclass
class FactRecord:
    """Everything `evaluate` needs for one fact, independent of where it was loaded from."""

    fact_id: int
    context_id: Any
    modality: str
    datatype: str
    gold_tags: list[str]
    hypotheses: dict[int, dict[str, Any]]  # hyp_idx -> {"dimensions": {...}, "query_lab": ...}
    rankings: dict[tuple[int, str], list[dict[str, Any]]]  # (hyp_idx, rendering) -> ranked candidates


def normalize_tags(tags: list[str]) -> list[str]:
    return [normalize_tag(tag) for tag in tags]


def fuse(
    rankings: list[list[dict[str, Any]]],
    kappa: float,
    fusion: str,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """sum-RRF or mean-RRF over the given rankings.

    "mean" divides by the number of rankings *containing* the candidate, not by the total
    number of rankings fused -- dividing by the total would just rescale sum-RRF by a
    constant and hide the multiplicity bonus entirely (section 3.6's trap).
    """
    contributions: dict[str, list[float]] = defaultdict(list)
    best_candidate: dict[str, dict[str, Any]] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        for candidate in ranking:
            tag = normalize_tag(candidate.get("tag", ""))
            rank = int(candidate.get("rank", 0) or 0)
            if not tag or rank <= 0:
                continue
            contributions[tag].append(1.0 / (kappa + rank))
            if tag not in best_rank or rank < best_rank[tag]:
                best_rank[tag] = rank
                best_candidate[tag] = candidate
    if fusion == "sum":
        scores = {tag: sum(values) for tag, values in contributions.items()}
    elif fusion == "mean":
        scores = {tag: sum(values) / len(values) for tag, values in contributions.items()}
    else:
        raise ValueError(f"Unknown fusion {fusion!r}; expected 'sum' or 'mean'")
    return scores, best_candidate


def hybrid_agree_score(
    candidate: dict[str, Any],
    hypothesis_dimensions: dict[str, Any],
    normalization_map: dict[str, Any],
    profile: dict[str, Any],
    llm_verdict: dict[str, Any] | None,
    llm_dimensions: tuple[str, ...],
    dimension_weights: dict[str, float] | None = None,
) -> float | None:
    """agree(), with the LLM verdict substituted on `llm_dimensions` when one was issued.

    Row 3.9's hybrid: for the M candidates the verifier actually saw, FAMILY/ROLE/EVENT come
    from the model judgment; everything else -- QUALIFIER/SCOPE/TEMPORAL always, and
    FAMILY/ROLE/EVENT for every candidate outside the top M -- stays symbolic. A verdict of
    `None` (the verifier abstained) falls back to the symbolic verdict for that dimension,
    which is what makes the firing rate visible in the aggregate: an all-abstaining verifier
    reduces exactly to plain agree().
    """
    canonical = canonical_hypothesis_dimensions(hypothesis_dimensions)
    weighted_sum = 0.0
    weight_total = 0.0
    for dimension, value in canonical.items():
        if is_unresolved(value):
            continue
        upper = dimension.upper()
        matched: bool | None = None
        if llm_verdict is not None and upper in llm_dimensions and upper in llm_verdict:
            matched = llm_verdict[upper]
        if matched is None:
            verdict = dimension_agreement_verdict(candidate, dimension, value, normalization_map, profile)
            matched = verdict["matched"]
        if matched is not None:
            # Defaults to 1.0, so with no weights supplied this is the plain unweighted mean
            # the original returned -- bit for bit, since sum(bools)/len == sum(1.0*bool)/sum(1.0).
            weight = dimension_weights.get(upper, 1.0) if dimension_weights else 1.0
            weighted_sum += weight * float(bool(matched))
            weight_total += weight
    if weight_total <= 0.0:
        return None
    return weighted_sum / weight_total


def contradicted_dimension_weights(
    hypothesis_dimensions: dict[str, Any],
    candidates: list[dict[str, Any]],
    normalization_map: dict[str, Any],
    top_m: int,
    contradicted_weight: float,
) -> dict[str, float]:
    """Per-dimension weights for agree(), from the deterministic verifier's own D- verdict.

    WHY THIS EXISTS
        agree(c,h) rewards candidates that match the hypothesis. When the hypothesis is wrong
        on a dimension, that rewards candidates sharing the wrong value. The deterministic
        verifier detects exactly this at 0.968 precision (tab:verifier), but nothing in the
        ranking path ever consulted it -- ags_frozen_grounding.py contains no reference to
        disagreement at all. So the strongest signal the symbolic layer produces was computed
        during search and then discarded before ranking.

        This routes it back in: a dimension the verifier marks CONTRADICTED (support fraction
        <= 0.25 over the assessed window, the production threshold in
        ags_symbolic_agreement.symbolic_feedback_from_candidates) is down-weighted instead of
        counted at par. Each layer then does what it measures best -- the deterministic layer
        decides which dimensions to trust, the LLM layer judges candidates on them.

    contradicted_weight=1.0 reproduces the unweighted mean exactly, so this is inert unless
    asked for.
    """
    if contradicted_weight >= 1.0:
        return {}
    feedback = symbolic_feedback_from_candidates(
        hypothesis_dimensions, candidates, top_m=top_m, normalization_map=normalization_map
    )
    return {dimension: contradicted_weight for dimension in feedback.get("contradicted_dimensions", [])}


def hybrid_consensus_scores(
    candidates: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    normalization_map: dict[str, Any],
    fact_id: int,
    hyp_indices: list[int],
    verdicts: dict[tuple[int, int, str], dict[str, Any]],
    llm_dimensions: tuple[str, ...],
    contradicted_weight: float = 1.0,
    top_m: int = 10,
) -> dict[str, float]:
    if not hypotheses:
        return {}
    # Computed once per hypothesis, not per candidate: the D- verdict is a property of the
    # hypothesis against the assessed window, so recomputing it inside the candidate loop
    # would be the same answer 200 times over.
    weights_by_hyp = {
        hyp_idx: contradicted_dimension_weights(
            hypothesis.get("dimensions", hypothesis), candidates, normalization_map, top_m, contradicted_weight
        )
        for hyp_idx, hypothesis in zip(hyp_indices, hypotheses)
    }
    scores: dict[str, float] = {}
    for candidate in candidates:
        tag = normalize_tag(candidate.get("tag", ""))
        if tag in scores:
            continue
        profile = parse_candidate_symbolic_profile(candidate, normalization_map)
        per_hypothesis = []
        for hyp_idx, hypothesis in zip(hyp_indices, hypotheses):
            llm_verdict = verdicts.get((fact_id, hyp_idx, tag))
            score = hybrid_agree_score(
                candidate,
                hypothesis.get("dimensions", hypothesis),
                normalization_map,
                profile,
                llm_verdict,
                llm_dimensions,
                weights_by_hyp.get(hyp_idx),
            )
            per_hypothesis.append(score if score is not None else 0.0)
        scores[tag] = round(sum(per_hypothesis) / len(per_hypothesis), 6)
    return scores


def truncate_fused_pool(
    scores: dict[str, float],
    best_candidate: dict[str, dict[str, Any]],
    top_k: int,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Keep the top_k candidates by fused score, as the deployed fusion does before reranking.

    Tie-break mirrors fuse_round_candidates: fused score descending, then the candidate's best
    round rank, then the tag. Only the boundary of the pool can be affected by the tie-break,
    and reproducing the deployed ordering there is the point.
    """
    if top_k <= 0 or len(scores) <= top_k:
        return scores, best_candidate
    ordered = sorted(
        scores,
        key=lambda tag: (-scores[tag], int(best_candidate[tag].get("rank", 10**9) or 10**9), tag),
    )[:top_k]
    kept = set(ordered)
    return (
        {tag: scores[tag] for tag in ordered},
        {tag: best_candidate[tag] for tag in kept},
    )


def llm_only_agree_score(
    llm_verdict: dict[str, Any] | None,
    llm_dimensions: tuple[str, ...],
    abstention: str,
    weights: dict[str, float] | None = None,
) -> float | None:
    """agree() over the LLM's own dimensions only, with no symbolic term anywhere.

    This is the "- deterministic verifier" arm: unlike hybrid_agree_score, an abstention does
    NOT fall back to the symbolic verdict, and QUALIFIER/SCOPE/TEMPORAL never enter. Two
    readings of what to do with an abstention, kept as separate arms because they are not the
    same experiment:

      abstention="drop"      average over the dimensions the verifier actually ruled on. A
                             candidate it declined to judge entirely scores None, which the
                             caller floors to 0.0 -- the same place an unseen candidate lands.
      abstention="negative"  an abstention counts as non-support. Silence is evidence against
                             the candidate, so a verifier that abstains on two of three
                             dimensions cannot reach the top of the ranking on the third.

    Candidates outside the top-M window have no verdict under either reading and score 0.0,
    which is what confines this term to the head of the ranking.
    """
    if abstention not in ("drop", "negative", "neutral"):
        raise ValueError(f"Unknown abstention {abstention!r}; expected 'drop', 'negative' or 'neutral'")
    if abstention == "neutral":
        if llm_verdict is None:
            return 0.5
        vals = [0.5 if llm_verdict.get(d) is None else float(bool(llm_verdict.get(d)))
                for d in llm_dimensions]
        return sum(vals) / len(vals) if vals else 0.5
    if llm_verdict is None:
        return None if abstention == "drop" else 0.0
    if weights is not None:
        num = den = 0.0
        for dimension in llm_dimensions:
            value = llm_verdict.get(dimension)
            if value is None:
                continue
            w = float(weights.get(dimension, 0.0))
            num += w * float(bool(value))
            den += w
        return (num / den) if den > 0 else None
    matches: list[bool] = []
    for dimension in llm_dimensions:
        value = llm_verdict.get(dimension)
        if value is None:
            if abstention == "negative":
                matches.append(False)
            continue
        matches.append(bool(value))
    if not matches:
        return None if abstention == "drop" else 0.0
    return sum(matches) / len(matches)


def det_window_consensus_scores(
    candidates: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    normalization_map: dict[str, Any],
    fact_id: int,
    hyp_indices: list[int],
    verdicts: dict[tuple[int, int, str], dict[str, Any]],
    llm_dimensions: tuple[str, ...],
    unjudged_fill: str = "mean",
) -> dict[str, float]:
    """The deterministic agree score, restricted to the SAME window the LLM verifier saw.

    WHY THIS ARM EXISTS
        The verifier ablation as originally posed is not a like-for-like comparison. The LLM
        term reaches exactly the top-K_v candidates -- 10 of 200, 5% of the pool -- and every
        other candidate receives a single constant under unjudged_fill, so their relative order
        is left at the fused ranking. The deterministic term reaches all 200 and reorders every
        one of them. Scored at top-1, that pits a narrow, low-risk edit of the head against a
        global reordering in which any misstep at depth costs recall, and the metric only ever
        looks at the head. "Hybrid vs LLM-only" therefore names a contrast the arms do not
        measure: the real difference is scope, not verdict source.

        This function equalizes scope. It takes the window from the verdicts file -- the exact
        (fact, hypothesis, tag) keys the LLM was asked about -- and scores those candidates with
        the symbolic verdict instead, mean-filling the rest exactly as llm_only_consensus_scores
        does. Against `llm_drop` it isolates the one variable the row label claims: which
        verifier issued the judgement.

    The verdicts are read for their KEYS only; no LLM judgement enters the score. Passing
    llm_verdict=None to hybrid_agree_score makes every dimension fall through to
    dimension_agreement_verdict, which is plain agree() over all six dimensions.
    """
    if not hypotheses:
        return {}
    if unjudged_fill not in ("zero", "mean"):
        raise ValueError(f"Unknown unjudged_fill {unjudged_fill!r}; expected 'zero' or 'mean'")

    tags: list[str] = []
    seen: set[str] = set()
    candidate_by_tag: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        tag = normalize_tag(candidate.get("tag", ""))
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
            candidate_by_tag[tag] = candidate

    profiles: dict[str, dict[str, Any]] = {}
    per_hyp_scores: dict[int, dict[str, float]] = {}
    for hyp_idx, hypothesis in zip(hyp_indices, hypotheses):
        judged: dict[str, float] = {}
        for tag in tags:
            # Window membership only -- the verdict's contents are deliberately not read.
            if verdicts.get((fact_id, hyp_idx, tag)) is None:
                continue
            candidate = candidate_by_tag[tag]
            if tag not in profiles:
                profiles[tag] = parse_candidate_symbolic_profile(candidate, normalization_map)
            score = hybrid_agree_score(
                candidate,
                hypothesis.get("dimensions", hypothesis),
                normalization_map,
                profiles[tag],
                None,
                llm_dimensions,
                None,
            )
            if score is not None:
                judged[tag] = score
        if unjudged_fill == "mean" and judged:
            fill = sum(judged.values()) / len(judged)
        else:
            fill = 0.0
        per_hyp_scores[hyp_idx] = {tag: judged.get(tag, fill) for tag in tags}

    return {
        tag: round(sum(per_hyp_scores[idx][tag] for idx in hyp_indices) / len(hyp_indices), 6)
        for tag in tags
    }


def llm_only_consensus_scores(
    candidates: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    fact_id: int,
    hyp_indices: list[int],
    verdicts: dict[tuple[int, int, str], dict[str, Any]],
    llm_dimensions: tuple[str, ...],
    abstention: str,
    unjudged_fill: str = "zero",
    global_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Consensus over hypotheses using llm_only_agree_score. Same shape as
    hybrid_consensus_scores -- mean over hypotheses -- so the two differ only in where the
    per-dimension verdict comes from and in what an unjudged candidate receives."""
    if not hypotheses:
        return {}
    if unjudged_fill not in ("zero", "mean"):
        raise ValueError(f"Unknown unjudged_fill {unjudged_fill!r}; expected 'zero' or 'mean'")

    tags = []
    seen: set[str] = set()
    for candidate in candidates:
        tag = normalize_tag(candidate.get("tag", ""))
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)

    # Per hypothesis: score what it judged, then decide the fill for what it did not.
    per_hyp_scores: dict[int, dict[str, float]] = {}
    for hyp_idx in hyp_indices:
        judged: dict[str, float] = {}
        # Corpus-level weights, when supplied, replace the per-window estimate: same weighted
        # arithmetic, but one number per dimension estimated over every window instead of from
        # the <=10 verdicts inside this one.
        weights = dict(global_weights) if global_weights else None
        if abstention == "varweight":
            # A dimension that rules the same way on every candidate in this window cannot
            # reorder them; it only contributes a constant, and -- because abstentions vary
            # across candidates -- perturbs the denominator. Weight each dimension by the
            # variance of its verdicts inside the window, so such a dimension goes to zero.
            weights = {}
            for dimension in llm_dimensions:
                vals = []
                for tag in tags:
                    verdict = verdicts.get((fact_id, hyp_idx, tag))
                    if verdict is None:
                        continue
                    value = verdict.get(dimension)
                    if value is not None:
                        vals.append(float(bool(value)))
                if len(vals) > 1:
                    mean = sum(vals) / len(vals)
                    weights[dimension] = sum((v - mean) ** 2 for v in vals) / len(vals)
                else:
                    weights[dimension] = 0.0
            if not any(weights.values()):
                weights = {d: 1.0 for d in llm_dimensions}
        for tag in tags:
            verdict = verdicts.get((fact_id, hyp_idx, tag))
            if verdict is None:
                continue
            score = llm_only_agree_score(verdict, llm_dimensions, abstention, weights=weights)
            if score is not None:
                judged[tag] = score
        if unjudged_fill == "mean" and judged:
            fill = sum(judged.values()) / len(judged)
        else:
            fill = 0.0
        per_hyp_scores[hyp_idx] = {tag: judged.get(tag, fill) for tag in tags}

    return {
        tag: round(sum(per_hyp_scores[idx][tag] for idx in hyp_indices) / len(hyp_indices), 6)
        for tag in tags
    }


def rerank_share(base_scores: dict[str, float], consensus: dict[str, float], beta: float) -> float | None:
    """Table 18's diagnostic: (beta * range(consensus)) / range(fused score), over the pool
    that survived to rerank. `base_scores` is the score consensus is added *to* -- the
    range-normalized score for the default scaling, the raw fused score for `scaling="raw"`."""
    if not base_scores:
        return None
    base_values = list(base_scores.values())
    base_range = max(base_values) - min(base_values)
    if base_range <= 1e-12:
        return None
    consensus_values = [consensus.get(tag, 0.0) for tag in base_scores]
    consensus_range = max(consensus_values) - min(consensus_values) if consensus_values else 0.0
    return round((beta * consensus_range) / base_range, 6)


def resolve_verifier_mode(config: AblationConfig) -> str:
    """Which verifier term this config asks for, with "auto" resolved.

    Fails loudly rather than silently degrading to the deterministic term: an LLM mode with
    no verdicts attached would otherwise produce a plausible-looking row that is really the
    deterministic arm under a different name.
    """
    mode = config.verifier_mode
    if mode not in VERIFIER_MODES:
        raise ValueError(f"Unknown verifier_mode {mode!r}; expected one of {VERIFIER_MODES}")
    if mode == "auto":
        return "hybrid" if config.llm_verifier_verdicts is not None else "deterministic"
    if mode != "deterministic" and config.llm_verifier_verdicts is None:
        raise ValueError(f"verifier_mode={mode!r} requires llm_verifier_verdicts; none were attached")
    return mode


_CONSENSUS_CACHE: dict[tuple[Any, ...], dict[str, float]] = {}


def reset_consensus_cache() -> None:
    _CONSENSUS_CACHE.clear()


def _consensus_for(
    fact_id: int,
    hyp_indices: list[int],
    renderings: tuple[str, ...],
    pool: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    normalization_map: dict[str, Any],
    config: AblationConfig,
) -> dict[str, float]:
    """agree()/consensus depends only on (fact, which hypotheses, which renderings) --
    fusion ("sum"/"mean") and scaling ("range"/"raw") and beta never change it. Table 5 asks
    for the same J=2-dual selection under five different fusion/scaling/beta settings, so
    caching here turns the dominant cost (symbolic profile parsing over the pool) from five
    computations per fact into one."""
    mode = resolve_verifier_mode(config)
    key = (
        fact_id,
        tuple(sorted(hyp_indices)),
        tuple(sorted(renderings)),
        mode,
        config.llm_unjudged_fill,
        config.contradicted_dimension_weight,
    )
    cached = _CONSENSUS_CACHE.get(key)
    if cached is not None:
        return cached
    if mode == "hybrid":
        result = hybrid_consensus_scores(
            pool,
            hypotheses,
            normalization_map,
            fact_id,
            hyp_indices,
            config.llm_verifier_verdicts,
            config.llm_verifier_dimensions,
            config.contradicted_dimension_weight,
            config.llm_verifier_top_m,
        )
    elif mode == "det_window":
        result = det_window_consensus_scores(
            pool,
            hypotheses,
            normalization_map,
            fact_id,
            hyp_indices,
            config.llm_verifier_verdicts,
            config.llm_verifier_dimensions,
            unjudged_fill=config.llm_unjudged_fill,
        )
    elif mode in ("llm_drop", "llm_strict", "llm_neutral", "llm_varweight"):
        result = llm_only_consensus_scores(
            pool,
            hypotheses,
            fact_id,
            hyp_indices,
            config.llm_verifier_verdicts,
            config.llm_verifier_dimensions,
            abstention=("drop" if mode == "llm_drop"
                    else "neutral" if mode == "llm_neutral"
                    else "varweight" if mode == "llm_varweight" else "negative"),
            unjudged_fill=config.llm_unjudged_fill,
            global_weights=(dict(config.llm_verifier_dimension_weights)
                            if config.llm_verifier_dimension_weights else None),
        )
    else:
        result = consensus_scores(pool, hypotheses, normalization_map)
    _CONSENSUS_CACHE[key] = result
    return result


def rerank(
    scores: dict[str, float],
    best_candidate: dict[str, dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    hyp_indices: list[int],
    renderings: tuple[str, ...],
    normalization_map: dict[str, Any],
    config: AblationConfig,
    fact_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if config.scaling == "range":
        base = range_normalize(scores)
    elif config.scaling == "raw":
        base = dict(scores)
    else:
        raise ValueError(f"Unknown scaling {config.scaling!r}; expected 'range' or 'raw'")

    pool = list(best_candidate.values())
    consensus = _consensus_for(fact_id, hyp_indices, renderings, pool, hypotheses, normalization_map, config)

    final_scores = {tag: base.get(tag, 0.0) + config.beta * consensus.get(tag, 0.0) for tag in base}
    order = sorted(final_scores, key=lambda tag: (-final_scores[tag], tag))[: config.top_k]
    ranked: list[dict[str, Any]] = []
    for rank, tag in enumerate(order, start=1):
        candidate = dict(best_candidate[tag])
        candidate["rank"] = rank
        candidate["fused_score"] = round(scores.get(tag, 0.0), 8)
        candidate["scaled_score"] = round(base.get(tag, 0.0), 8)
        candidate["consensus"] = round(consensus.get(tag, 0.0), 6)
        candidate["final_score"] = round(final_scores[tag], 8)
        ranked.append(candidate)
    diagnostics = {"rerank_share": rerank_share(base, consensus, config.beta)}
    return ranked, diagnostics


def metric_row(tags: list[str], gold_tags: list[str], top_ks: tuple[int, ...] = TOP_KS) -> dict[str, Any]:
    gold = {normalize_tag(tag) for tag in gold_tags}
    rank = None
    for index, tag in enumerate(tags, start=1):
        if normalize_tag(tag) in gold:
            rank = index
            break
    row: dict[str, Any] = {
        "rank": rank,
        "mrr": 0.0 if rank is None else 1.0 / rank,
        # Pure rank==1 on this variant's own final ranking -- no downstream LLM selector is
        # invoked anywhere in this table, so "the same selector, unchanged, across every row"
        # (section 4) holds trivially. If the paper's headline accuracy elsewhere comes from
        # a post-hoc listwise-rerank stage, that is a *different* quantity than this column
        # and the two must not be conflated; see the caveat in metrics.json.
        "top1_accuracy": bool(rank == 1),
    }
    for k in top_ks:
        row[f"recall_at_{k}"] = bool(rank is not None and rank <= k)
    return row


def _selected_hypothesis_indices(fact: FactRecord, config: AblationConfig) -> list[int]:
    available = sorted(fact.hypotheses)
    if config.n_hypotheses >= len(available):
        return available
    if config.n_hypotheses == 1:
        if config.kept_hypothesis_idx is None:
            raise ValueError(
                "n_hypotheses=1 requires kept_hypothesis_idx (section 3.2's seed-noise "
                "mitigation means the caller must average over both choices explicitly, "
                "not silently default to index 0)."
            )
        if config.kept_hypothesis_idx not in available:
            raise KeyError(f"kept_hypothesis_idx={config.kept_hypothesis_idx} not in {available}")
        return [config.kept_hypothesis_idx]
    raise ValueError(
        f"n_hypotheses={config.n_hypotheses} is neither 1 nor the full available set {available}; "
        "this table only ever drops to J=1 or keeps J as logged."
    )


def _rankings_for(
    fact: FactRecord, hyp_indices: list[int], renderings: tuple[str, ...]
) -> tuple[list[list[dict[str, Any]]], bool]:
    """Collect the rankings for the requested hypotheses/renderings. Returns (rankings,
    lab_only_all_missing) where the second flag is set only when `renderings == ("lab",)`
    and none of the kept hypotheses produced a label-form ranking (section 3.5)."""
    rankings: list[list[dict[str, Any]]] = []
    for idx in hyp_indices:
        for rendering in renderings:
            key = (idx, rendering)
            if key in fact.rankings:
                rankings.append(fact.rankings[key])
    lab_only_missing = renderings == ("lab",) and not rankings
    return rankings, lab_only_missing


def evaluate(fact: FactRecord, config: AblationConfig, normalization_map: dict[str, Any]) -> dict[str, Any]:
    """The one function every Table 5 row is a call to (section 1)."""
    if config.oracle_best_single:
        return _evaluate_oracle(fact, config, normalization_map)

    hyp_indices = _selected_hypothesis_indices(fact, config)
    actual_renderings = config.renderings
    rankings, lab_only_missing = _rankings_for(fact, hyp_indices, config.renderings)
    fallback_used = False
    if lab_only_missing and config.lab_only_fallback == "def":
        actual_renderings = ("def",)
        rankings, _ = _rankings_for(fact, hyp_indices, actual_renderings)
        fallback_used = bool(rankings)

    hypotheses_used = [fact.hypotheses[idx] for idx in hyp_indices if idx in fact.hypotheses]
    if not rankings:
        tags: list[str] = []
        diagnostics: dict[str, Any] = {"rerank_share": None}
        n_rankings = 0
    else:
        scores, best_candidate = fuse(rankings, config.rrf_kappa, config.fusion)
        if config.truncate_pool_to_top_k:
            scores, best_candidate = truncate_fused_pool(scores, best_candidate, config.top_k)
        ranked, diagnostics = rerank(
            scores, best_candidate, hypotheses_used, hyp_indices, actual_renderings, normalization_map, config, fact.fact_id
        )
        tags = [candidate["tag"] for candidate in ranked]
        n_rankings = len(rankings)

    row = metric_row(tags, fact.gold_tags)
    row.update(
        {
            "fact_id": fact.fact_id,
            "context_id": fact.context_id,
            "modality": fact.modality,
            "datatype": fact.datatype,
            "hypotheses_used": hyp_indices,
            "n_rankings_fused": n_rankings,
            "lab_query_null_for_all_kept": lab_only_missing,
            "lab_fallback_used": fallback_used,
            "candidate_tags": tags,
            **diagnostics,
        }
    )
    return row


def _evaluate_oracle(fact: FactRecord, config: AblationConfig, normalization_map: dict[str, Any]) -> dict[str, Any]:
    """Section 3.11: per hypothesis j, fuse only h_j's own rankings, range/raw-scale,
    +beta*agree(c,h_j); pick per fact whichever j ranks gold highest. An oracle over
    hypotheses, not over candidates -- it bounds what a better *selection/aggregation* rule
    over this same hypothesis set could achieve."""
    per_hypothesis: dict[int, dict[str, Any]] = {}
    for idx, hypothesis in fact.hypotheses.items():
        own_rankings, _ = _rankings_for(fact, [idx], config.renderings)
        if not own_rankings:
            continue
        scores, best_candidate = fuse(own_rankings, config.rrf_kappa, config.fusion)
        ranked, diagnostics = rerank(
            scores, best_candidate, [hypothesis], [idx], config.renderings, normalization_map, config, fact.fact_id
        )
        tags = [candidate["tag"] for candidate in ranked]
        row = metric_row(tags, fact.gold_tags)
        row.update({"candidate_tags": tags, "n_rankings_fused": len(own_rankings), **diagnostics})
        per_hypothesis[idx] = row

    if not per_hypothesis:
        row = metric_row([], fact.gold_tags)
        row.update(
            {
                "fact_id": fact.fact_id,
                "context_id": fact.context_id,
                "modality": fact.modality,
                "datatype": fact.datatype,
                "hypotheses_used": [],
                "n_rankings_fused": 0,
                "lab_query_null_for_all_kept": False,
                "lab_fallback_used": False,
                "oracle_selected_hypothesis_idx": None,
                "oracle_per_hypothesis_ranks": {},
                "rerank_share": None,
            }
        )
        return row

    def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        idx, row = item
        rank = row["rank"]
        return (rank if rank is not None else 10**9, idx)

    selected_idx, selected_row = min(per_hypothesis.items(), key=sort_key)
    out = dict(selected_row)
    out.update(
        {
            "fact_id": fact.fact_id,
            "context_id": fact.context_id,
            "modality": fact.modality,
            "datatype": fact.datatype,
            "hypotheses_used": [selected_idx],
            "lab_query_null_for_all_kept": False,
            "lab_fallback_used": False,
            "oracle_selected_hypothesis_idx": selected_idx,
            "oracle_per_hypothesis_ranks": {idx: row["rank"] for idx, row in per_hypothesis.items()},
        }
    )
    return out


def aggregate(rows: list[dict[str, Any]], top_ks: tuple[int, ...] = TOP_KS) -> dict[str, Any]:
    """Mean of each metric over `rows`.

    float(), NOT bool(). For a row produced by evaluate() the two are identical -- the metrics
    really are booleans and float(True) == 1.0. They diverge only when a caller has already
    averaged two per-fact rows together, which section 3.2's `-ensemble` row does: it reports
    the mean over WHICH hypothesis is kept, so its per-fact recall is 0.0, 0.5 or 1.0.
    bool(0.5) is True, which silently promoted every split decision to a full hit and turned
    that row into "either hypothesis hit" -- the union, i.e. exactly the oracle best-single
    row (3.11). The two rows came out numerically identical, and -ensemble appeared to BEAT
    AGS full while its own paired-bootstrap delta (computed from the same floats, correctly)
    said it was worse. mrr never showed the bug because it was already summed as float.
    """
    n = len(rows)
    if n == 0:
        return {"n": 0}
    out: dict[str, Any] = {
        "n": n,
        "mrr": round(sum(float(row["mrr"]) for row in rows) / n, 6),
        "top1_accuracy": round(sum(float(row["top1_accuracy"]) for row in rows) / n, 6),
    }
    for k in top_ks:
        out[f"recall_at_{k}"] = round(sum(float(row[f"recall_at_{k}"]) for row in rows) / n, 6)
    return out
