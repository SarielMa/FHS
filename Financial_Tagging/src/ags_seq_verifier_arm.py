#!/usr/bin/env python3
"""FHS-Seq: the sequential control the paper actually describes.

The paper says FHS-Seq is FHS with the one-shot parallel fan replaced by a revise-and-refetch
loop, and that EVERY other component is identical. The earlier arms in `ags_sequential_arms.py`
are not that control. They differ from FHS in three ways at once:

  1. they branch from the deterministic core and hold the candidate-level verifier OUT of both
     round one and the full episode, so neither side of the comparison is FHS;
  2. they take revision feedback from the program-driven check, which the paper reports as a
     rejected design and does not use for ranking;
  3. they select edits with a Thompson-sampling controller (`ags_seq`) or uniformly
     (`ags_seq_random`), which makes the controller a second variable.

Measured consequence of (1): on a 40-fact sample the old round-one pool overlaps frozen FHS by
only ~156/200 tags and is set-equal on 10/40 facts, so "round one is FHS's parallel round" was
not true of it either.

This module is the matched control. Only the control flow differs from FHS:

  * round one is FHS, verifier included, by construction -- it calls the same consolidation;
  * every later round consolidates through the SAME verifier-in-the-loop path, so both sides of
    round-1-vs-full are scored identically;
  * feedback comes from the candidate-level verifier's own per-dimension verdicts, not from the
    program-driven check;
  * the revision target is chosen deterministically, so there is no controller to confound.

Revision targets whichever dimensions the verifier rules on, read live from
`LLM_VERIFIER_DIMENSIONS_DEFAULT` -- as of 2026-07-30 that is all six the generator emits, so the
loop can revise QUALIFIER, SCOPE and TEMPORAL too (`DIMENSION_OPERATOR` has a REFINE operator for
each, and the tally is keyed off the same constant, so widening it needs no change here). Before
2026-07-30 the constant held FAMILY/ROLE/EVENT only and the other three were rendered into queries
but never verified, which left the loop no signal on which to revise them.

Cost note: FHS makes J verifier calls per fact. This loop makes J per round, so a four-round
budget is ~4x the verifier cost of FHS. That is inherent to the control being matched -- an arm
that skipped verification in later rounds would not be FHS-with-a-loop.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ags_table5_ablation"))

from ags_sequential_arms import (  # noqa: E402
    DIMENSION_OPERATOR,
    AgsSeqConfig,
    cluster_representatives,
    fuse_round_candidates,
    range_normalize,
)
from verifier.core import LLM_VERIFIER_DIMENSIONS_DEFAULT  # noqa: E402
from verifier.run_llm_verifier import (  # noqa: E402
    build_verifier_messages,
    parse_verifier_output,
)
from ags_symbolic_agreement import normalize_tag  # noqa: E402
from run_fintagging_grounding_baseline import build_prompt_under_query_budget  # noqa: E402

VERIFIER_DIMENSIONS = LLM_VERIFIER_DIMENSIONS_DEFAULT
# How many verifier calls to allow before insisting that at least one parsed cleanly.
MIN_CALLS_BEFORE_PARSE_GUARD = 8


def support_value(verdict: dict[str, Any] | None) -> float | None:
    """The candidate's support under llm_drop: matched / ruled-on.

    None means the verifier declined the candidate entirely; the caller floors that to 0.0,
    which is where an unseen candidate also lands. Abstaining on a single dimension only
    removes that dimension from the denominator -- it is not evidence against the candidate.
    Matches `core.llm_only_agree_score(abstention="drop")`.
    """
    if verdict is None:
        return None
    ruled = [bool(verdict[d]) for d in VERIFIER_DIMENSIONS if verdict.get(d) is not None]
    if not ruled:
        return None
    return sum(ruled) / len(ruled)


def verify_window(
    generator: Any,
    args: Any,
    record: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
    cfg: AgsSeqConfig,
    normalization_map: dict[str, Any],
    top_m: int,
) -> tuple[dict[int, dict[str, dict[str, Any]]], dict[str, Any]]:
    """One verifier call per hypothesis over the top-`top_m` window of `ranking`.

    Returns (verdicts_by_hypothesis_index, stats). Batched through generate_many: calling
    generate_one per prompt runs vLLM at batch size 1 regardless of --vllm-batch-size, which is
    what made the original verdict runs take hours per thousand calls.
    """
    window = cluster_representatives(ranking, normalization_map, top_m, cfg.cluster_scan_depth)
    tags = [normalize_tag(c.get("tag", "")) for c in window]
    prompts, index = [], []
    for hyp_idx, hypothesis in enumerate(hypotheses):
        # generate_many takes rendered prompt strings, not chat messages, and the context has
        # to be trimmed to the token budget before it gets there -- both are what
        # build_prompt_under_query_budget does, and it is the same call the one-shot verifier
        # in run_llm_verifier.py makes, so the two arms send identical prompts.
        prompt, _prompt_tokens, _used_chars = build_prompt_under_query_budget(
            generator.tokenizer,
            lambda ctx_chars, hypothesis=hypothesis: build_verifier_messages(
                record,
                hypothesis,
                window,
                ctx_chars,
                args.candidate_doc_max_chars,
            ),
            context_max_chars=args.query_context_max_chars,
            max_input_tokens=args.query_max_input_tokens,
        )
        prompts.append(prompt)
        index.append(hyp_idx)

    raw_outputs = generator.generate_many(prompts)

    verdicts: dict[int, dict[str, dict[str, Any]]] = {}
    n_clean = 0
    for hyp_idx, raw in zip(index, raw_outputs):
        by_tag, ok, mode = parse_verifier_output(raw, tags)
        verdicts[hyp_idx] = by_tag
        n_clean += int(ok and mode == "clean")
    stats = {
        "calls": len(prompts),
        "clean_parses": n_clean,
        "window_size": len(window),
        "window_tags": tags,
    }
    return verdicts, stats


def consolidate_with_verifier(
    rounds: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    cfg: AgsSeqConfig,
    normalization_map: dict[str, Any],
    top_k: int | None,
    generator: Any,
    args: Any,
    record: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, dict[str, Any]]], dict[str, Any]]:
    """FHS's consolidation: sum-RRF -> range-normalize -> + beta * mean verifier support.

    This is `ags_sequential_arms.consolidate` with the deterministic consensus term replaced by
    the candidate-level verifier's support, which is the only difference between the deployed
    method and its deterministic core. Round one and every later round both go through here, so
    the round-1-vs-full comparison is internal.
    """
    fused = fuse_round_candidates(rounds, top_k, cfg.frozen.rrf_kappa)
    if not fused:
        return [], {}, {"calls": 0, "clean_parses": 0, "window_size": 0, "window_tags": []}
    rrf_by_tag = {normalize_tag(c["tag"]): float(c.get("rrf_score", 0.0)) for c in fused}
    normed = range_normalize(rrf_by_tag)

    # The window is cut from the fused ranking, never from an already-reranked order. Cutting it
    # from a reranked order is what made the earlier verifier ablations non-independent of the
    # component they ablated.
    fused_order = sorted(normed, key=lambda t: (-normed[t], t))
    by_tag = {normalize_tag(c["tag"]): c for c in fused}
    fused_ranking = [by_tag[t] for t in fused_order if t in by_tag]

    verdicts, stats = verify_window(
        generator, args, record, hypotheses, fused_ranking, cfg, normalization_map, cfg.feedback_top_m
    )

    support: dict[str, float] = {}
    for tag in normed:
        vals = []
        for hyp_idx in range(len(hypotheses)):
            v = support_value(verdicts.get(hyp_idx, {}).get(tag))
            vals.append(0.0 if v is None else v)
        support[tag] = sum(vals) / len(vals) if vals else 0.0

    final = {t: normed.get(t, 0.0) + cfg.frozen.rerank_beta * support.get(t, 0.0) for t in normed}
    order = sorted(final, key=lambda t: (-final[t], t))
    if top_k is not None and top_k > 0:
        order = order[:top_k]
    ranked = []
    for rank, tag in enumerate(order, start=1):
        cand = dict(by_tag[tag])
        cand["rank"] = rank
        cand["seq_rrf_normalized"] = round(normed.get(tag, 0.0), 6)
        cand["seq_verifier_support"] = round(support.get(tag, 0.0), 6)
        cand["seq_final_score"] = round(final[tag], 6)
        ranked.append(cand)
    return ranked, verdicts, stats


def verifier_feedback(
    verdicts: dict[int, dict[str, dict[str, Any]]],
    window_tags: list[str],
) -> dict[str, dict[str, int]]:
    """Per-dimension support/no-support tallies over the judged window.

    Counts every (hypothesis, candidate) verdict the verifier actually returned. Abstentions are
    counted separately rather than folded into either side: a dimension the verifier never ruled
    on carries no evidence that the hypothesis is wrong on it.
    """
    tally = {d: {"support": 0, "against": 0, "abstain": 0} for d in VERIFIER_DIMENSIONS}
    for by_tag in verdicts.values():
        for tag in window_tags:
            verdict = by_tag.get(tag)
            if verdict is None:
                continue
            for dim in VERIFIER_DIMENSIONS:
                value = verdict.get(dim)
                if value is None:
                    tally[dim]["abstain"] += 1
                elif value:
                    tally[dim]["support"] += 1
                else:
                    tally[dim]["against"] += 1
    return tally


def choose_revision_target(tally: dict[str, dict[str, int]]) -> str | None:
    """The dimension the verifier least supports, chosen deterministically.

    No controller: the target is the dimension with the largest (against - support) margin,
    ties broken by the fixed order FAMILY, ROLE, EVENT. Returns None when no dimension is net
    unsupported, which stops the episode -- continuing would mean revising a dimension the
    verifier endorses, which is not what the paper describes.
    """
    best, best_margin = None, 0
    for dim in VERIFIER_DIMENSIONS:
        margin = tally[dim]["against"] - tally[dim]["support"]
        if margin > best_margin:
            best, best_margin = dim, margin
    return best


def revision_directive(dimension: str) -> dict[str, Any]:
    """A REFINE directive on `dimension`, the only directive kind this arm issues."""
    return {
        "mode": "REFINE",
        "operator": DIMENSION_OPERATOR[dimension],
        "target_dimension": dimension,
        "rationale": "candidate-level verifier does not support this dimension",
    }


def evidence_record(example: Any) -> dict[str, Any]:
    """The flattened fields `build_verifier_messages` reads, taken from a live Example.

    The verifier module normally adapts a persisted trace record; inside the loop the trace does
    not exist yet, so the same fields are read off the Example directly. Field names follow
    `_EvidenceView` exactly -- a rename there must be mirrored here or the prompt silently loses
    its evidence and every verdict degrades to a guess from the candidate list alone.
    """
    return {
        "row_context": getattr(example, "row_context", "") or "",
        "column_context": getattr(example, "column_context", "") or "",
        "query_context": getattr(example, "query_context", "") or "",
        "input_type": getattr(example, "input_type", "") or "",
        "entity": getattr(example, "entity", "") or "",
        "type": getattr(example, "entity_type", "") or "",
    }


def build_seq_verifier_record(
    args: Any,
    arm: str,
    generator: Any,
    retriever: Any,
    example: Any,
    normalization_map: dict[str, Any],
    bank: Any = None,
    cfg: AgsSeqConfig | None = None,
) -> dict[str, Any]:
    """One episode of the matched sequential control.

    Signature matches `build_ags_seq_method_record` so the runner branch is a one-line addition.
    `bank` is accepted and ignored: this arm has no controller and therefore no posteriors.
    """
    from ags_sequential_arms import (
        compact_candidates,
        retrieve_for_hypothesis,
        revise_hypothesis,
        sample_hypotheses,
    )
    from ags_frozen_grounding import frozen_ags_rankings
    from run_fintagging_grounding_baseline import finalize_candidate_record

    cfg = cfg or AgsSeqConfig()
    record_view = evidence_record(example)

    # --- round one IS FHS -----------------------------------------------------
    hypotheses, calls, used_fallback = sample_hypotheses(generator, args, example, cfg.frozen)
    pool = frozen_ags_rankings(retriever, example, hypotheses, cfg.frozen)
    pool_hypotheses = list(hypotheses)
    ranking, verdicts, vstats = consolidate_with_verifier(
        pool, pool_hypotheses, cfg, normalization_map, cfg.frozen.top_k, generator, args, record_view
    )
    round1_ranking = list(ranking)
    verifier_calls = vstats["calls"]
    clean_parses = vstats["clean_parses"]

    current = dict(hypotheses[0])
    rounds_log: list[dict[str, Any]] = []
    stop_reason = "budget"
    realized_rounds = 1

    for _ in range(2, cfg.max_rounds + 1):
        tally = verifier_feedback(verdicts, vstats["window_tags"])
        target = choose_revision_target(tally)
        if target is None:
            # Every ruled-on dimension is net supported. Revising one anyway would edit a
            # dimension the verifier endorses, which is not the loop the paper describes.
            stop_reason = "no_unsupported_dimension"
            break
        directive = revision_directive(target)
        revised, _reverted, revision_calls, parse_ok = revise_hypothesis(
            args, generator, example, current, directive, cfg, "seq_verifier_revision"
        )
        calls.extend(revision_calls)
        if not parse_ok:
            stop_reason = "revision_failed"
            break

        new_rounds = retrieve_for_hypothesis(retriever, example, revised, cfg, len(pool))
        pool = pool + new_rounds
        pool_hypotheses = pool_hypotheses + [revised]
        ranking, verdicts, vstats = consolidate_with_verifier(
            pool, pool_hypotheses, cfg, normalization_map, cfg.frozen.top_k, generator, args, record_view
        )
        verifier_calls += vstats["calls"]
        clean_parses += vstats["clean_parses"]
        # A truncated verdict array is indistinguishable from agreement downstream: salvage keeps
        # the first few candidates, the tally then finds no unsupported dimension, and the loop
        # exits after round one. That produced a silent 0/171-clean-parse run once, so refuse to
        # continue past the first few facts when nothing is parsing cleanly.
        if verifier_calls >= MIN_CALLS_BEFORE_PARSE_GUARD and clean_parses == 0:
            raise SystemExit(
                f"seq_verifier: {verifier_calls} verifier calls, 0 clean parses. Responses are "
                f"being truncated -- raise --query-max-new-tokens (currently "
                f"{getattr(args, 'query_max_new_tokens', '?')}); a verdict array over "
                f"{cfg.feedback_top_m} candidates x {len(VERIFIER_DIMENSIONS)} dimensions needs "
                "roughly 2,800 tokens."
            )
        current = revised
        realized_rounds += 1
        rounds_log.append(
            {"round": realized_rounds, "target_dimension": target, "tally": tally}
        )

    prompt_tokens = sum(int(c.get("prompt_tokens", 0) or 0) for c in calls)
    completion_tokens = sum(int(c.get("completion_tokens", 0) or 0) for c in calls)
    stored_rounds = [dict(r, candidates=compact_candidates(r["candidates"])) for r in pool]
    record = finalize_candidate_record(
        example,
        query_mode=arm,
        rounds=stored_rounds,
        top_k=cfg.frozen.top_k,
        rrf_kappa=cfg.frozen.rrf_kappa,
        total_llm_calls=len(calls) + verifier_calls,
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
    )
    record["final_candidates"] = ranking
    record["round1_candidates"] = [c.get("tag", "") for c in round1_ranking]
    record["seq_verifier_rounds"] = rounds_log
    record["seq_verifier_realized_rounds"] = realized_rounds
    record["seq_verifier_stop_reason"] = stop_reason
    record["seq_verifier_calls"] = verifier_calls
    record["seq_verifier_clean_parses"] = clean_parses
    return record
