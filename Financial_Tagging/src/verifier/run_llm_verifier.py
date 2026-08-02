#!/usr/bin/env python3
"""Row 3.9: + LLM verification layer (ags_table5_ablation_spec.md section 3.9). GPU, vLLM.

The only row requiring new generation: J=2 calls per fact (~2 x 2,509 = 5,020) re-judging
FAMILY/ROLE/EVENT on the top M=10 cluster-representative candidates from AGS's own fused
ranking. QUALIFIER/SCOPE/TEMPORAL, and FAMILY/ROLE/EVENT outside the top M, stay symbolic --
core.hybrid_agree_score enforces that split; this script only produces the verdicts file
core.py's `llm_verifier_verdicts` hook consumes.

Standalone by design: it does not import or extend run_fintagging_grounding_baseline.py's
--query-mode dispatch (build_comparison_candidate_records), which other experiments run
against concurrently. It reads AGS's already-computed test-split trace, reuses the prompt-
building primitives (serialize_evidence, format_candidate_for_prompt, QueryGenerator,
parse_json_object) as pure, read-only imports, and writes only to its own output file.

The verifier never sees the gold concept -- nothing in the prompt below includes it.
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

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from ags_sequential_arms import cluster_representatives  # noqa: E402
from ags_symbolic_agreement import (  # noqa: E402
    DEFAULT_NORMALIZATION_MAP,
    canonical_hypothesis_dimensions,
    is_unresolved,
    load_normalization_map,
    symbolic_feedback_from_candidates,
)
from verifier.data_prep import DEFAULT_TEST_TRACE, stream_jsonl  # noqa: E402
from run_fintagging_grounding_baseline import (  # noqa: E402
    QueryGenerator,
    build_prompt_under_query_budget,
    format_candidate_for_prompt,
    llm_call_record,
    normalize_tag,
    parse_json_object,
    serialize_evidence,
)

# The six dimensions the generator actually emits. SYMBOLIC_DIMENSIONS carries a seventh name,
# AGGREGATION, that no hypothesis ever fills (measured 0/1,200), so it is deliberately not here.
ALL_JUDGED_DIMENSIONS = ("FAMILY", "ROLE", "EVENT", "QUALIFIER", "SCOPE", "TEMPORAL")
# The pre-2026-07-30 configuration, kept only to re-read verdict files generated under it.
LEGACY_JUDGED_DIMENSIONS = ("FAMILY", "ROLE", "EVENT")
# Deployed as of 2026-07-30: ask about every generated dimension. The old three-dimension set
# required a per-domain judgement about which dimensions a candidate's text can decide, and that
# judgement does not transfer (ICD-10-CM writes laterality and encounter type into the code text).
VERIFIER_DIMENSIONS = ALL_JUDGED_DIMENSIONS
# Descriptions follow the paper's own typing (Section: task instantiation, M=6): FAMILY/ROLE/
# EVENT are label-derived and compared by token overlap; QUALIFIER/SCOPE/TEMPORAL are
# metadata-determined and matched against controlled vocabularies of 18, 7 and 11 categories.
DIMENSION_GLOSS = {
    "FAMILY": "broad accounting domain",
    "ROLE": "specific function",
    "EVENT": "event or state",
    "QUALIFIER": "measurement basis, e.g. gross versus net",
    "SCOPE": "entity or consolidation scope",
    "TEMPORAL": "period type, e.g. instant versus duration",
}
VERDICT_TO_MATCHED = {"support": True, "contradict": False, "unresolved": None}


class _EvidenceView:
    """serialize_evidence needs an Example; the trace only persists the flattened fields it
    already rendered. This adapts the persisted record without re-deriving anything."""

    __slots__ = ("row_context", "column_context", "query_context", "input_type", "entity", "entity_type")

    def __init__(self, record: dict[str, Any]) -> None:
        self.row_context = record.get("row_context", "")
        self.column_context = record.get("column_context", "")
        self.query_context = record.get("query_context", "")
        self.input_type = record.get("input_type", "")
        self.entity = record.get("entity", "")
        self.entity_type = record.get("type", "")


def window_ranking(record: dict[str, Any], source: str) -> list[dict[str, Any]]:
    """The candidate ordering the top-M verifier window is cut from.

    WHY THIS IS NOT JUST record["final_candidates"]
        Those are ordered by frozen_ags_final_score, which already includes the deterministic
        agree term -- measured over 200 facts, the top 10 is sorted by that score 200/200
        times and by the fused score only 27/200. Cutting the verifier's window from it means
        the deterministic verifier decides which candidates the LLM is ever shown, in every
        arm, including the ones that claim to have removed it. The two windows overlap 8.67/10
        on average and are identical as a set on only 23.4% of facts, so it is not a rounding
        difference.

        "fused" restores what this module's docstring always described: the window comes from
        the fused retrieval ranking, before either verifier touches it, so a deterministic arm
        and an LLM arm are scored over the same candidate set.
    """
    candidates = record.get("final_candidates") or record.get("candidates") or []
    if source == "deployed":
        return candidates
    # Ties broken by tag, matching core.py's own (-score, ..., tag) convention, so the window
    # is reproducible rather than dependent on the trace's insertion order.
    return sorted(candidates, key=lambda c: (-(c.get("frozen_ags_rrf_normalized") or 0.0), c.get("tag", "")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-trace", type=Path, default=DEFAULT_TEST_TRACE)
    parser.add_argument("--output-dir", type=Path, default=_PARENT / "runs_ags_table5_ablation" / "qwen3_32b")
    parser.add_argument("--top-m", type=int, default=10)
    parser.add_argument("--cluster-scan-depth", type=int, default=60)
    parser.add_argument(
        "--window-source",
        choices=("fused", "deployed"),
        default="fused",
        help="Which ranking the top-M verifier window is taken from. 'fused' uses the fused "
        "retrieval score alone (frozen_ags_rrf_normalized), which is what this script's "
        "docstring always claimed and what a clean verifier ablation needs. 'deployed' uses "
        "the trace's final_candidates order, which is ALREADY deterministically reranked -- "
        "that makes the deterministic verifier choose which candidates the LLM ever judges, "
        "so the '- deterministic verifier' arms are not then independent of it. 'deployed' "
        "exists only to reproduce verdicts generated before this flag.",
    )
    parser.add_argument(
        "--ask-decisive-dimensions",
        action="store_true",
        help="Also ask the verifier which dimensions actually discriminate among the candidates "
        "it was shown, and persist that list per (fact, hypothesis). Costs no extra call. The "
        "per-dimension verdicts are unchanged, so a verdicts file generated with this flag is a "
        "superset of one generated without it: score it as usual, or filter each verdict to the "
        "model's own decisive set to test LLM-chosen scoring against a hand-picked one.",
    )
    parser.add_argument(
        "--window-tags",
        type=Path,
        default=None,
        help="JSONL of per-fact windows written by stage_arm_windows.py: "
        '{"fact_id", "hypothesis_indices", "window_tags"}. Given, it REPLACES --window-source: '
        "the window is the listed tags in the listed order and only the listed hypotheses are "
        "judged. This is how an ablation arm gets verdicts over its OWN fused ranking instead "
        "of the deployed one -- without it every arm inherits FHS's window, which is why the "
        "rendering/ensemble/fusion rows of tab:ablation could only be scored with the "
        "deterministic term. A fact absent from the file is skipped (the arm has no ranking "
        "for it, e.g. lab-only on narrative evidence).",
    )
    parser.add_argument(
        "--generation-chunk",
        type=int,
        default=64,
        help="How many prompts to buffer before calling generate_many. The original loop "
        "called generate_one per prompt, which runs vLLM at batch size 1 regardless of "
        "--vllm-batch-size and is why a full run took hours per thousand calls.",
    )
    parser.add_argument(
        "--symbolic-hint",
        action="store_true",
        help="Put the deterministic verifier's D- verdict in the prompt instead of using it as "
        "a scoring term. The symbolic layer names the dimensions the hypothesis is probably "
        "wrong on (its measured strength, tab:verifier F1 0.756 vs the LLM's 0.254) and the LLM "
        "decides per candidate what to do with that. Verdicts produced this way are meant to be "
        "scored with verifier_mode=llm_drop, so the symbolic layer never enters the score.",
    )
    parser.add_argument("--query-generation-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--query-generation-backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--query-context-max-chars", type=int, default=12000)
    parser.add_argument("--query-max-input-tokens", type=int, default=16000)
    # THIS is the cap that bounds the verdict JSON: QueryGenerator.generate_many passes
    # query_max_new_tokens as vLLM's max_tokens (run_fintagging_grounding_baseline.py:1331).
    # One verdict entry is ~110-130 tokens (tag + 3 dimensions + confidence + punctuation), so
    # top_m=10 needs ~1,100-1,300 plus the wrapper. The original 512 truncated EVERY response
    # mid-array: a full 2,369-call run produced a 100% parse failure rate and zero verdicts.
    # Sized at 1536 to leave headroom; raise it alongside --top-m if that is ever increased.
    parser.add_argument("--query-max-new-tokens", type=int, default=1536)
    # load_vllm_engine (run_fintagging_grounding_baseline.py) sizes max_model_len off the
    # plain (non-query-prefixed) max_input_tokens/max_new_tokens as well; both are read even
    # though this script only ever calls generate_one on the query_* generation path.
    parser.add_argument("--max-input-tokens", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--candidate-doc-max-chars", type=int, default=320)
    parser.add_argument("--query-temperature", type=float, default=0.0)
    parser.add_argument("--query-top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--vllm-batch-size", type=int, default=32)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    # Fail-fast. The truncation bug ran 4h34m across 2,369 calls at a 0% parse rate before
    # anyone looked at the output; a zero firing rate is also the *expected* result of genuine
    # abstention, so nothing downstream flags it. Check early and abort loudly instead.
    parser.add_argument("--abort-check-after", type=int, default=25, help="0 disables the guard.")
    parser.add_argument("--abort-if-parse-rate-below", type=float, default=0.5)
    # WHICH DIMENSIONS THE HINT MAY MENTION.
    #   "llm" (default, and what the first --symbolic-hint run used) restricts the hint to
    #   FAMILY/ROLE/EVENT -- exactly the dimensions the LLM already judges. That makes the hint
    #   a second opinion on the model's own task, and withholds precisely the dimensions the
    #   symbolic layer uniquely covers, so it cannot test whether the symbolic layer's
    #   EXCLUSIVE knowledge is useful as prompt context.
    #   "all" passes every dimension the pre-check resolved. It also widens the hypothesis
    #   shown in the prompt to match, because naming a dimension whose value the model was
    #   never given -- under an instruction that says to judge three dimensions only -- is an
    #   incoherent prompt, not a wider experiment. The judged output schema is unchanged
    #   either way: the model still returns FAMILY/ROLE/EVENT verdicts and nothing else.
    # WHICH DIMENSIONS THE LLM IS ASKED TO RETURN. This is the control for --hint-dimensions:
    # the hint experiment gives the model the symbolic layer's opinion on extra dimensions,
    # while this gives the model the extra dimensions to judge itself. Comparing the two
    # separates "more dimensions help" from "the symbolic layer's verdict helps".
    # NOTE QUALIFIER/SCOPE/TEMPORAL are metadata-determined -- the symbolic layer matches them
    # exactly against controlled vocabularies -- so asking an LLM for them is a genuinely
    # different proposition from asking for the label-derived three.
    parser.add_argument(
        "--judge-dimensions",
        choices=("llm", "all", "legacy"),
        default="llm",
        help="Dimensions the LLM is asked to return a verdict on. 'llm' is FAMILY/ROLE/EVENT "
        "(every run before this flag). 'all' adds QUALIFIER/SCOPE/TEMPORAL. Score the result "
        "with a matching --llm-dimensions on the ablation side.",
    )
    parser.add_argument(
        "--hint-dimensions",
        choices=("llm", "all", "legacy"),
        default="llm",
        help="Dimensions the --symbolic-hint pre-check may mention. 'all' also widens the "
        "hypothesis shown in the prompt so every named dimension has a visible value.",
    )
    return parser.parse_args()


def symbolic_hint(
    hypothesis: dict[str, Any],
    representatives: list[dict[str, Any]],
    normalization_map: dict[str, Any],
    top_m: int,
    scope: str = "llm",
) -> str:
    """The deterministic verifier's D- verdict, rendered for the LLM prompt.

    tab:verifier measures the symbolic layer at 0.968 precision / 0.756 F1 for deciding that a
    HYPOTHESIS is wrong on a dimension, against the LLM layer's 0.254 F1 -- but that verdict
    never reached the ranking, and weighting agree() by it changes nothing, because a D- verdict
    is a property of the hypothesis and so discounts every candidate equally.

    Passing it as prompt context instead keeps each layer on the task it wins: the symbolic
    layer says which dimensions are unreliable, and the LLM -- which is the better candidate
    discriminator (87.1% of calls favour gold) -- decides what to do about it per candidate.

    Gold-free by construction: symbolic_feedback_from_candidates reads only the hypothesis and
    the retrieved candidates, so nothing here leaks the answer into the prompt.
    """
    feedback = symbolic_feedback_from_candidates(
        hypothesis.get("dimensions", hypothesis),
        representatives,
        top_m=top_m,
        normalization_map=normalization_map,
    )
    # scope="llm" keeps the hint on FAMILY/ROLE/EVENT; scope="all" lets the symbolic layer's
    # exclusive dimensions (QUALIFIER/SCOPE/TEMPORAL/AGGREGATION) through, which is the only
    # configuration that tests whether that exclusive knowledge helps as prompt context.
    allowed = None if scope == "all" else set(VERIFIER_DIMENSIONS)
    contradicted = [d for d in feedback.get("contradicted_dimensions", []) if allowed is None or d in allowed]
    supported = [d for d in feedback.get("supported_dimensions", []) if allowed is None or d in allowed]
    if not contradicted and not supported:
        return ""
    lines = ["", "Symbolic pre-check of the HYPOTHESIS against these candidates (not the answer key):"]
    if contradicted:
        lines.append(
            f"  likely WRONG in the hypothesis: {', '.join(contradicted)} -- few candidates carry the "
            "hypothesis's value here, so treat the hypothesis as unreliable on these and judge each "
            "candidate on its own merits rather than on matching the hypothesis."
        )
    if supported:
        lines.append(f"  corroborated: {', '.join(supported)} -- most candidates agree with the hypothesis here.")
    lines.append("  This is a heuristic over the candidate list. Override it when the evidence disagrees.")
    return "\n".join(lines)


def build_verifier_messages(
    record: dict[str, Any],
    hypothesis: dict[str, Any],
    representatives: list[dict[str, Any]],
    context_max_chars: int,
    doc_max_chars: int,
    hint: str = "",
    hypothesis_scope: str = "llm",
    judged_dimensions: tuple[str, ...] = VERIFIER_DIMENSIONS,
    ask_decisive: bool = False,
) -> list[dict[str, str]]:
    evidence = serialize_evidence(_EvidenceView(record), context_max_chars)
    candidate_text = "\n\n".join(
        format_candidate_for_prompt(candidate, doc_max_chars) for candidate in representatives
    )
    dims = hypothesis.get("dimensions", {})
    # Under hypothesis_scope="all" the hint may name a dimension outside FAMILY/ROLE/EVENT, so
    # the hypothesis has to carry a value for it. The judged dimensions are unchanged -- this
    # widens what the model can READ, never what it is asked to RETURN.
    #
    # Only dimensions the hypothesis ACTUALLY RESOLVED are shown. Enumerating SYMBOLIC_DIMENSIONS
    # instead would inject `"AGGREGATION": "UNRESOLVED"` into every prompt: that constant carries
    # a seventh name the generator never emits (measured over 1,200 hypotheses -- FAMILY/ROLE/
    # EVENT/QUALIFIER/SCOPE/TEMPORAL appear on all of them, AGGREGATION on none). Padding the
    # prompt with a field that stands for nothing would be noise in an experiment whose whole
    # question is whether extra information helps. Unresolved dimensions are dropped for the same
    # reason -- QUALIFIER resolves on 0.81 of hypotheses and SCOPE on 0.63, so a fixed list would
    # also be mostly placeholders.
    if hypothesis_scope == "all":
        canonical = canonical_hypothesis_dimensions(dims)
        shown = {
            dimension.upper(): value
            for dimension, value in canonical.items()
            if not is_unresolved(value)
        }
    else:
        shown = {
            dimension: dims.get(dimension, dims.get(dimension.lower(), "UNRESOLVED"))
            for dimension in VERIFIER_DIMENSIONS
        }
    hypothesis_text = json.dumps(shown, ensure_ascii=False)
    hypothesis_label = (
        "resolved dimensions" if hypothesis_scope == "all" else "FAMILY/ROLE/EVENT only"
    )
    # The judged set is what the model is asked to RETURN. It is independent of
    # hypothesis_scope, which controls what the model may READ.
    count_word = {3: "exactly three", 6: "exactly six"}.get(len(judged_dimensions), f"exactly {len(judged_dimensions)}")
    dimension_list = ", ".join(f"{d} ({DIMENSION_GLOSS[d]})" for d in judged_dimensions)
    dimension_names = "/".join(judged_dimensions)
    schema_fields = ", ".join(f'"{d}": "support|contradict|unresolved"' for d in judged_dimensions)
    # An extra field, not an extra call: the model also names which dimensions actually separate
    # THESE candidates. That replaces a hand-picked or variance-estimated scoring set with the
    # model's own choice, which needs no per-domain decision -- while the per-dimension verdicts
    # above stay exactly as they were, so the factorized mechanism is untouched.
    decisive_block = ""
    decisive_schema = ""
    if ask_decisive:
        decisive_block = (
            "\n\nAlso report which of these dimensions actually DISCRIMINATE among the candidates shown --"
            "\na dimension on which every candidate gets the same verdict cannot separate them, however"
            "\nimportant it is in general. List them most discriminating first; the list may be empty."
        )
        decisive_schema = ', "decisive_dimensions": ["EVENT"]'

    user = f"""Judge how well each candidate taxonomy concept matches a grounding hypothesis, on {count_word}
dimensions: {dimension_list}.

Do not judge any other dimension. Do not try to identify which candidate is the correct answer -- assess
each dimension independently for every candidate. If the evidence does not let you decide a dimension for
a candidate, say "unresolved" rather than guessing.{decisive_block}

Hypothesis ({hypothesis_label}):
{hypothesis_text}
{hint}

Evidence:
{evidence}

Candidates:
{candidate_text}

Return JSON only, one entry per candidate tag, with this schema:
{{"verdicts": [{{"tag": "us-gaap:Example", {schema_fields}, "confidence": 0.0}}]{decisive_schema}}}"""
    return [
        {"role": "system", "content": f"You verify US-GAAP grounding hypotheses against candidate concepts on {dimension_names} only."},
        {"role": "user", "content": user},
    ]


def salvage_verdict_entries(raw_output: str) -> list[dict[str, Any]]:
    """Recover the complete objects from a truncated `{"verdicts": [ {...}, {...}, {...`.

    A verdict list is independent per candidate, so a response cut off mid-array still carries
    usable judgements for every candidate that finished. Whole-document json.loads throws all
    of them away because the array never closes. This scans the array and keeps each balanced
    `{...}`, stopping at the point of truncation.

    Defence in depth, not the primary fix -- --query-max-new-tokens is now sized so responses
    complete. Any salvage is counted and reported so silent partial coverage stays visible.
    """
    text = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL | re.IGNORECASE)
    anchor = text.find('"verdicts"')
    start = text.find("[", anchor if anchor >= 0 else 0)
    if start < 0:
        return []

    entries: list[dict[str, Any]] = []
    depth = 0
    in_string = False
    escape = False
    entry_start = -1
    for index in range(start + 1, len(text)):
        char = text[index]
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
                entry_start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and entry_start >= 0:
                try:
                    parsed = json.loads(text[entry_start : index + 1])
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(parsed, dict):
                        entries.append(parsed)
                entry_start = -1
        elif char == "]" and depth == 0:
            break
    return entries


def parse_decisive_dimensions(raw_output: str, judged_dimensions: tuple[str, ...]) -> list[str]:
    """The model's own list of discriminating dimensions, or [] if it did not answer usably.

    Parsed separately from the verdicts so `parse_verifier_output`'s contract is unchanged -- the
    sequential arm calls that function too, and a truncated response must still yield the
    per-candidate verdicts it did finish.
    """
    parsed, _ = parse_json_object(raw_output)
    if not isinstance(parsed, dict):
        return []
    raw = parsed.get("decisive_dimensions")
    if not isinstance(raw, list):
        return []
    allowed = set(judged_dimensions)
    seen: list[str] = []
    for item in raw:
        name = str(item).strip().upper()
        if name in allowed and name not in seen:
            seen.append(name)
    return seen


def parse_verifier_output(
    raw_output: str, candidate_tags: list[str], judged_dimensions: tuple[str, ...] = VERIFIER_DIMENSIONS
) -> tuple[dict[str, dict[str, Any]], bool, str]:
    """Returns (verdicts_by_tag, parse_ok, parse_mode).

    parse_mode is "clean" (whole document parsed), "salvaged" (recovered from a truncated
    array) or "failed" (nothing usable), so a run can be audited for silent truncation rather
    than only for total failure.
    """
    parsed, whole_ok = parse_json_object(raw_output)
    entries = parsed.get("verdicts") if isinstance(parsed, dict) else None
    mode = "clean"
    if not isinstance(entries, list) or not entries:
        entries = salvage_verdict_entries(raw_output)
        mode = "salvaged" if entries else "failed"

    verdicts_by_tag: dict[str, dict[str, Any]] = {}
    allowed = set(candidate_tags)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        tag = normalize_tag(entry.get("tag", ""))
        if tag not in allowed:
            continue
        verdict: dict[str, Any] = {}
        for dimension in judged_dimensions:
            raw_verdict = str(entry.get(dimension, "")).strip().lower()
            verdict[dimension] = VERDICT_TO_MATCHED.get(raw_verdict, None)
        verdict["confidence"] = float(entry.get("confidence", 0.0) or 0.0)
        verdicts_by_tag[tag] = verdict

    if not verdicts_by_tag:
        return {}, False, "failed"
    if mode == "clean" and not whole_ok:
        mode = "salvaged"
    return verdicts_by_tag, True, mode


def load_existing(path: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], int]:
    """Reusable rows only: a row is reusable if it actually carries verdicts.

    Resuming must NOT treat a failed call as done. The truncation bug produced 2,369 rows with
    parse_ok=false and empty verdicts_by_tag; keying resume purely on (fact_id, hypothesis_idx)
    would skip every one of them forever and silently bake the failure into the final table.
    Failed rows are counted and reported, then regenerated.
    """
    existing: dict[tuple[int, int], dict[str, Any]] = {}
    skipped = 0
    if not path.exists():
        return existing, skipped
    for row in stream_jsonl(path):
        key = (int(row["fact_id"]), int(row["hypothesis_idx"]))
        if row.get("parse_ok") and row.get("verdicts_by_tag"):
            existing[key] = row
        else:
            skipped += 1
            existing.pop(key, None)
    return existing, skipped


def main() -> None:
    args = parse_args()
    # "llm"/"all" both mean the deployed set, which is all six. "legacy" reproduces the
    # pre-2026-07-30 three-dimension configuration; nothing in the paper uses it any more.
    JUDGED = LEGACY_JUDGED_DIMENSIONS if args.judge_dimensions == "legacy" else ALL_JUDGED_DIMENSIONS
    normalization_map = load_normalization_map(DEFAULT_NORMALIZATION_MAP)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_calls_path = args.output_dir / "llm_verifier_calls.jsonl"
    verdicts_path = args.output_dir / "llm_verifier_verdicts.json"

    existing, unusable_existing = load_existing(raw_calls_path) if args.resume else ({}, 0)
    if args.resume:
        print(
            f"Resume: {len(existing)} reusable calls, {unusable_existing} previous calls had no "
            "verdicts and will be regenerated.",
            flush=True,
        )
    firing_counts = Counter()
    opportunity_counts = Counter()
    parse_modes = Counter()
    generated = 0
    all_verdicts: list[dict[str, Any]] = []
    generator: QueryGenerator | None = None

    handle = raw_calls_path.open("a" if (args.resume and raw_calls_path.exists()) else "w", encoding="utf-8")
    pending: list[dict[str, Any]] = []
    guard_checked = False

    def accumulate(row: dict[str, Any]) -> None:
        """Fold one call's verdicts into the counters and the verdicts list.

        Order-independent by construction: the counters are Counters, and all_verdicts is
        consumed as a (fact_id, hypothesis_idx, tag) dict by load_llm_verifier_verdicts. That
        is what makes it safe to emit resumed rows immediately while generated rows arrive a
        chunk later.
        """
        for tag in row["candidate_tags"]:
            verdict = row["verdicts_by_tag"].get(tag, {})
            for dimension in JUDGED:
                opportunity_counts[dimension] += 1
                if verdict.get(dimension) is not None:
                    firing_counts[dimension] += 1
            all_verdicts.append(
                {
                    "fact_id": row["fact_id"],
                    "hypothesis_idx": row["hypothesis_idx"],
                    "tag": tag,
                    "verdicts": {dimension: verdict.get(dimension) for dimension in JUDGED},
                }
            )

    def flush_pending() -> None:
        nonlocal generated, guard_checked
        if not pending:
            return
        raw_outputs = generator.generate_many([item["prompt"] for item in pending])
        if len(raw_outputs) != len(pending):
            raise SystemExit(
                f"generate_many returned {len(raw_outputs)} outputs for {len(pending)} prompts; "
                "refusing to pair verdicts with the wrong candidates."
            )
        for item, raw_output in zip(pending, raw_outputs):
            verdicts_by_tag, parse_ok, parse_mode = parse_verifier_output(raw_output, item["candidate_tags"], JUDGED)
            completion_tokens = generator.count_text_tokens(raw_output)
            generated += 1
            parse_modes[parse_mode] += 1
            call = llm_call_record(
                "table5_llm_verifier",
                raw_output=raw_output,
                prompt_tokens=item["prompt_tokens"],
                completion_tokens=completion_tokens,
                parse_ok=parse_ok,
                backend=generator.backend,
                model_name=generator.model_name,
                extra_fields={
                    "used_context_max_chars": item["used_context_chars"],
                    "parse_mode": parse_mode,
                    # True when the response used the entire budget, i.e. it was almost
                    # certainly cut off rather than finishing on its own.
                    "hit_token_cap": completion_tokens >= args.query_max_new_tokens,
                },
            )
            row = {
                "fact_id": item["fact_id"],
                "hypothesis_idx": item["hypothesis_idx"],
                "candidate_tags": item["candidate_tags"],
                "verdicts_by_tag": verdicts_by_tag,
                "parse_ok": parse_ok,
                "parse_mode": parse_mode,
                "call": call,
            }
            if args.ask_decisive_dimensions:
                # Persisted per (fact, hypothesis), which is the grain the model answered at.
                # Empty list means it answered nothing usable -- distinguishable from "it said
                # no dimension discriminates", which is also an empty list only because those
                # two are the same claim for scoring purposes.
                row["decisive_dimensions"] = parse_decisive_dimensions(raw_output, JUDGED)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            accumulate(row)
        handle.flush()
        pending.clear()

        # Batched generation overshoots the exact call count the guard used to trigger on, so
        # it fires on the first chunk that reaches the threshold instead of on equality.
        if args.abort_check_after and not guard_checked and generated >= args.abort_check_after:
            guard_checked = True
            rate = (generated - parse_modes["failed"]) / generated
            if rate < args.abort_if_parse_rate_below:
                raise SystemExit(
                    f"\nABORTING after {generated} calls: only {rate:.0%} produced "
                    f"verdicts (threshold {args.abort_if_parse_rate_below:.0%}).\n"
                    f"parse modes: {dict(parse_modes)}\n"
                    f"Most likely cause is truncation -- check whether raw outputs "
                    f"end mid-JSON and raise --query-max-new-tokens (currently "
                    f"{args.query_max_new_tokens}). Inspect {raw_calls_path}.\n"
                    f"Re-run with --abort-check-after 0 to override."
                )
            print(
                f"Parse-rate guard passed: {rate:.0%} of the first {generated} calls "
                f"produced verdicts ({dict(parse_modes)}).",
                flush=True,
            )

    arm_windows: dict[int, dict[str, Any]] | None = None
    if args.window_tags is not None:
        arm_windows = {}
        for line in args.window_tags.open(encoding="utf-8"):
            entry = json.loads(line)
            arm_windows[int(entry["fact_id"])] = {
                "window_tags": [normalize_tag(t) for t in entry["window_tags"]],
                "hypothesis_indices": [int(i) for i in entry["hypothesis_indices"]],
            }
        if not arm_windows:
            raise SystemExit(f"--window-tags {args.window_tags} is empty")
        print(
            f"per-arm windows: {len(arm_windows)} facts from {args.window_tags.name} "
            f"(--window-source {args.window_source} is overridden)",
            flush=True,
        )

    facts_seen = 0
    try:
        for offset, record in enumerate(stream_jsonl(args.test_trace), start=1):
            if args.limit is not None and offset > args.limit:
                break
            facts_seen += 1
            fact_id = int(record["example_idx"])
            keep_hypotheses: set[int] | None = None
            if arm_windows is not None:
                entry = arm_windows.get(fact_id)
                if entry is None:
                    continue
                # Candidate objects live in the per-round lists as well; a fused head computed
                # from those rounds can contain a tag the record's own top-K final list dropped.
                # Only looking at final_candidates lost 4.9% of the w_cov=0 arm's window.
                pool = list(record.get("final_candidates") or record.get("candidates") or [])
                for round_record in record.get("rounds") or []:
                    pool.extend(round_record.get("candidates") or [])
                by_tag = {normalize_tag(c.get("tag", "")): c for c in pool}
                representatives = [by_tag[tag] for tag in entry["window_tags"] if tag in by_tag]
                if len(representatives) != len(entry["window_tags"]):
                    raise SystemExit(
                        f"fact {fact_id}: {len(entry['window_tags']) - len(representatives)} window "
                        "tags have no candidate object in the trace, so they cannot be put in the "
                        "prompt. stage_arm_windows.py checks this on CPU -- regenerate the window file."
                    )
                keep_hypotheses = {int(i) for i in entry["hypothesis_indices"]}
                # The window came from the file, not from the deployed fused score. An earlier
                # indentation slip let a for/else clause overwrite it with the deployed window on
                # every record, which would have produced plausible verdicts for the wrong window.
                assert [normalize_tag(c.get("tag", "")) for c in representatives] == list(entry["window_tags"]), (
                    f"fact {fact_id}: window does not match the staged file"
                )
            else:
                ranking = window_ranking(record, args.window_source)
                representatives = cluster_representatives(
                    ranking, normalization_map, args.top_m, args.cluster_scan_depth
                )
            candidate_tags = [normalize_tag(candidate["tag"]) for candidate in representatives]

            for hypothesis in record.get("frozen_ags_hypotheses", []):
                hyp_idx = int(hypothesis["hypothesis_idx"])
                if keep_hypotheses is not None and hyp_idx not in keep_hypotheses:
                    continue
                key = (fact_id, hyp_idx)
                if key in existing:
                    accumulate(existing[key])
                    continue

                if generator is None:
                    generator = QueryGenerator(args)
                hint = (
                    symbolic_hint(
                        hypothesis, representatives, normalization_map, args.top_m, args.hint_dimensions
                    )
                    if args.symbolic_hint
                    else ""
                )
                # The hypothesis must show every dimension the prompt refers to -- whether the
                # reference comes from a widened hint or from a widened judged set. Naming a
                # dimension the model was never given a value for is an incoherent prompt.
                # Without either widening, the prompt is unchanged from every prior run.
                hyp_scope = (
                    "all"
                    if (args.symbolic_hint and args.hint_dimensions == "all") or args.judge_dimensions == "all"
                    else "llm"
                )
                prompt, prompt_tokens, used_context_chars = build_prompt_under_query_budget(
                    generator.tokenizer,
                    lambda ctx_chars: build_verifier_messages(
                        record, hypothesis, representatives, ctx_chars, args.candidate_doc_max_chars,
                        hint, hyp_scope, JUDGED, args.ask_decisive_dimensions,
                    ),
                    context_max_chars=args.query_context_max_chars,
                    max_input_tokens=args.query_max_input_tokens,
                )
                pending.append(
                    {
                        "fact_id": fact_id,
                        "hypothesis_idx": hyp_idx,
                        "candidate_tags": candidate_tags,
                        "prompt": prompt,
                        "prompt_tokens": prompt_tokens,
                        "used_context_chars": used_context_chars,
                    }
                )
                if len(pending) >= args.generation_chunk:
                    flush_pending()

            if offset % args.log_every == 0:
                print(f"LLM verifier: {offset} facts processed ({generated} calls)", flush=True)

        flush_pending()
    finally:
        handle.close()
        if generator is not None:
            generator.close()

    firing_rate = {
        dimension: round(firing_counts[dimension] / opportunity_counts[dimension], 6) if opportunity_counts[dimension] else None
        for dimension in JUDGED
    }
    verdicts_path.write_text(json.dumps(all_verdicts, ensure_ascii=False) + "\n", encoding="utf-8")
    calls_with_verdicts = generated - parse_modes["failed"]
    parse_rate = round(calls_with_verdicts / generated, 6) if generated else None
    summary = {
        "verdicts_path": str(verdicts_path),
        "raw_calls_path": str(raw_calls_path),
        "top_m": args.top_m,
        "symbolic_hint": bool(args.symbolic_hint),
        "judge_dimensions": list(JUDGED),
        "hint_dimensions": args.hint_dimensions if args.symbolic_hint else None,
        "window_source": args.window_source,
        # Which arm these verdicts belong to. Without this a per-arm verdict file is
        # indistinguishable from the deployed one, and scoring an arm with the wrong verdicts
        # is exactly the mistake this flag exists to prevent.
        "window_tags_path": str(args.window_tags) if args.window_tags else None,
        "query_max_new_tokens": args.query_max_new_tokens,
        "firing_counts": dict(firing_counts),
        "opportunity_counts": dict(opportunity_counts),
        "firing_rate_per_dimension": firing_rate,
        # Read this BEFORE interpreting firing_rate. A parse failure and a genuine abstention
        # both drive the firing rate to zero, and they mean opposite things.
        "calls_generated": generated,
        "parse_modes": dict(parse_modes),
        "parse_rate": parse_rate,
        "resume_regenerated_unusable": unusable_existing,
        "note": (
            "A near-zero firing rate here (Appendix H found 5/1,160 FAMILY opportunities) "
            "means row 3.9 will show little difference from AGS full for a real reason -- "
            "abstention, not disagreement -- and that is the finding, not a bug. "
            "That reading is ONLY valid when parse_rate is high: a truncated or unparseable "
            "response also yields a zero firing rate, and that IS a bug. Check parse_rate and "
            "parse_modes first; if parse_modes['salvaged'] is large, responses are being cut "
            "off and --query-max-new-tokens needs raising."
        ),
    }
    (args.output_dir / "llm_verifier_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)

    # COMPLETENESS. Both files are already on disk above, so a shortfall here loses no work and a
    # resume picks up where this run stopped -- but it must NOT exit 0, because the downstream
    # listwise rerank is chained with `afterok` and would otherwise consume a partial verdict set.
    # That failure mode has already cost one GPU allocation: a ranking staged from 42,910 of 50,180
    # keys looked completely normal in its own summary (see the --top-m guard in
    # dump_reranked_ranking.py). Silent partial coverage is the thing to make impossible.
    if args.window_tags is not None and args.limit is None:
        expected = sum(
            len(entry["window_tags"]) * len(entry["hypothesis_indices"])
            for entry in (arm_windows or {}).values()
        )
        got = len(all_verdicts)
        print(f"completeness: {got}/{expected} (fact, hypothesis, tag) verdicts", flush=True)
        if got != expected:
            raise SystemExit(
                f"INCOMPLETE: {got} of {expected} verdict keys were written, {expected - got} "
                f"short of {args.window_tags.name}'s window. Both output files are intact -- rerun "
                "the same command with --resume to finish; do not score an arm from this file."
            )
    elif args.limit is None:
        trace_facts = sum(1 for _ in stream_jsonl(args.test_trace))
        print(f"completeness: {facts_seen}/{trace_facts} facts visited", flush=True)
        if facts_seen != trace_facts:
            raise SystemExit(
                f"INCOMPLETE: {facts_seen} of {trace_facts} trace facts were visited. Both output "
                "files are intact -- rerun with --resume to finish; do not score from this file."
            )


if __name__ == "__main__":
    main()
