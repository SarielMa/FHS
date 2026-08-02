#!/usr/bin/env python3
"""Materialise the candidate-level-LLM-reranked ranking as a candidates file.

`run_verifier_row_only.py` already computes this ranking -- evaluate() builds it in
`ranked` (core.py:369) -- but keeps only the aggregate metrics, so the ranking itself
is discarded. The deployed-stage measurement needs it on disk: the listwise reranker
reranks whatever sits at OUTPUT_DIR/bm25_candidates.jsonl under --reuse-candidates.

This script re-runs the same evaluate() under the same config and persists the per-fact
ranking in the schema run_fintagging_grounding_baseline.py expects.

HOW THE RECORDS ARE BUILT
    The ranking that evaluate() returns is a list of tags. Full candidate objects
    (documentation, retrieval_text, type, references) are taken verbatim from the frozen
    AGS trace's own per-round candidate lists, which is where they were logged. Only the
    order and the `rank` field change; no field is re-derived, so the listwise reranker
    sees exactly the candidate text it would have seen, in the new order.

    `query_mode` is preserved as the trace's own value because validate_candidate_records
    (run_fintagging_grounding_baseline.py:883) rejects a mismatch against --query-mode.

CPU only. No GPU, no generation, no retrieval.
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
import sys
from pathlib import Path
from typing import Any

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map  # noqa: E402
from verifier.core import (  # noqa: E402
    VERIFIER_MODES,
    AblationConfig,
    evaluate,
    reset_consensus_cache,
)
from verifier.data_prep import DEFAULT_TEST_TRACE, load_test_facts, stream_jsonl  # noqa: E402
from verifier.run_test_rows import load_llm_verifier_verdicts  # noqa: E402
from run_ags_verifier_ablation import load_call_windows, restrict_verdicts  # noqa: E402
from run_fintagging_grounding_baseline import normalize_tag  # noqa: E402


DEFAULT_VERDICTS = _PARENT / "runs_ags_table5_ablation" / "qwen3_32b" / "llm_verifier_verdicts.json"
DEFAULT_CALLS = _PARENT / "runs_ags_table5_ablation" / "qwen3_32b" / "llm_verifier_calls.jsonl"
DEFAULT_OUTPUT = (
    _PARENT / "runs_ags_verifier_bridge" / "qwen3_32b" / "candidate_level_reranked_candidates.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-trace", type=Path, default=DEFAULT_TEST_TRACE)
    parser.add_argument("--verdicts", type=Path, default=DEFAULT_VERDICTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--normalization-map", type=Path, default=DEFAULT_NORMALIZATION_MAP)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument(
        "--verifier-mode",
        choices=VERIFIER_MODES,
        default="auto",
        help="Which verifier supplies the rerank term. 'auto' keeps the original behaviour "
        "(hybrid, since verdicts are always loaded here). Use --beta 0 for the no-verifier arm.",
    )
    parser.add_argument(
        "--top-m",
        type=int,
        default=None,
        help="Restrict the LLM verdicts to the first M candidates of each logged call window. "
        "Defaults to the window the verdicts were generated at. Cannot exceed it.",
    )
    parser.add_argument("--calls", type=Path, default=DEFAULT_CALLS, help="Call log, needed only for --top-m.")
    parser.add_argument(
        "--llm-unjudged-fill",
        choices=("mean", "zero"),
        default="mean",
        help="What an LLM-only arm (llm_drop / llm_strict) scores a candidate the verifier never "
        "saw. Must match run_ags_verifier_ablation.py, which uses 'mean', or this arm's final "
        "accuracy is computed from a different ranking than its own retrieval-stage row. "
        "AblationConfig defaults to 'zero'; that default is deliberately NOT inherited here.",
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Rerank the whole fused pool and truncate afterwards. This is NOT the deployed "
        "order and inflates Recall@200 by 0.0128; kept only to reproduce the original bridge run.",
    )
    # --- tab:ablation's deterministic-core diagnostic arms -------------------------------------
    # These reach the same evaluate() through AblationConfig knobs rather than the verifier
    # ones, so their rankings can be dumped and reranked exactly like the verifier arms. Each
    # default below is AblationConfig's own, so omitting all of them changes nothing.
    parser.add_argument(
        "--n-hypotheses",
        type=int,
        default=2,
        help="J. Use 1 with --kept-hypothesis-idx for the -ensemble arm.",
    )
    parser.add_argument(
        "--kept-hypothesis-idx",
        type=int,
        default=None,
        help="Which hypothesis survives when --n-hypotheses 1. Required in that case. The "
        "paper's -ensemble row is the MEAN of idx 0 and idx 1, so it needs one dump and one "
        "rerank per index, averaged afterwards -- there is no single ranking for that row.",
    )
    parser.add_argument(
        "--renderings",
        default="def,lab",
        help="Comma-separated rendering forms. 'def' alone is the -label-form arm; 'lab' alone "
        "is the -definition-form arm.",
    )
    parser.add_argument(
        "--lab-only-fallback",
        choices=("none", "def"),
        default="none",
        help="For a lab-only rendering, what a label-less concept falls back to. 'none' is the "
        "zero-recall variant, which is the one tab:ablation reports.",
    )
    parser.add_argument(
        "--llm-verifier-dimensions",
        default=None,
        help=(
            "Comma-separated dimensions the SCORE averages over. Unset keeps "
            "LLM_VERIFIER_DIMENSIONS_DEFAULT, so every existing invocation is unchanged. This is "
            "the scoring set only; which dimensions the verifier was ASKED about is fixed in the "
            "verdicts file and cannot be changed here."
        ),
    )
    parser.add_argument("--fusion", choices=("sum", "mean"), default="sum")
    parser.add_argument("--scaling", choices=("range", "raw"), default="range")
    parser.add_argument(
        "--oracle-best-single",
        action="store_true",
        help="Oracle row: keep the best of the J hypotheses per fact using gold.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Where to write the run summary. Defaults to candidate_level_reranked_summary.json "
        "beside the output, which is the name the original single-arm run used.",
    )
    return parser.parse_args()


def candidate_pool(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """tag -> full candidate object, from everywhere the trace logged one.

    Rounds first (they carry every retrieved candidate with full fields), then the
    consolidated lists, so a tag that only survives consolidation is still resolvable.
    """
    pool: dict[str, dict[str, Any]] = {}
    for round_record in record.get("rounds") or []:
        for candidate in round_record.get("candidates") or []:
            pool.setdefault(normalize_tag(candidate.get("tag", "")), candidate)
    for key in ("final_candidates", "candidates"):
        for candidate in record.get(key) or []:
            pool.setdefault(normalize_tag(candidate.get("tag", "")), candidate)
    return pool


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for path in (args.test_trace, args.verdicts):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    normalization_map = load_normalization_map(args.normalization_map)
    # The deterministic-core arms never consult a verdict, so requiring the verdicts file for
    # them would make an unrelated missing input block the whole diagnostic block.
    if args.verifier_mode == "deterministic":
        verdicts = {}
        print("verifier_mode=deterministic: no verdicts needed", flush=True)
    else:
        verdicts = load_llm_verifier_verdicts(args.verdicts)
        if not verdicts:
            raise SystemExit(f"No verdicts loaded from {args.verdicts}")
        print(f"verdicts loaded: {len(verdicts)}", flush=True)

    facts = load_test_facts(args.test_trace, limit=args.limit)
    print(f"test facts: {len(facts)}", flush=True)

    generated_top_m = None
    restricted_away = 0
    if args.top_m is not None:
        # --top-m restricts the verdicts through --calls, so the two files must come from the SAME
        # generation run. They did not, silently, for every llmonly_* arm of
        # apply_server_verifier_ablation_rerank.sh: --calls was left at DEFAULT_CALLS, the
        # 2026-07-25 deterministic-window log, so fused-window verdicts were filtered by
        # membership in the deterministic window. Measured: verdicts_k10_judge6 50,180 -> 42,910
        # keys, and 68% of facts changed their top-10. Refuse the default pairing outright.
        # DEFAULT_CALLS pairs with DEFAULT_VERDICTS -- that combination is one run and is fine
        # (hybrid_m5 uses it to shrink a window on purpose). Only the mixed pair is the defect.
        if args.calls == DEFAULT_CALLS and args.verdicts != DEFAULT_VERDICTS:
            raise SystemExit(
                "--top-m needs --calls pointing at the call log of the SAME run as --verdicts "
                f"(currently the default, {DEFAULT_CALLS.name} from the 2026-07-25 "
                "deterministic-window generation). Pairing a fused-window verdicts file with "
                "that log filters the verdicts through the deterministic window and reproduces "
                "the confound the fused windows exist to remove. Verdicts already carry the "
                "window they were generated at, so the usual fix is to drop --top-m entirely; "
                "pass it only to SHRINK a window on purpose, with that run's own --calls."
            )
        windows, call_cost = load_call_windows(args.calls)
        generated_top_m = call_cost["generated_top_m"]
        if args.top_m > generated_top_m:
            raise SystemExit(
                f"--top-m {args.top_m} exceeds the window these verdicts were generated at "
                f"({generated_top_m}); a larger window needs its own generation run."
            )
        before = len(verdicts)
        verdicts = restrict_verdicts(verdicts, windows, args.top_m)
        restricted_away = before - len(verdicts)
        print(f"verdicts restricted to M={args.top_m}: {len(verdicts)}", flush=True)
        if args.top_m == generated_top_m and restricted_away:
            # A no-op restriction that is not a no-op means the two files disagree about the
            # window, which is the same defect with a matching --calls path.
            raise SystemExit(
                f"--top-m {args.top_m} equals the generated window, so it should drop nothing, "
                f"but {restricted_away} of {before} verdict keys are absent from "
                f"{args.calls}. The call log and the verdicts file are from different runs."
            )

    renderings = tuple(part.strip() for part in args.renderings.split(",") if part.strip())
    if not renderings:
        raise SystemExit("--renderings must name at least one form")
    if args.n_hypotheses == 1 and args.kept_hypothesis_idx is None:
        raise SystemExit("--n-hypotheses 1 requires --kept-hypothesis-idx (evaluate() rejects it otherwise)")

    uses_llm = args.verifier_mode != "deterministic"
    config = AblationConfig(
        name=f"verifier_mode={args.verifier_mode} beta={args.beta}",
        beta=args.beta,
        verifier_mode=args.verifier_mode,
        n_hypotheses=args.n_hypotheses,
        kept_hypothesis_idx=args.kept_hypothesis_idx,
        renderings=renderings,
        lab_only_fallback=None if args.lab_only_fallback == "none" else args.lab_only_fallback,
        fusion=args.fusion,
        scaling=args.scaling,
        oracle_best_single=args.oracle_best_single,
        # Deployed truncation order: cut the fused pool to top_k before reranking. Under this
        # setting the deterministic arm reproduces the deployed retrieval stage to 0.000000,
        # so the ranking handed to the listwise reranker is the one the pipeline would build.
        truncate_pool_to_top_k=not args.no_truncate,
        llm_verifier_top_m=args.top_m or generated_top_m or 10,
        llm_verifier_verdicts=verdicts if uses_llm else None,
        llm_unjudged_fill=args.llm_unjudged_fill,
        **(
            {"llm_verifier_dimensions": tuple(
                d.strip().upper() for d in args.llm_verifier_dimensions.split(",") if d.strip())}
            if args.llm_verifier_dimensions else {}
        ),
    )
    reset_consensus_cache()
    ranking_by_fact: dict[int, list[str]] = {}
    for index, (fact_id, fact) in enumerate(sorted(facts.items()), start=1):
        row = evaluate(fact, config, normalization_map)
        ranking_by_fact[fact_id] = [normalize_tag(tag) for tag in row.get("candidate_tags") or []]
        if args.log_every and index % args.log_every == 0:
            print(f"ranked {index}/{len(facts)} facts", flush=True)

    written = 0
    unresolved_total = 0
    changed_order = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for record in stream_jsonl(args.test_trace):
            fact_id = int(record["example_idx"])
            ranking = ranking_by_fact.get(fact_id)
            if ranking is None:
                continue
            pool = candidate_pool(record)
            original = [normalize_tag(c.get("tag", "")) for c in (record.get("candidates") or [])]

            rebuilt: list[dict[str, Any]] = []
            unresolved = 0
            for tag in ranking:
                candidate = pool.get(tag)
                if candidate is None:
                    # A tag the trace never logged a full object for cannot be shown to the
                    # reranker. Count it rather than emitting a stub with empty text.
                    unresolved += 1
                    continue
                entry = dict(candidate)
                entry["rank"] = len(rebuilt) + 1
                rebuilt.append(entry)
                if len(rebuilt) >= args.top_k:
                    break
            unresolved_total += unresolved

            out = dict(record)
            # gold_rank recomputed for the NEW order. Copying the trace's value made it look as
            # though the rerank changed nothing (measured: identical on every fact, while the
            # actual order changed on 711 of 2,509), which is a diagnostic trap even though no
            # reported metric reads this field -- every metric is computed from the order itself.
            gold = {normalize_tag(t) for t in (record.get("gold_tags") or [])}
            out["gold_rank"] = next(
                (entry["rank"] for entry in rebuilt if normalize_tag(entry.get("tag", "")) in gold),
                None,
            )
            out["candidates"] = rebuilt
            # Keep the trace's own consolidated view available for diagnostics, but the
            # reranker reads `candidates`.
            out["final_candidates"] = rebuilt
            out["candidate_level_llm_reranked"] = True
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
            if [c["tag"] for c in rebuilt] != original:
                changed_order += 1

    summary = {
        "output": str(args.output),
        "facts_written": written,
        "facts_with_changed_ranking": changed_order,
        "unresolved_tags_dropped": unresolved_total,
        "beta": args.beta,
        "top_k": args.top_k,
        "verifier_mode": args.verifier_mode,
        "top_m": args.top_m or generated_top_m,
        "llm_verdicts_used": len(verdicts) if uses_llm else 0,
        # Which verdicts file, and how many keys --top-m removed. `llm_verdicts_used` alone cannot
        # distinguish "this file has N keys" from "this file had more and was filtered"; that
        # ambiguity is what made the det-window contamination invisible in a summary.
        "llm_verdicts_path": str(args.verdicts) if uses_llm else None,
        "llm_verdicts_dropped_by_top_m": restricted_away if uses_llm else 0,
        "calls_path": str(args.calls) if args.top_m is not None else None,
        "llm_unjudged_fill": args.llm_unjudged_fill if uses_llm else None,
        # Which dimensions were SCORED. Not recoverable from the verdicts file, which records
        # only which were ASKED -- and the two differ whenever a scoring set is pinned. Without
        # this an arm's own metadata cannot distinguish score3 from score6.
        "llm_verifier_dimensions": list(config.llm_verifier_dimensions) if uses_llm else None,
        "note": (
            "Candidate objects are copied verbatim from the trace; only order and rank change. "
            "Feed to run_fintagging_grounding_baseline.py with --reuse-candidates."
        ),
    }
    summary_path = args.summary or (args.output.parent / "candidate_level_reranked_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
