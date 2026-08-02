#!/usr/bin/env python3
"""Verifier ablations: which verifier supplies the rerank term.

Five arms over ONE frozen hypothesis set and ONE candidate pool per fact. Nothing is
regenerated and nothing is re-retrieved -- every arm is a different scoring rule applied to
the same `runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags` trace, so the arms are
paired by construction and a difference between them cannot be sampling noise in generation
or retrieval.

All arms run at the DEPLOYED truncation order (truncate_pool_to_top_k=True): the fused pool is
cut to top_k before the consensus rerank, exactly as run_fintagging_grounding_baseline does.
Under that setting the deterministic arm reproduces
runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags/metrics.json's retrieval stage to
0.000000 on Recall@10/50/200, MRR and top-1, so these arms are the deployed pipeline with one
component swapped rather than a reimplementation of it.

    hybrid            deterministic dimension agreement + candidate-level LLM verdicts, the
                      LLM substituting on FAMILY/ROLE/EVENT for the M candidates it saw and
                      the symbolic verdict covering everything else. This is the full method.
    - LLM verifier    deterministic only (what the paper previously called "AGS (full)").
    - determ. verif.  LLM dimensions only, abstentions dropped from the average.
    LLM verifier only LLM dimensions only, abstentions counted as non-support.
    no verifier       beta=0: the fused retrieval score alone.

WHY TWO LLM-ONLY ARMS
    "Disable the deterministic term while keeping LLM verification" and "rank by retrieval
    fusion plus the LLM verifier" describe the same configuration unless the abstention rule
    differs, and the verifier abstains on 44% of dimension opportunities, so the rule is not
    a detail. Both readings are reported rather than one being chosen silently. Collapse them
    if the paper only needs one.

WHAT THIS SCRIPT DOES NOT MEASURE
    Final reranked accuracy. Every number here is retrieval-stage, before the listwise
    reranker. The reranked column needs a GPU pass per arm; dump the arm's ranking with
    verifier/dump_reranked_ranking.py --verifier-mode and feed it to
    run_fintagging_grounding_baseline.py --reuse-candidates --run-rerank.

CPU only.
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
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
FHS_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map  # noqa: E402
from verifier.core import AblationConfig, aggregate, evaluate, reset_consensus_cache  # noqa: E402
from verifier.data_prep import DEFAULT_TEST_TRACE, load_test_facts  # noqa: E402
from verifier.run_llm_verifier import ALL_JUDGED_DIMENSIONS, VERIFIER_DIMENSIONS  # noqa: E402
from verifier.run_test_rows import load_llm_verifier_verdicts, rows_for_modality  # noqa: E402
from compute_ags_seq_arm_metrics import paired_bootstrap  # noqa: E402
from run_fintagging_grounding_baseline import normalize_tag  # noqa: E402

DEFAULT_VERDICTS = FHS_ROOT / "runs" / "runs_ags_table5_ablation" / "qwen3_32b" / "llm_verifier_verdicts.json"
DEFAULT_CALLS = FHS_ROOT / "runs" / "runs_ags_table5_ablation" / "qwen3_32b" / "llm_verifier_calls.jsonl"
DEFAULT_OUTPUT_DIR = FHS_ROOT / "runs" / "runs_ags_verifier_ablation" / "qwen3_32b"

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260724  # same seed as run_test_rows.py, so CIs are comparable to Table 5
METRICS = ("recall_at_10", "recall_at_20", "recall_at_50", "recall_at_200", "mrr", "top1_accuracy")

# name -> (verifier_mode, beta, uses_llm). The baseline every CI is measured against is first.
ARMS: tuple[tuple[str, str, float, bool], ...] = (
    ("Hybrid AGS (full)", "hybrid", 0.6, True),
    ("- LLM verifier", "deterministic", 0.6, False),
    ("- deterministic verifier", "llm_drop", 0.6, True),
    ("LLM verifier only", "llm_strict", 0.6, True),
    ("- both verifiers", "deterministic", 0.0, False),
    # Scope-matched control. "- deterministic verifier" vs "Hybrid AGS (full)" compares a term
    # reaching 10 of 200 candidates against one reaching all 200, then scores it at top-1 --
    # the one rank the narrow term was aimed at. This arm gives the symbolic verdict the LLM's
    # window and fill rule, so the contrast against "- deterministic verifier" varies only the
    # verdict source. It consumes the verdicts file for its KEYS; no LLM judgement is scored,
    # hence uses_llm=False for the cost columns.
    ("Deterministic, window-matched", "det_window", 0.6, False),
)
BASELINE_ARM = ARMS[0][0]
LLM_DIMS: tuple[str, ...] = VERIFIER_DIMENSIONS  # set from --llm-dimensions in main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-trace", type=Path, default=DEFAULT_TEST_TRACE)
    parser.add_argument("--verdicts", type=Path, default=DEFAULT_VERDICTS)
    parser.add_argument("--calls", type=Path, default=DEFAULT_CALLS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--normalization-map", type=Path, default=DEFAULT_NORMALIZATION_MAP)
    # Must match the --judge-dimensions the verdicts were generated with. Verdicts carrying
    # QUALIFIER/SCOPE/TEMPORAL scored under the default 3-tuple would silently drop half the
    # judgements and look like a weaker version of the 3-dimension arm rather than a control.
    parser.add_argument(
        "--llm-dimensions",
        choices=("llm", "all"),
        default="llm",
        help="Dimensions read from the verdicts file. 'llm' is FAMILY/ROLE/EVENT; use 'all' for "
        "verdicts generated with run_llm_verifier.py --judge-dimensions all.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--sensitivity-top-m",
        type=int,
        nargs="*",
        default=[],
        help="Candidate-window sizes to evaluate by TRUNCATING the logged call windows. This is a "
        "diagnostic, not a window-size experiment: a verifier shown five candidates sees a "
        "different prompt and may judge differently than one shown ten and truncated to five. "
        "The reported K_v sensitivity uses one generation run per window, each passed via "
        "--verdicts/--calls. Defaults to empty so a truncated row is never produced by accident.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_call_windows(path: Path) -> tuple[dict[tuple[int, int], list[str]], dict[str, Any]]:
    """(fact_id, hyp_idx) -> the ordered candidate tags that call actually scored, plus cost.

    The window order is the retrieval order the verifier was shown, so truncating it to the
    first M reproduces what a smaller-M run would have judged -- for the M it was generated
    at or below. It cannot manufacture a larger window.
    """
    windows: dict[tuple[int, int], list[str]] = {}
    prompt_tokens = 0
    completion_tokens = 0
    calls = 0
    max_window = 0
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        key = (int(record["fact_id"]), int(record["hypothesis_idx"]))
        tags = [normalize_tag(tag) for tag in record.get("candidate_tags") or []]
        windows[key] = tags
        max_window = max(max_window, len(tags))
        call = record.get("call") or {}
        prompt_tokens += int(call.get("prompt_tokens") or 0)
        completion_tokens += int(call.get("completion_tokens") or 0)
        calls += 1
    cost = {
        "calls": calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "generated_top_m": max_window,
    }
    return windows, cost


def restrict_verdicts(
    verdicts: dict[tuple[int, int, str], dict[str, Any]],
    windows: dict[tuple[int, int], list[str]],
    top_m: int,
) -> dict[tuple[int, int, str], dict[str, Any]]:
    """Keep only verdicts on the first `top_m` candidates of each call's own window."""
    keep: dict[tuple[int, int, str], dict[str, Any]] = {}
    for (fact_id, hyp_idx), tags in windows.items():
        for tag in tags[:top_m]:
            key = (fact_id, hyp_idx, tag)
            if key in verdicts:
                keep[key] = verdicts[key]
    return keep


def rows_by_fact(rows: list[dict[str, Any]], metric: str) -> dict[int, dict[str, Any]]:
    return {int(row["fact_id"]): {"context_id": row["context_id"], metric: float(row[metric])} for row in rows}


def bootstrap_rows(
    variant: str,
    arm_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    samples: int,
    seed: int,
    extra: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for modality in ("pooled", "table", "text"):
        subset = rows_for_modality(arm_rows, modality)
        baseline_subset = rows_for_modality(baseline_rows, modality)
        if not subset:
            continue
        agg = aggregate(subset)
        for metric in METRICS:
            ci = paired_bootstrap(
                rows_by_fact(subset, metric), rows_by_fact(baseline_subset, metric), metric, samples, seed
            )
            row = {
                "variant": variant,
                "modality": modality,
                "metric": metric,
                "value": agg[metric],
                "delta_vs_full": ci.get("mean_difference"),
                "ci_low": ci.get("ci_low"),
                "ci_high": ci.get("ci_high"),
                "ci_excludes_zero": ci.get("ci_excludes_zero"),
                "n_facts": ci.get("facts", len(subset)),
                "n_contexts": ci.get("contexts"),
            }
            row.update(extra)
            out.append(row)
    return out


def slugify(name: str) -> str:
    keep = [char.lower() if char.isalnum() else "_" for char in name]
    return "".join(keep).strip("_").replace("__", "_")


def write_per_fact(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "fact_id": row["fact_id"],
                        "context_id": row["context_id"],
                        "modality": row["modality"],
                        "rank": row["rank"],
                    }
                )
                + "\n"
            )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    global LLM_DIMS
    LLM_DIMS = ALL_JUDGED_DIMENSIONS if args.llm_dimensions == "all" else VERIFIER_DIMENSIONS
    print(f"scoring LLM verdicts over: {', '.join(LLM_DIMS)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    normalization_map = load_normalization_map(args.normalization_map)
    verdicts = load_llm_verifier_verdicts(args.verdicts)
    if not verdicts:
        raise SystemExit(f"no verdicts loaded from {args.verdicts}")
    windows, call_cost = load_call_windows(args.calls)
    print(f"verdicts {len(verdicts)}  calls {call_cost['calls']}  window {call_cost['generated_top_m']}", flush=True)

    facts = list(load_test_facts(args.test_trace).values())
    if args.limit:
        facts = facts[: args.limit]
    n_facts = len(facts)
    print(f"test facts: {n_facts}", flush=True)

    def run(name: str, config: AblationConfig) -> tuple[list[dict[str, Any]], float]:
        reset_consensus_cache()
        started = time.time()
        rows = [evaluate(fact, config, normalization_map) for fact in facts]
        elapsed = time.time() - started
        print(f"  {name:<26} {elapsed:7.1f}s", flush=True)
        return rows, elapsed

    # ---- the five arms, all at the generated window ---------------------------------------
    print("\n=== verifier arms ===", flush=True)
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    arm_cost: list[dict[str, Any]] = []
    for name, mode, beta, uses_llm in ARMS:
        config = AblationConfig(
            name=name,
            beta=beta,
            verifier_mode=mode,
            truncate_pool_to_top_k=True,
            # Neutral fill for candidates the verifier never saw, so an LLM-only arm tests
            # "which verdict source" rather than "verdict source plus a top-K_v prior".
            llm_unjudged_fill="mean",
            llm_verifier_dimensions=LLM_DIMS,
            llm_verifier_verdicts=verdicts if mode != "deterministic" else None,
        )
        rows, elapsed = run(name, config)
        arm_rows[name] = rows
        # Persist the gold rank per fact. Every metric in this file is a function of it, so a
        # later question -- another depth, a re-bootstrap against a different baseline -- is a
        # read rather than another 90-minute pass over the trace.
        write_per_fact(args.output_dir / "per_fact" / f"{slugify(name)}.jsonl", rows)
        arm_cost.append(
            {
                "variant": name,
                "verifier_mode": mode,
                "beta": beta,
                "top_m": call_cost["generated_top_m"] if uses_llm else None,
                "verifier_llm_calls_per_fact": round(call_cost["calls"] / n_facts, 4) if uses_llm else 0.0,
                "verifier_prompt_tokens_per_fact": round(call_cost["prompt_tokens"] / n_facts, 1) if uses_llm else 0.0,
                "verifier_completion_tokens_per_fact": (
                    round(call_cost["completion_tokens"] / n_facts, 1) if uses_llm else 0.0
                ),
                "scoring_cpu_sec_total": round(elapsed, 1),
                "scoring_cpu_sec_per_fact": round(elapsed / n_facts, 4),
            }
        )

    results: list[dict[str, Any]] = []
    for name, _, _, _ in ARMS:
        results.extend(
            bootstrap_rows(
                name, arm_rows[name], arm_rows[BASELINE_ARM], args.bootstrap_samples, args.seed, extra={}
            )
        )
    write_csv(args.output_dir / "verifier_ablation.csv", results)
    write_csv(args.output_dir / "verifier_ablation_cost.csv", arm_cost)

    # ---- M sensitivity, hybrid arm only ----------------------------------------------------
    print("\n=== candidate-window sensitivity (hybrid arm) ===", flush=True)
    sensitivity: list[dict[str, Any]] = []
    generated_m = call_cost["generated_top_m"]
    unavailable: list[int] = []
    for top_m in sorted(args.sensitivity_top_m):
        if top_m > generated_m:
            unavailable.append(top_m)
            print(f"  M={top_m:<3} SKIPPED: needs its own generation run (verdicts cover M={generated_m})", flush=True)
            continue
        restricted = restrict_verdicts(verdicts, windows, top_m)
        config = AblationConfig(
            name=f"hybrid M={top_m}",
            beta=0.6,
            verifier_mode="hybrid",
            llm_verifier_top_m=top_m,
            llm_verifier_verdicts=restricted,
        )
        rows, elapsed = run(f"M={top_m}", config)
        sensitivity.extend(
            bootstrap_rows(
                f"Hybrid AGS (M={top_m})",
                rows,
                arm_rows[BASELINE_ARM],
                args.bootstrap_samples,
                args.seed,
                extra={
                    "top_m": top_m,
                    "is_default": top_m == generated_m,
                    "verdicts_retained": len(restricted),
                    "cost_measured": top_m == generated_m,
                },
            )
        )
    write_csv(args.output_dir / "verifier_window_sensitivity.csv", sensitivity)

    # ---- the 2x2 interaction study, retrieval-stage half -----------------------------------
    # The reranker-on cells are a GPU pass over these same rankings; this records the two
    # reranker-off cells and the pairing that makes the other two interpretable.
    interaction = [
        {
            "llm_verifier": "off",
            "listwise_reranker": "off",
            "source_arm": "- LLM verifier",
            "stage": "retrieval",
            **{metric: aggregate(arm_rows["- LLM verifier"])[metric] for metric in METRICS},
        },
        {
            "llm_verifier": "on",
            "listwise_reranker": "off",
            "source_arm": BASELINE_ARM,
            "stage": "retrieval",
            **{metric: aggregate(arm_rows[BASELINE_ARM])[metric] for metric in METRICS},
        },
    ]
    write_csv(args.output_dir / "interaction_retrieval_stage.csv", interaction)

    summary = {
        "n_facts": n_facts,
        "n_contexts": len({row["context_id"] for row in arm_rows[BASELINE_ARM]}),
        "baseline_arm": BASELINE_ARM,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "verdicts_path": str(args.verdicts),
        "calls_path": str(args.calls),
        "generated_top_m": generated_m,
        "llm_unjudged_fill": "mean",
        "sensitivity_unavailable_top_m": unavailable,
        "verifier_call_cost_total": call_cost,
        "pooled": {
            name: {metric: aggregate(arm_rows[name])[metric] for metric in METRICS} for name, _, _, _ in ARMS
        },
        "caveats": [
            "Every metric here is retrieval-stage, before the listwise reranker.",
            "Truncated windows are a diagnostic only. A real K_v sensitivity row comes from a "
            "generation run at that window, because the verifier's prompt -- and therefore its "
            "judgement -- depends on how many candidates it was shown.",
            "'- deterministic verifier' and 'LLM verifier only' differ only in whether an "
            "abstention is dropped from the average or counted as non-support.",
        ],
    }
    (args.output_dir / "verifier_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\n=== pooled, retrieval stage ===", flush=True)
    header = f"{'variant':<26} " + " ".join(f"{metric:>14}" for metric in METRICS)
    print(header, flush=True)
    for name, _, _, _ in ARMS:
        agg = aggregate(arm_rows[name])
        print(f"{name:<26} " + " ".join(f"{agg[metric]:>14.6f}" for metric in METRICS), flush=True)
    print(f"\nwrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
