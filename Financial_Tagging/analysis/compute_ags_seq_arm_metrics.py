#!/usr/bin/env python3
"""Sequential-arm reporting (ags_seq_arms_spec.md section 5).

Reads the candidate traces of the three arms that share a row in the paper's sequential
table -- AGS (frozen, one round), AGS+Seq, AGS+Seq-random -- and emits

    per_fact.jsonl   fact_id, context_id, modality, arm, round1/final candidates and ranks,
                     realized_rounds, stop_reason
    rounds.jsonl     the per-round controller records, one row per (fact, arm, round)
    arm_summary.csv  arm x modality metrics, the round1-vs-full column, realized rounds, AULC

and metrics.json holding the three contrasts section 6 asks for. Metrics are paired per
fact and bootstrapped by resampling source contexts (2,000 iterations by default), because
this benchmark puts ~21 facts under one table and a fact-level bootstrap would understate
the interval. Table and text are reported separately as well as pooled.

AULC is the mean R@50 over rounds 1..B of a fact's episode, with the last realized value
carried forward for episodes that stopped early -- an arm that stops at round 2 is credited
with what it had at round 2 for the rounds it did not spend.
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
from pathlib import Path
from typing import Any

import numpy as np

from run_fintagging_grounding_baseline import SCRIPT_DIR, first_gold_rank, normalize_tag


DEFAULT_RUNS_ROOT = FHS_ROOT / "runs" / "runs_fintagging_grounding_baseline"
DEPTHS = (10, 50, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--ags-dir", type=Path, default=None, help="Default: <runs-root>/qwen3_32b_frozen_ags")
    parser.add_argument("--seq-dir", type=Path, default=None, help="Default: <runs-root>/qwen3_32b_ags_seq")
    parser.add_argument(
        "--seq-random-dir", type=Path, default=None, help="Default: <runs-root>/qwen3_32b_ags_seq_random"
    )
    parser.add_argument("--output-dir", type=Path, default=FHS_ROOT / "runs" / "runs_ags_seq" / "qwen3_32b")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    return parser.parse_args()


def stream_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def recall_at(tags: list[str], gold_tags: list[str], depth: int) -> float:
    rank = first_gold_rank(tags, gold_tags)
    return 1.0 if rank is not None and rank <= depth else 0.0


def fact_row(record: dict[str, Any], arm: str, max_rounds: int) -> dict[str, Any]:
    """One per_fact.jsonl row, tolerant of the single-round AGS arm."""
    gold_tags = [normalize_tag(tag) for tag in record.get("gold_tags", [])]
    final_tags = [normalize_tag(tag) for tag in record.get("candidate_union_tags", [])]
    round1_tags = [normalize_tag(tag) for tag in record.get("round1_candidates", [])] or final_tags
    round_records = record.get("ags_seq_rounds", [])
    realized_rounds = int(record.get("realized_rounds", 1))

    # R@50 after each round, last value carried forward to B.
    curve = [recall_at(round1_tags, gold_tags, 50)]
    for round_record in round_records:
        curve.append(recall_at([normalize_tag(tag) for tag in round_record.get("candidate_list", [])], gold_tags, 50))
    while len(curve) < max_rounds:
        curve.append(curve[-1])
    curve = curve[:max_rounds]

    final_rank = first_gold_rank(final_tags, gold_tags)
    round1_rank = first_gold_rank(round1_tags, gold_tags)
    return {
        "fact_id": record.get("example_idx"),
        "context_id": record.get("context_id"),
        "modality": record.get("input_type"),
        "arm": arm,
        "round1_candidates": round1_tags,
        "final_candidates": final_tags,
        "round1_rank_gold": round1_rank,
        "final_rank_gold": final_rank,
        "realized_rounds": realized_rounds,
        "stop_reason": record.get("stop_reason", "single_round"),
        "round1_recall_at_50": curve[0],
        "full_recall_at_50": recall_at(final_tags, gold_tags, 50),
        "recall_at_10": recall_at(final_tags, gold_tags, 10),
        "recall_at_50": recall_at(final_tags, gold_tags, 50),
        "recall_at_200": recall_at(final_tags, gold_tags, 200),
        "mrr": 1.0 / final_rank if final_rank else 0.0,
        "coverage": 1.0 if final_rank is not None else 0.0,
        "top1_accuracy": 1.0 if final_rank == 1 else 0.0,
        "aulc": round(float(np.mean(curve)), 8),
        "r50_curve": curve,
        "round1_parity_ok": bool((record.get("ags_seq_round1_parity") or {}).get("round1_parity_ok", True)),
    }


def round_rows(record: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    rows = []
    for round_record in record.get("ags_seq_rounds", []):
        row = {
            "fact_id": record.get("example_idx"),
            "context_id": record.get("context_id"),
            "modality": record.get("input_type"),
            "arm": arm,
        }
        row.update(round_record)
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], arm: str, modality: str, max_rounds: int) -> dict[str, Any]:
    if not rows:
        return {}
    def mean(field: str) -> float:
        return round(float(np.mean([row[field] for row in rows])), 6)

    round1_r50 = mean("round1_recall_at_50")
    full_r50 = mean("full_recall_at_50")
    return {
        "arm": arm,
        "modality": modality,
        "facts": len(rows),
        "contexts": len({row["context_id"] for row in rows}),
        "recall_at_10": mean("recall_at_10"),
        "recall_at_50": mean("recall_at_50"),
        "recall_at_200": mean("recall_at_200"),
        "mrr": mean("mrr"),
        "coverage": mean("coverage"),
        "top1_accuracy": mean("top1_accuracy"),
        "round1_recall_at_50": round1_r50,
        "full_recall_at_50": full_r50,
        "full_minus_round1_recall_at_50": round(full_r50 - round1_r50, 6),
        "realized_rounds_mean": mean("realized_rounds"),
        "realized_rounds_max": max(row["realized_rounds"] for row in rows),
        "aulc": mean("aulc"),
        "max_rounds": max_rounds,
        "round1_parity_failures": sum(1 for row in rows if not row["round1_parity_ok"]),
        "stop_reasons": json.dumps(
            {
                reason: sum(1 for row in rows if row["stop_reason"] == reason)
                for reason in sorted({row["stop_reason"] for row in rows})
            },
            sort_keys=True,
        ),
    }


def paired_bootstrap(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    field: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Context-level paired bootstrap of mean(left) - mean(right) on shared facts."""
    shared = sorted(set(left) & set(right))
    if not shared:
        return {}
    by_context: dict[Any, list[float]] = {}
    for fact_id in shared:
        by_context.setdefault(left[fact_id]["context_id"], []).append(
            float(left[fact_id][field]) - float(right[fact_id][field])
        )
    contexts = list(by_context)
    context_means = np.asarray([float(np.mean(by_context[context])) for context in contexts])
    context_sizes = np.asarray([len(by_context[context]) for context in contexts], dtype=float)
    observed = float(np.sum(context_means * context_sizes) / np.sum(context_sizes))

    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        picks = rng.integers(0, len(contexts), size=len(contexts))
        weights = context_sizes[picks]
        draws[index] = float(np.sum(context_means[picks] * weights) / np.sum(weights))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "metric": field,
        "facts": len(shared),
        "contexts": len(contexts),
        "iterations": iterations,
        "left_mean": round(float(np.mean([float(left[fact_id][field]) for fact_id in shared])), 6),
        "right_mean": round(float(np.mean([float(right[fact_id][field]) for fact_id in shared])), 6),
        "mean_difference": round(observed, 6),
        "ci_low": round(float(low), 6),
        "ci_high": round(float(high), 6),
        "ci_excludes_zero": bool(low > 0 or high < 0),
    }


def within_arm_bootstrap(
    rows: dict[int, dict[str, Any]],
    left_field: str,
    right_field: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Same bootstrap for two fields of one arm (full episode vs round one)."""
    shifted = {
        fact_id: {"context_id": row["context_id"], "value": row[right_field]}
        for fact_id, row in rows.items()
    }
    left = {fact_id: {"context_id": row["context_id"], "value": row[left_field]} for fact_id, row in rows.items()}
    result = paired_bootstrap(left, shifted, "value", iterations, seed)
    if result:
        result["metric"] = f"{left_field}_minus_{right_field}"
    return result


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
    arm_dirs = {
        "AGS": args.ags_dir or args.runs_root / "qwen3_32b_frozen_ags",
        "AGS+Seq": args.seq_dir or args.runs_root / "qwen3_32b_ags_seq",
        "AGS+Seq-random": args.seq_random_dir or args.runs_root / "qwen3_32b_ags_seq_random",
    }
    missing = {arm: path for arm, path in arm_dirs.items() if not (path / "bm25_candidates.jsonl").exists()}
    if missing:
        raise SystemExit(
            "Missing candidate traces for: "
            + ", ".join(f"{arm} ({path / 'bm25_candidates.jsonl'})" for arm, path in missing.items())
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_fact_path = args.output_dir / "per_fact.jsonl"
    rounds_path = args.output_dir / "rounds.jsonl"

    by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    with per_fact_path.open("w", encoding="utf-8") as fact_handle, rounds_path.open(
        "w", encoding="utf-8"
    ) as round_handle:
        for arm, path in arm_dirs.items():
            rows: dict[int, dict[str, Any]] = {}
            for record in stream_jsonl(path / "bm25_candidates.jsonl"):
                row = fact_row(record, arm, args.max_rounds)
                rows[int(row["fact_id"])] = row
                fact_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                for round_row in round_rows(record, arm):
                    round_handle.write(json.dumps(round_row, ensure_ascii=False) + "\n")
            by_arm[arm] = rows
            print(f"{arm}: {len(rows)} facts from {path}", flush=True)

    # Instance order and coverage must match, or the arms are not paired.
    fact_sets = {arm: set(rows) for arm, rows in by_arm.items()}
    shared_facts = set.intersection(*fact_sets.values())
    order_mismatch = {arm: sorted(facts - shared_facts)[:5] for arm, facts in fact_sets.items() if facts - shared_facts}

    summary_rows: list[dict[str, Any]] = []
    for arm, rows in by_arm.items():
        values = list(rows.values())
        summary_rows.append(summarize(values, arm, "all", args.max_rounds))
        for modality in ("table", "text"):
            subset = [row for row in values if row["modality"] == modality]
            if subset:
                summary_rows.append(summarize(subset, arm, modality, args.max_rounds))
    write_csv(args.output_dir / "arm_summary.csv", summary_rows)

    def subset(arm: str, modality: str) -> dict[int, dict[str, Any]]:
        return {
            fact_id: row
            for fact_id, row in by_arm[arm].items()
            if fact_id in shared_facts and (modality == "all" or row["modality"] == modality)
        }

    contrasts: list[dict[str, Any]] = []
    for modality in ("all", "table", "text"):
        # 1. full episode - round one, at R@50, inside each sequential arm.
        for arm in ("AGS+Seq", "AGS+Seq-random"):
            result = within_arm_bootstrap(
                subset(arm, modality),
                "full_recall_at_50",
                "round1_recall_at_50",
                args.bootstrap_samples,
                args.bootstrap_seed,
            )
            if result:
                contrasts.append(
                    {
                        "question": "does the full episode beat its own round one?",
                        "contrast": f"{arm}: full - round1",
                        "modality": modality,
                        **result,
                    }
                )
        # 2. Thompson vs uniform selection, on AULC.
        result = paired_bootstrap(
            subset("AGS+Seq", modality),
            subset("AGS+Seq-random", modality),
            "aulc",
            args.bootstrap_samples,
            args.bootstrap_seed,
        )
        if result:
            contrasts.append(
                {
                    "question": "does consulting the posteriors beat uniform selection?",
                    "contrast": "AGS+Seq - AGS+Seq-random",
                    "modality": modality,
                    **result,
                }
            )
        # 3. both sequential arms against AGS.
        for arm in ("AGS+Seq", "AGS+Seq-random"):
            for field in ("recall_at_50", "mrr"):
                result = paired_bootstrap(
                    subset(arm, modality), subset("AGS", modality), field, args.bootstrap_samples, args.bootstrap_seed
                )
                if result:
                    contrasts.append(
                        {
                            "question": "does the sequential arm exceed AGS?",
                            "contrast": f"{arm} - AGS",
                            "modality": modality,
                            **result,
                        }
                    )
    write_csv(args.output_dir / "contrasts.csv", contrasts)

    parity_failures = {
        arm: sum(1 for row in rows.values() if not row["round1_parity_ok"]) for arm, rows in by_arm.items()
    }
    metrics = {
        "arm_dirs": {arm: str(path) for arm, path in arm_dirs.items()},
        "facts_per_arm": {arm: len(rows) for arm, rows in by_arm.items()},
        "shared_facts": len(shared_facts),
        "instance_order_mismatch": order_mismatch,
        "round1_parity_failures": parity_failures,
        "bootstrap": {
            "iterations": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "unit": "source_context",
            "paired": True,
        },
        "arm_summary": summary_rows,
        "contrasts": contrasts,
        "artifact_paths": {
            "per_fact": str(per_fact_path),
            "rounds": str(rounds_path),
            "arm_summary": str(args.output_dir / "arm_summary.csv"),
            "contrasts": str(args.output_dir / "contrasts.csv"),
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    headline = [
        row
        for row in contrasts
        if row["modality"] == "all"
        and (
            row["contrast"].endswith("full - round1")
            or row["contrast"] == "AGS+Seq - AGS+Seq-random"
            or (row["contrast"].endswith("- AGS") and row["metric"] == "recall_at_50")
        )
    ]
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "shared_facts": len(shared_facts),
                "round1_parity_failures": parity_failures,
                "headline_contrasts": headline,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
