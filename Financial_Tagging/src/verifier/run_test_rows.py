#!/usr/bin/env python3
"""Test-split rows for Table 5 (ags_table5_ablation_spec.md sections 3-4, 6).

Computes every OFFLINE row (3.1 AGS full, 3.2 -ensemble, 3.4 -label-form, 3.5
-definition-form, 3.6 -summed fusion, 3.7 -score normalization, 3.8 -consensus rerank,
3.11 oracle) plus row 3.3 (-factorized hypothesis), which is not computed by this module at
all -- it is imported verbatim from the existing `parallel_sampling` test-split artifact and
only asserted equal to a fresh aggregation of that same artifact (section 3.3: "reuse that
run rather than re-running it, and assert numerical equality... if they differ, one of the
two used a different aggregator and the tables are inconsistent").

Row 3.9 (LLM verifier) and 3.10 (index ablation) are NOT produced here -- 3.9 needs a
verdicts file from run_llm_verifier.py (a GPU script) and is folded in only if that file
exists; 3.10 is its own separate panel (run_index_ablation.py, index_ablation.csv).

Every row is paired per fact against `AGS (full)`, bootstrap CIs resampled at the
source-context level, 2000 iterations, table and text reported separately (plus pooled).
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
from pathlib import Path
from typing import Any

from tqdm import tqdm

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map  # noqa: E402
from verifier.core import AblationConfig, aggregate, evaluate, metric_row, reset_consensus_cache  # noqa: E402
from verifier.data_prep import DEFAULT_TEST_TRACE, load_test_facts, stream_jsonl  # noqa: E402
from compute_ags_seq_arm_metrics import paired_bootstrap  # noqa: E402
from run_fintagging_grounding_baseline import normalize_tag  # noqa: E402


DEFAULT_PARALLEL_SAMPLING_TRACE = (
    _PARENT / "runs_fintagging_grounding_baseline" / "qwen3_32b_parallel_sampling" / "bm25_candidates.jsonl"
)
DEFAULT_PARALLEL_SAMPLING_METRICS = (
    _PARENT / "runs_fintagging_grounding_baseline" / "qwen3_32b_parallel_sampling" / "metrics.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-trace", type=Path, default=DEFAULT_TEST_TRACE)
    parser.add_argument("--parallel-sampling-trace", type=Path, default=DEFAULT_PARALLEL_SAMPLING_TRACE)
    parser.add_argument("--parallel-sampling-metrics", type=Path, default=DEFAULT_PARALLEL_SAMPLING_METRICS)
    parser.add_argument(
        "--selected-betas",
        type=Path,
        default=_PARENT / "runs_ags_table5_ablation" / "qwen3_32b" / "selected_betas.json",
    )
    parser.add_argument("--llm-verifier-verdicts", type=Path, default=None, help="Optional row 3.9 verdicts file.")
    parser.add_argument("--output-dir", type=Path, default=_PARENT / "runs_ags_table5_ablation" / "qwen3_32b")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _as_bootstrap_input(rows: list[dict[str, Any]], field: str) -> dict[int, dict[str, Any]]:
    return {int(row["fact_id"]): {"context_id": row["context_id"], field: float(bool(row[field]))} for row in rows}


def bootstrap_all_metrics(
    variant_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    modality: str,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    out = []
    metrics = ("recall_at_10", "recall_at_50", "recall_at_200", "mrr", "top1_accuracy")
    for metric in metrics:
        left = _as_bootstrap_input(variant_rows, metric) if metric != "mrr" else {
            int(row["fact_id"]): {"context_id": row["context_id"], "mrr": float(row["mrr"])} for row in variant_rows
        }
        right = _as_bootstrap_input(full_rows, metric) if metric != "mrr" else {
            int(row["fact_id"]): {"context_id": row["context_id"], "mrr": float(row["mrr"])} for row in full_rows
        }
        field = metric
        result = paired_bootstrap(left, right, field, iterations, seed)
        if result:
            result["modality"] = modality
            out.append(result)
    return out


def rows_for_modality(rows: list[dict[str, Any]], modality: str) -> list[dict[str, Any]]:
    if modality == "pooled":
        return rows
    return [row for row in rows if row["modality"] == modality]


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


def load_llm_verifier_verdicts(path: Path | None) -> dict[tuple[int, int, str], dict[str, Any]] | None:
    if path is None or not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    verdicts = {}
    for item in raw:
        key = (int(item["fact_id"]), int(item["hypothesis_idx"]), normalize_tag(item["tag"]))
        verdicts[key] = item["verdicts"]
    return verdicts


def compute_freetext_row(
    trace_path: Path,
    existing_metrics_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Row 3.3: recompute bm25-level (pre-rerank) metrics directly from parallel_sampling's
    own final_candidates, then assert equality against the pipeline's own aggregator (the
    `bm25_retrieval` block already stored in its metrics.json). Any mismatch means one of the
    two used a different aggregation code path and the tables are inconsistent (section 3.3).
    """
    per_fact_rows: list[dict[str, Any]] = []
    for record in stream_jsonl(trace_path):
        tags = [candidate["tag"] for candidate in record.get("final_candidates", record.get("candidates", []))]
        row = metric_row(tags, record.get("gold_tags", []))
        row.update(
            {
                "fact_id": record["example_idx"],
                "context_id": record.get("context_id"),
                "modality": record.get("input_type"),
            }
        )
        per_fact_rows.append(row)

    my_pooled = aggregate(per_fact_rows)
    my_table = aggregate(rows_for_modality(per_fact_rows, "table"))
    my_text = aggregate(rows_for_modality(per_fact_rows, "text"))

    pipeline_metrics = json.loads(existing_metrics_path.read_text(encoding="utf-8"))
    pipeline_pooled = pipeline_metrics.get("bm25_retrieval", {})
    pipeline_table = pipeline_metrics.get("by_input_type", {}).get("table", {}).get("bm25_retrieval", {})
    pipeline_text = pipeline_metrics.get("by_input_type", {}).get("text", {}).get("bm25_retrieval", {})

    def close(a: dict[str, Any], b: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
        diffs = {}
        for key in keys:
            if key not in a or key not in b:
                continue
            left, right = float(a[key]), float(b[key])
            if abs(left - right) > 1e-6:
                diffs[key] = {"mine": left, "pipeline": right}
        return diffs

    keys = ("recall_at_10", "recall_at_50", "recall_at_200", "mrr", "accuracy")
    # my aggregate names top1_accuracy where the pipeline's own aggregator names it accuracy.
    pipeline_pooled_renamed = dict(pipeline_pooled, top1_accuracy=pipeline_pooled.get("accuracy"))
    pipeline_table_renamed = dict(pipeline_table, top1_accuracy=pipeline_table.get("accuracy"))
    pipeline_text_renamed = dict(pipeline_text, top1_accuracy=pipeline_text.get("accuracy"))
    my_keys = ("recall_at_10", "recall_at_50", "recall_at_200", "mrr", "top1_accuracy")

    assertion = {
        "pooled_diffs": close(my_pooled, pipeline_pooled_renamed, my_keys),
        "table_diffs": close(my_table, pipeline_table_renamed, my_keys),
        "text_diffs": close(my_text, pipeline_text_renamed, my_keys),
        "note": (
            "row 3.3 is imported verbatim from parallel_sampling's own test-split artifact "
            "(retrieval_rounds config as actually run -- see the run's own metrics.json -- "
            "not necessarily J=2; this must be reconciled with whatever J the paper's Table 2 "
            "caption states for 'parallel sampling, stochastic')."
        ),
    }
    assertion["passed"] = not (assertion["pooled_diffs"] or assertion["table_diffs"] or assertion["text_diffs"])
    return per_fact_rows, assertion


def main() -> None:
    args = parse_args()
    normalization_map = load_normalization_map(DEFAULT_NORMALIZATION_MAP)
    selected = json.loads(args.selected_betas.read_text(encoding="utf-8"))["selected"]

    print("Loading AGS test-split trace ...", flush=True)
    test_facts = load_test_facts(args.test_trace, limit=args.limit)
    facts_list = list(test_facts.values())
    print(f"Test facts loaded: {len(facts_list)}", flush=True)

    llm_verdicts = load_llm_verifier_verdicts(args.llm_verifier_verdicts)

    lab_null_facts = sum(
        1 for fact in facts_list if all((idx, "lab") not in fact.rankings for idx in fact.hypotheses)
    )

    def run_variant(name: str, config: AblationConfig) -> list[dict[str, Any]]:
        reset_consensus_cache()
        return [
            evaluate(fact, config, normalization_map)
            for fact in tqdm(facts_list, desc=name, unit="fact")
        ]

    variants: dict[str, list[dict[str, Any]]] = {}
    configs: dict[str, AblationConfig] = {}

    configs["AGS (full)"] = AblationConfig(name="AGS (full)", beta=0.6)
    variants["AGS (full)"] = run_variant("AGS (full)", configs["AGS (full)"])
    full_rows = variants["AGS (full)"]

    ensemble_beta = selected["ensemble"]["selected_beta"]
    ensemble_idx_rows: dict[int, list[dict[str, Any]]] = {}
    for idx in (0, 1):
        cfg = AblationConfig(name=f"-ensemble (J=1, idx={idx})", n_hypotheses=1, kept_hypothesis_idx=idx, beta=ensemble_beta)
        configs[f"-ensemble idx{idx}"] = cfg
        ensemble_idx_rows[idx] = run_variant(f"ensemble_idx{idx}", cfg)
    # section 3.2's mitigation: the reported row is the mean over which hypothesis is kept.
    ensemble_mean_rows = []
    for row0, row1 in zip(ensemble_idx_rows[0], ensemble_idx_rows[1]):
        merged = {"fact_id": row0["fact_id"], "context_id": row0["context_id"], "modality": row0["modality"]}
        for metric in ("recall_at_10", "recall_at_50", "recall_at_200", "mrr", "top1_accuracy"):
            merged[metric] = (float(row0[metric]) + float(row1[metric])) / 2.0
        ensemble_mean_rows.append(merged)
    variants["-ensemble (mean of idx0/idx1)"] = ensemble_mean_rows
    configs["-ensemble (mean of idx0/idx1)"] = AblationConfig(
        name="-ensemble (mean of idx0/idx1)", n_hypotheses=1, kept_hypothesis_idx=0, beta=ensemble_beta
    )

    configs["-label-form (def only)"] = AblationConfig(
        name="-label-form (def only)", renderings=("def",), beta=selected["label_form"]["selected_beta"]
    )
    variants["-label-form (def only)"] = run_variant("label_form", configs["-label-form (def only)"])

    configs["-definition-form (lab only, zero-recall)"] = AblationConfig(
        name="-definition-form (lab only, zero-recall)",
        renderings=("lab",),
        lab_only_fallback=None,
        beta=selected["definition_form"]["selected_beta"],
    )
    variants["-definition-form (lab only, zero-recall)"] = run_variant(
        "definition_form_zero", configs["-definition-form (lab only, zero-recall)"]
    )
    configs["-definition-form (lab only, def fallback)"] = AblationConfig(
        name="-definition-form (lab only, def fallback)",
        renderings=("lab",),
        lab_only_fallback="def",
        beta=selected["definition_form"]["selected_beta"],
    )
    variants["-definition-form (lab only, def fallback)"] = run_variant(
        "definition_form_fallback", configs["-definition-form (lab only, def fallback)"]
    )

    configs["-summed fusion (mean RRF)"] = AblationConfig(name="-summed fusion (mean RRF)", fusion="mean", beta=0.6)
    variants["-summed fusion (mean RRF)"] = run_variant("mean_fusion", configs["-summed fusion (mean RRF)"])

    configs["-score normalization (raw, beta=0.6)"] = AblationConfig(
        name="-score normalization (raw, beta=0.6)", scaling="raw", beta=0.6
    )
    variants["-score normalization (raw, beta=0.6)"] = run_variant(
        "raw_at_06", configs["-score normalization (raw, beta=0.6)"]
    )
    configs["-score normalization (raw, own optimum)"] = AblationConfig(
        name="-score normalization (raw, own optimum)", scaling="raw", beta=selected["raw"]["selected_beta"]
    )
    variants["-score normalization (raw, own optimum)"] = run_variant(
        "raw_at_own", configs["-score normalization (raw, own optimum)"]
    )

    configs["-consensus rerank (beta=0)"] = AblationConfig(name="-consensus rerank (beta=0)", beta=0.0)
    variants["-consensus rerank (beta=0)"] = run_variant("consensus_off", configs["-consensus rerank (beta=0)"])
    # sanity required by 3.8: raw and range must be numerically identical when beta=0.
    #
    # Checked on the metrics, not on exact tag order. range_normalize is affine with positive
    # slope, so it cannot reorder candidates whose scores genuinely differ -- but its division
    # DOES erase differences at the far end of float precision. Measured over the full test
    # split: 3 of 2,509 facts have two candidates whose summed-RRF scores differ by ~1.7e-18
    # (float accumulation order over 1/(kappa+rank)); raw keeps them ordered by that
    # difference while range sees an exact tie and falls back to the alphabetical tie-break,
    # so the pair swaps. Zero facts change any metric, the candidate SET is identical, and the
    # gold rank is unchanged in all three.
    #
    # Asserting exact list equality therefore reports a permanent, meaningless `false`, which
    # is worse than no assertion at all: it trains the reader to ignore the assertions block.
    # The tag-order discrepancy is still counted and reported as a diagnostic.
    reset_consensus_cache()
    consensus_off_raw = [
        evaluate(fact, AblationConfig(name="beta0_raw_check", beta=0.0, scaling="raw"), normalization_map)
        for fact in tqdm(facts_list, desc="beta0_raw_check", unit="fact")
    ]
    _beta0_metric_keys = ("rank", "mrr", "top1_accuracy", "recall_at_10", "recall_at_50", "recall_at_200")
    _beta0_pairs = list(zip(variants["-consensus rerank (beta=0)"], consensus_off_raw))
    beta0_metrics_equal = all(
        all(a[key] == b[key] for key in _beta0_metric_keys) for a, b in _beta0_pairs
    )
    beta0_same_candidate_set = all(set(a["candidate_tags"]) == set(b["candidate_tags"]) for a, b in _beta0_pairs)
    beta0_tag_order_diffs = sum(1 for a, b in _beta0_pairs if a["candidate_tags"] != b["candidate_tags"])
    beta0_raw_eq_range = {
        "passed": bool(beta0_metrics_equal and beta0_same_candidate_set),
        "metrics_identical": bool(beta0_metrics_equal),
        "candidate_set_identical": bool(beta0_same_candidate_set),
        "facts_with_tag_order_difference": beta0_tag_order_diffs,
        "n_facts": len(_beta0_pairs),
        "note": (
            "Compared on metrics and candidate set. A nonzero tag-order difference is expected "
            "and benign: range-normalization's division collapses score gaps near float "
            "epsilon (~1e-18) into exact ties, which then break alphabetically instead of by "
            "score. It is only a problem if metrics_identical or candidate_set_identical is "
            "false -- that would mean the scaling actually reordered the pool."
        ),
    }

    if llm_verdicts is not None:
        configs["+ LLM verification layer"] = AblationConfig(
            name="+ LLM verification layer", beta=0.6, llm_verifier_verdicts=llm_verdicts
        )
        variants["+ LLM verification layer"] = run_variant("llm_verifier", configs["+ LLM verification layer"])

    oracle_beta = selected["oracle_best_single"]["selected_beta"]
    configs["Oracle best single hypothesis"] = AblationConfig(
        name="Oracle best single hypothesis", oracle_best_single=True, beta=oracle_beta
    )
    variants["Oracle best single hypothesis"] = run_variant("oracle", configs["Oracle best single hypothesis"])

    freetext_rows, freetext_assertion = compute_freetext_row(
        args.parallel_sampling_trace, args.parallel_sampling_metrics
    )
    variants["-factorized hypothesis (freetext, == Table 2 parallel sampling)"] = freetext_rows

    # ---- assemble ablation.csv ----------------------------------------------------------
    ablation_rows: list[dict[str, Any]] = []
    for variant_name, rows in variants.items():
        for modality in ("pooled", "table", "text"):
            subset = rows_for_modality(rows, modality)
            if not subset:
                continue
            agg = aggregate(subset)
            full_subset = rows_for_modality(full_rows, modality)
            beta_used = getattr(configs.get(variant_name), "beta", None)
            n_rankings = subset[0].get("n_rankings_fused") if "n_rankings_fused" in subset[0] else None
            for metric, value in (
                ("recall_at_10", agg["recall_at_10"]),
                ("recall_at_50", agg["recall_at_50"]),
                ("recall_at_200", agg["recall_at_200"]),
                ("mrr", agg["mrr"]),
                ("top1_accuracy", agg["top1_accuracy"]),
            ):
                bootstrap_key = "mrr" if metric == "mrr" else metric
                left = {
                    int(row["fact_id"]): {
                        "context_id": row["context_id"],
                        bootstrap_key: float(row[bootstrap_key]) if metric != "mrr" else float(row["mrr"]),
                    }
                    for row in subset
                }
                right = {
                    int(row["fact_id"]): {
                        "context_id": row["context_id"],
                        bootstrap_key: float(row[bootstrap_key]) if metric != "mrr" else float(row["mrr"]),
                    }
                    for row in full_subset
                }
                ci = paired_bootstrap(left, right, bootstrap_key, args.bootstrap_samples, args.bootstrap_seed)
                ablation_rows.append(
                    {
                        "variant": variant_name,
                        "modality": modality,
                        "beta_used": beta_used,
                        "n_rankings_fused": n_rankings,
                        "metric": metric,
                        "value": value,
                        "delta_vs_full": ci.get("mean_difference"),
                        "ci_low": ci.get("ci_low"),
                        "ci_high": ci.get("ci_high"),
                        "ci_excludes_zero": ci.get("ci_excludes_zero"),
                        "n_facts": ci.get("facts", len(subset)),
                        "n_contexts": ci.get("contexts"),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ablation.csv", ablation_rows)

    lab_null_rate = round(lab_null_facts / len(facts_list), 6) if facts_list else None
    fallback_rate = round(
        sum(row["lab_fallback_used"] for row in variants["-definition-form (lab only, def fallback)"])
        / max(1, len(variants["-definition-form (lab only, def fallback)"])),
        6,
    )

    metrics = {
        "configs_as_executed": {name: config.as_dict() for name, config in configs.items()},
        "selected_betas": selected,
        "assertions": {
            "freetext_equals_parallel_sampling": freetext_assertion,
            "beta0_raw_identical_to_range": beta0_raw_eq_range,
        },
        "label_lab_null_rate": lab_null_rate,
        "definition_form_fallback_rate": fallback_rate,
        "definition_form_empty_query_policy": "3.5a zero-recall (primary row); 3.5b def-fallback reported as a second row",
        "llm_verifier_included": llm_verdicts is not None,
        "top1_accuracy_basis": (
            "rank==1 on each variant's own final ranking; no downstream LLM selector is "
            "invoked anywhere in this table (see core.metric_row docstring)."
        ),
        "n_test_facts": len(facts_list),
        "artifact_paths": {
            "ablation_csv": str(args.output_dir / "ablation.csv"),
            "test_trace": str(args.test_trace),
            "parallel_sampling_trace": str(args.parallel_sampling_trace),
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"assertions": metrics["assertions"], "lab_null_rate": lab_null_rate}, indent=2), flush=True)
    print(f"Wrote {args.output_dir / 'ablation.csv'} and metrics.json", flush=True)


if __name__ == "__main__":
    main()
