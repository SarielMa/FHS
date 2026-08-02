#!/usr/bin/env python3
"""Every reported cell must come from ONE code path. This checks that, from the runs themselves.

WHY THIS EXISTS
    The paper's numbers were assembled over weeks, and the failures that actually happened were
    never "the code is wrong" -- they were "this row came from a different configuration than the
    row next to it": a deterministic rerank term next to an LLM-only one, a verifier window cut
    from an already-reranked order, w_cov=0 baselines beside a w_cov=1 method, a scoring set of
    six dimensions where the verifier was asked about three. Each looked plausible in the table.

    So this script does not inspect code. It reads the configuration each run RECORDED FOR ITSELF
    (ranking_summary.json / metrics.json / llm_verifier_summary.json) and compares it against one
    pinned deployed configuration. A row may differ from the pin only in the fields that define
    that row -- `- summed fusion` may differ in `fusion` and nothing else. Anything further is
    reported as DRIFT, which is the thing that silently produces a wrong delta.

USAGE
    python3 verify_single_code_path.py                 # check every registered row
    python3 verify_single_code_path.py --json out.json # machine-readable, for CI or a commit hook

Exit code is non-zero if any row drifts, so it can gate a table edit.
CPU only, reads metadata only, seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent.parent
ABL = SCRIPT_DIR / "runs" / "runs_ags_verifier_ablation" / "qwen3_32b"
BASE = SCRIPT_DIR / "runs" / "runs_fintagging_grounding_baseline"

# The one configuration. Anything not listed here is not part of the contract.
PIN: dict[str, Any] = {
    "verifier_mode": "llm_drop",       # the deployed candidate-level verifier, LLM only
    "beta": 0.6,                       # rerank weight, selected on the development sample
    "top_m": 10,                       # K_v, the verifier window
    "top_k": 200,                      # retrieval depth
    "llm_unjudged_fill": "mean",       # a candidate no window reached is neutral, not zero
    "fusion": "sum",                   # summed RRF, the multiplicity bonus
    "scaling": "range",                # range-normalized before beta is applied
    "renderings": ("def", "lab"),      # dual rendering on tabular evidence
    "n_hypotheses": 2,                 # J
    "label_coverage_weight": 1.0,      # shared index property
    "window_source": "fused",          # window cut before either verifier touches the order
    # Deployed 2026-07-30: ask and score every generated dimension. See core.py for why.
    "judged_dimensions": ("FAMILY", "ROLE", "EVENT", "QUALIFIER", "SCOPE", "TEMPORAL"),
    "llm_verifier_dimensions": ("FAMILY", "ROLE", "EVENT", "QUALIFIER", "SCOPE", "TEMPORAL"),
    # The shared generation budget. It was method-dependent until 2026-07-30, which starved four
    # baselines; a run that records a different cap is not on the same path as the others.
    "token_cap": 2048,
    "calls_at_token_cap": 0,
}

# paper row -> (directory, the fields this row is DEFINED by differing in)
ROWS: dict[str, tuple[Path, dict[str, Any]]] = {
    # tab:ablation, LLM-only family
    "FHS (full)":                (ABL / "rerank_no_determ_k10fused", {}),
    "- verifier":                (ABL / "rerank_no_verifier", {"beta": 0.0, "verifier_mode": "deterministic"}),
    "Program-driven score":      (ABL / "rerank_no_llm", {"verifier_mode": "deterministic"}),
    "- score norm. (LLM-only)":  (ABL / "rerank_llmonly_raw_scaling", {"scaling": "raw"}),
    # tab:ablation rows still on the deterministic term; these SHOULD report drift until their
    # llmonly_* replacements land. Listed so the drift is visible rather than forgotten.
    "- ensemble idx0":           (ABL / "rerank_ensemble_idx0", {"n_hypotheses": 1}),
    "- ensemble idx1":           (ABL / "rerank_ensemble_idx1", {"n_hypotheses": 1}),
    "- label-form":              (ABL / "rerank_label_form", {"renderings": ("def",), "beta": 0.8}),
    "- definition-form":         (ABL / "rerank_definition_form", {"renderings": ("lab",), "beta": 0.2}),
    "- summed fusion":           (ABL / "rerank_mean_fusion", {"fusion": "mean"}),
    "- score norm.":             (ABL / "rerank_raw_scaling", {"scaling": "raw"}),
    "- label coverage":          (ABL / "rerank_wcov0", {"label_coverage_weight": 0.0}),
    "Oracle best single":        (ABL / "rerank_oracle_single", {"n_hypotheses": 1}),
    # verdict generations that any row consumes
    "verdicts K_v=5":            (ABL / "verdicts_k5_fused", {"top_m": 5}),
    "verdicts K_v=10":           (ABL / "verdicts_k10_fused", {}),
    "verdicts K_v=20":           (ABL / "verdicts_k20_fused", {"top_m": 20}),
}

# Baseline rows carry a different contract: they have no verifier at all, and the coverage term is
# the field under repair. Checked separately so a missing verifier is not reported as drift.
BASELINE_ROWS: dict[str, Path] = {
    "Direct retrieval":          BASE / "qwen3_32b_direct_retrieval",
    "Direct retrieval @w_cov=1": BASE / "qwen3_32b_direct_retrieval_wcov1",
    "One-pass free-text":        BASE / "qwen3_32b_one_pass_grounding",
    "One-pass free-text @w_cov=1": BASE / "qwen3_32b_one_pass_grounding_wcov1",
    "One-pass structured":       BASE / "qwen3_32b_one_pass_structured",
    "Parallel stochastic":       BASE / "qwen3_32b_parallel_sampling",
    "Parallel diversity":        BASE / "qwen3_32b_parallel_sampling_diversity",
    "Decomposed":                BASE / "qwen3_32b_decomposed_retrieval",
    "Intrinsic refine.":         BASE / "qwen3_32b_intrinsic_self_refinement",
    "Feedback refine.":          BASE / "qwen3_32b_retrieval_feedback_refinement",
}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def declared_config(directory: Path) -> tuple[dict[str, Any], list[str]]:
    """What this run says about itself, and which files it said it in."""
    got: dict[str, Any] = {}
    sources: list[str] = []
    for name in ("ranking_summary.json", "llm_verifier_summary.json", "metrics.json"):
        payload = read_json(directory / name)
        if payload is None:
            continue
        sources.append(name)
        for key in ("verifier_mode", "beta", "top_m", "top_k", "llm_unjudged_fill",
                    "fusion", "scaling", "label_coverage_weight", "window_source",
                    "llm_verifier_dimensions"):
            if payload.get(key) is not None and key not in got:
                got[key] = payload[key]
        if payload.get("judge_dimensions"):
            got["judged_dimensions"] = tuple(payload["judge_dimensions"])
        elif "firing_counts" in payload:
            # Pre-dates the judge_dimensions field; the firing counters enumerate what was asked.
            got["judged_dimensions"] = tuple(sorted(payload["firing_counts"], key=lambda d: PIN["judged_dimensions"].index(d) if d in PIN["judged_dimensions"] else 99))
        if payload.get("query_mode"):
            got["query_mode"] = payload["query_mode"]
        trunc = payload.get("truncation")
        if isinstance(trunc, dict):
            if trunc.get("token_cap") is not None:
                got["token_cap"] = trunc["token_cap"]
            if trunc.get("calls_at_token_cap") is not None:
                got["calls_at_token_cap"] = trunc["calls_at_token_cap"]
    return got, sources


def compare(row: str, directory: Path, allowed: dict[str, Any]) -> dict[str, Any]:
    if not directory.exists():
        return {"row": row, "status": "MISSING", "detail": f"{directory.name} does not exist"}
    got, sources = declared_config(directory)
    if not sources:
        return {"row": row, "status": "NO METADATA", "detail": str(directory)}
    expected = dict(PIN)
    expected.update(allowed)
    drift = {}
    for key, want in expected.items():
        if key not in got:
            continue  # the run does not record it; absence is not evidence of drift
        have = got[key]
        if isinstance(want, tuple):
            have_t = tuple(have) if isinstance(have, (list, tuple)) else (have,)
            if tuple(sorted(have_t)) != tuple(sorted(want)):
                drift[key] = {"declared": list(have_t), "pinned": list(want)}
        elif isinstance(want, float):
            if abs(float(have) - want) > 1e-9:
                drift[key] = {"declared": have, "pinned": want}
        elif have != want:
            drift[key] = {"declared": have, "pinned": want}
    return {"row": row, "status": "DRIFT" if drift else "OK", "declared_in": sources,
            "defined_deviation": allowed, "drift": drift}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    results = [compare(row, d, allowed) for row, (d, allowed) in ROWS.items()]

    print("=== rows under the deployed contract ===")
    for r in results:
        mark = {"OK": "  ok  ", "DRIFT": " DRIFT", "MISSING": " ---- ", "NO METADATA": " ???? "}[r["status"]]
        extra = ""
        if r["status"] == "DRIFT":
            extra = "  " + "; ".join(f"{k}: {v['declared']} != {v['pinned']}" for k, v in r["drift"].items())
        elif r["status"] in ("MISSING", "NO METADATA"):
            extra = "  " + r.get("detail", "")
        print(f"{mark} {r['row']:28}{extra}")

    print("\n=== baseline rows (no verifier; coverage term is the field under repair) ===")
    base_results = []
    for row, d in BASELINE_ROWS.items():
        if not d.exists():
            print(f" ---- {row:30} {d.name} does not exist")
            base_results.append({"row": row, "status": "MISSING"})
            continue
        metrics = read_json(d / "metrics.json")
        if metrics is None:
            print(f" ???? {row:30} no metrics.json (rerank may still be running)")
            base_results.append({"row": row, "status": "NO METADATA"})
            continue
        wcov = metrics.get("label_coverage_weight")
        qm = metrics.get("query_mode")
        rounds = metrics.get("retrieval_rounds")
        kappa = metrics.get("rrf_kappa")
        note = f"query_mode={qm} rounds={rounds} kappa={kappa} w_cov={wcov if wcov is not None else '(not recorded)'}"
        print(f"  --   {row:30} {note}")
        base_results.append({"row": row, "status": "INFO", "note": note})

    drifted = [r for r in results if r["status"] == "DRIFT"]
    print(f"\n{len(results) - len(drifted)}/{len(results)} rows match the pinned configuration; "
          f"{len(drifted)} drift.")
    if drifted:
        print("A drifting row's delta against FHS mixes its own change with the drift. Either "
              "regenerate it under the pin or state the deviation in the caption.")
    if args.json:
        args.json.write_text(json.dumps({"pin": {k: list(v) if isinstance(v, tuple) else v for k, v in PIN.items()},
                                         "rows": results, "baselines": base_results}, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 1 if drifted else 0


if __name__ == "__main__":
    sys.exit(main())
