#!/usr/bin/env python3
"""Per-arm verifier windows for tab:ablation, so each ablation row can be scored by the LLM-only
verifier over ITS OWN fused ranking instead of FHS's.

THE PROBLEM THIS SOLVES
    `run_llm_verifier.py` cuts its window from the trace's stored `frozen_ags_rrf_normalized` --
    the deployed J=2 dual-rendering summed-RRF score. Every arm therefore inherits FHS's window,
    which is why the rendering/ensemble/fusion rows of tab:ablation could only ever be scored with
    the deterministic term: their own heads are 76-84% covered by FHS's verdicts, not 100%
    (`check_ablation_window_coverage.py`). This script computes each arm's own top-M cluster
    representatives from `ags_table5_ablation.core`, and `run_llm_verifier.py --window-tags` reads
    them, so the verifier judges the candidates that arm actually ranks first.

WHAT IT GUARANTEES BEFORE ANY GPU TIME IS SPENT
    Every window tag is resolved against the trace record the verifier will read. A tag the record
    cannot supply has no label or definition to put in the prompt, so it would silently shrink the
    window; this script counts those and refuses to write a file if any arm loses more than
    --max-unresolved of its tags.

SELF-CHECK
    Run with --arm full: the window it emits must equal what the deployed verifier already judged,
    i.e. every (fact, hyp, tag) key must be present in verdicts_k10_fused. --verify-against does
    exactly that comparison and exits non-zero on a mismatch.

CPU only, about a minute per arm.

    ./stage_arm_windows.py --arm ensemble_idx0 --arm ensemble_idx1 --arm def_only \
        --arm lab_only --arm mean_fusion
    ./stage_arm_windows.py --arm full --verify-against \
        runs_ags_verifier_ablation/qwen3_32b/verdicts_k10_fused/llm_verifier_verdicts.json
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

SCRIPT_DIR = FHS_ROOT

from ags_sequential_arms import cluster_representatives  # noqa: E402
from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map, normalize_tag  # noqa: E402
from verifier.core import (  # noqa: E402
    AblationConfig,
    _rankings_for,
    _selected_hypothesis_indices,
    fuse,
    truncate_fused_pool,
)
from verifier.data_prep import DEFAULT_TEST_TRACE, load_test_facts  # noqa: E402
from verifier.run_test_rows import load_llm_verifier_verdicts  # noqa: E402

# Arm name -> the AblationConfig fields that change the fused ranking. beta, verifier_mode and
# scaling are absent on purpose: they do not reorder the fused score the window is cut from
# (range-normalization is monotone), so an arm that only changes those reuses the full window.
ARM_CONFIGS: dict[str, dict[str, Any]] = {
    "full": {},
    "ensemble_idx0": {"n_hypotheses": 1, "kept_hypothesis_idx": 0},
    "ensemble_idx1": {"n_hypotheses": 1, "kept_hypothesis_idx": 1},
    "def_only": {"renderings": ("def",)},
    "lab_only": {"renderings": ("lab",)},
    "mean_fusion": {"fusion": "mean"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", choices=sorted(ARM_CONFIGS), required=True)
    parser.add_argument("--test-trace", type=Path, default=DEFAULT_TEST_TRACE)
    parser.add_argument("--normalization-map", type=Path, default=DEFAULT_NORMALIZATION_MAP)
    parser.add_argument("--top-m", type=int, default=10)
    parser.add_argument("--cluster-scan-depth", type=int, default=60)
    parser.add_argument("--out-dir", type=Path, default=FHS_ROOT / "runs" / "runs_ags_verifier_ablation" / "qwen3_32b" / "arm_windows")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-unresolved",
        type=float,
        default=0.005,
        help="Refuse to write an arm whose window tags are missing from the trace more often "
        "than this. A missing tag cannot be put in the prompt, so it would shrink the window "
        "silently and understate the arm.",
    )
    parser.add_argument("--verify-against", type=Path, default=None,
                        help="A verdicts JSON. Every (fact, hyp, tag) this script emits must be a key in it.")
    return parser.parse_args()


def trace_tags(path: Path, limit: int | None) -> dict[int, set[str]]:
    """Which tags each record can supply a candidate object for."""
    available: dict[int, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for offset, line in enumerate(handle, start=1):
            if limit is not None and offset > limit:
                break
            record = json.loads(line)
            fact_id = int(record["example_idx"])
            # Candidate objects live in the per-round lists as well; a fused head computed from
            # those rounds can contain a tag that the record's own top-K final list dropped. Only
            # looking at final_candidates lost 4.9% of the w_cov=0 arm's window.
            pool = list(record.get("final_candidates") or record.get("candidates") or [])
            for round_record in record.get("rounds") or []:
                pool.extend(round_record.get("candidates") or [])
            available[fact_id] = {normalize_tag(c.get("tag", "")) for c in pool}
    return available


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    normalization_map = load_normalization_map(args.normalization_map)
    facts = list(load_test_facts(args.test_trace).values())
    if args.limit:
        facts = facts[: args.limit]
    available = trace_tags(args.test_trace, args.limit)
    verdicts = load_llm_verifier_verdicts(args.verify_against) if args.verify_against else None
    print(f"facts: {len(facts)}   top_m: {args.top_m}", flush=True)

    for arm in args.arm:
        config = AblationConfig(name=arm, beta=0.6, verifier_mode="llm_drop", truncate_pool_to_top_k=True,
                                llm_unjudged_fill="mean", llm_verifier_top_m=args.top_m, **ARM_CONFIGS[arm])
        rows: list[dict[str, Any]] = []
        emitted = unresolved = no_ranking = 0
        missing_verdicts = 0
        for fact in facts:
            hyp_indices = _selected_hypothesis_indices(fact, config)
            rankings, lab_only_missing = _rankings_for(fact, hyp_indices, config.renderings)
            if lab_only_missing and config.lab_only_fallback == "def":
                # Same fallback policy core.evaluate applies, so the window matches the row.
                rankings, _ = _rankings_for(fact, hyp_indices, ("def",))
            if not rankings:
                no_ranking += 1
                continue
            scores, best_candidate = fuse(rankings, config.rrf_kappa, config.fusion)
            if config.truncate_pool_to_top_k:
                scores, best_candidate = truncate_fused_pool(scores, best_candidate, config.top_k)
            ordered = [best_candidate[tag] for tag in sorted(scores, key=lambda t: (-scores[t], t))]
            window = cluster_representatives(ordered, normalization_map, args.top_m, args.cluster_scan_depth)
            tags = [normalize_tag(c.get("tag", "")) for c in window]
            have = available.get(fact.fact_id, set())
            keep = [t for t in tags if t in have]
            unresolved += len(tags) - len(keep)
            emitted += len(tags)
            if verdicts is not None:
                missing_verdicts += sum(
                    1 for idx in hyp_indices for t in keep if (fact.fact_id, idx, t) not in verdicts
                )
            rows.append({"fact_id": fact.fact_id, "hypothesis_indices": hyp_indices, "window_tags": keep})

        rate = unresolved / emitted if emitted else 0.0
        print(f"{arm:14} facts={len(rows):5} window_tags={emitted:6} unresolved={unresolved} ({rate:.4f})"
              f"  facts_without_ranking={no_ranking}", flush=True)
        if rate > args.max_unresolved:
            raise SystemExit(
                f"{arm}: {rate:.3%} of window tags are not in the trace (limit {args.max_unresolved:.3%}). "
                "The verifier cannot prompt with a candidate it has no text for; widen the trace's "
                "stored pool instead of shipping a shrunken window."
            )
        if verdicts is not None:
            print(f"{arm:14} keys missing from --verify-against: {missing_verdicts}", flush=True)
            if arm == "full" and missing_verdicts:
                raise SystemExit(
                    f"self-check FAILED: the 'full' arm window should already be judged, but "
                    f"{missing_verdicts} keys are absent. The window computation here does not "
                    "match the one the deployed verdicts were generated with."
                )
        out = args.out_dir / f"window_{arm}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(f"{arm:14} wrote {out}", flush=True)


if __name__ == "__main__":
    main()
