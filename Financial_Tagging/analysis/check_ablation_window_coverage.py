#!/usr/bin/env python3
"""Can each tab:ablation arm be rescored under the LLM-only verifier from verdicts already on disk?

WHY THIS EXISTS
    Table 3's FHS row is the `llm_drop` arm; rows 4-10 were produced with
    `verifier_mode="deterministic"`, so their deltas against FHS mix component removal with a
    verifier switch. Regenerating them under `llm_drop` needs one verdict-generation job per arm
    ONLY IF the arm's head candidates fall outside the window the deployed verifier judged.
    `verdicts_k10_fused` is keyed (fact_id, hypothesis_idx, tag) over the top-10 cluster
    representatives of FHS's own fused ranking; `core.llm_only_consensus_scores` gives any tag
    without a verdict the `unjudged_fill` value, so an arm whose head is already judged can be
    rescored on CPU and an arm whose head is not cannot.

WHAT IT REPORTS, per arm
    head_covered   fraction of the arm's own fused top-10 (per kept hypothesis) that has a verdict
    any_covered    fraction of facts where at least one head tag is judged
    A high head_covered means "reuse the verdicts"; a low one means "generate new ones on GPU".

    The fused ranking is taken BEFORE the rerank term, which is what defines the window, so the
    answer does not depend on which verifier the arm was originally scored with.

CPU only. Reads the logged trace and the verdicts file; regenerates nothing.
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

DEFAULT_VERDICTS = (
    FHS_ROOT / "runs" / "runs_ags_verifier_ablation" / "qwen3_32b" / "verdicts_k10_fused" / "llm_verifier_verdicts.json"
)

# (row label in the paper, kwargs that define the arm). Only the fields that change the fused
# ranking matter here -- beta and verifier_mode do not.
ARMS: tuple[tuple[str, dict], ...] = (
    ("FHS (full) [reference]", {}),
    ("- ensemble (J=1, idx0)", {"n_hypotheses": 1, "kept_hypothesis_idx": 0}),
    ("- ensemble (J=1, idx1)", {"n_hypotheses": 1, "kept_hypothesis_idx": 1}),
    ("- label-form (def only)", {"renderings": ("def",)}),
    ("- definition-form (lab only)", {"renderings": ("lab",)}),
    ("- summed fusion (mean RRF)", {"fusion": "mean"}),
    ("- score norm. (raw scores)", {"scaling": "raw"}),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-trace", type=Path, default=DEFAULT_TEST_TRACE)
    parser.add_argument("--verdicts", type=Path, default=DEFAULT_VERDICTS)
    parser.add_argument("--normalization-map", type=Path, default=DEFAULT_NORMALIZATION_MAP)
    parser.add_argument("--top-m", type=int, default=10, help="Window size the verdicts were generated at.")
    parser.add_argument("--head", type=int, default=10, help="How deep to call the arm's 'head'.")
    parser.add_argument("--cluster-scan-depth", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "ablation_window_coverage.json")
    return parser.parse_args()


def head_tags(fact, config, normalization_map, head, scan_depth) -> tuple[list[int], list[str]]:
    """The arm's own fused head, as tags, plus the hypothesis indices it keeps."""
    hyp_indices = _selected_hypothesis_indices(fact, config)
    rankings, lab_only_missing = _rankings_for(fact, hyp_indices, config.renderings)
    if lab_only_missing and config.lab_only_fallback == "def":
        rankings, _ = _rankings_for(fact, hyp_indices, ("def",))
    if not rankings:
        return hyp_indices, []
    scores, best_candidate = fuse(rankings, config.rrf_kappa, config.fusion)
    if config.truncate_pool_to_top_k:
        scores, best_candidate = truncate_fused_pool(scores, best_candidate, config.top_k)
    pool = [best_candidate[tag] for tag in sorted(scores, key=lambda t: (-scores[t], t))]
    # Same window rule the verifier used at generation time.
    window = cluster_representatives(pool, normalization_map, head, scan_depth)
    return hyp_indices, [normalize_tag(c.get("tag", "")) for c in window]


def main() -> None:
    args = parse_args()
    normalization_map = load_normalization_map(args.normalization_map)
    verdicts = load_llm_verifier_verdicts(args.verdicts)
    judged_keys = set(verdicts)
    print(f"verdicts: {len(judged_keys)} (fact, hyp, tag) keys from {args.verdicts.parent.name}", flush=True)

    facts = list(load_test_facts(args.test_trace).values())
    if args.limit:
        facts = facts[: args.limit]
    print(f"facts: {len(facts)}\n", flush=True)

    results = {}
    for label, kwargs in ARMS:
        config = AblationConfig(name=label, beta=0.6, verifier_mode="llm_drop",
                                truncate_pool_to_top_k=True, llm_unjudged_fill="mean",
                                llm_verifier_top_m=args.top_m, llm_verifier_verdicts=verdicts, **kwargs)
        needed = covered = 0
        facts_any = facts_none = 0
        for fact in facts:
            hyp_indices, tags = head_tags(fact, config, normalization_map, args.head, args.cluster_scan_depth)
            if not tags:
                continue
            hit = 0
            for idx in hyp_indices:
                for tag in tags:
                    needed += 1
                    if (fact.fact_id, idx, tag) in judged_keys:
                        covered += 1
                        hit += 1
            facts_any += int(hit > 0)
            facts_none += int(hit == 0)
        frac = covered / needed if needed else float("nan")
        any_frac = facts_any / (facts_any + facts_none) if (facts_any + facts_none) else float("nan")
        results[label] = {"head_covered": frac, "any_covered": any_frac,
                          "needed_keys": needed, "covered_keys": covered}
        verdict = "REUSE OK" if frac >= 0.90 else ("PARTIAL" if frac >= 0.5 else "NEEDS GPU")
        print(f"{label:30} head_covered={frac:6.3f}  facts_with_any={any_frac:6.3f}  -> {verdict}", flush=True)

    args.out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
