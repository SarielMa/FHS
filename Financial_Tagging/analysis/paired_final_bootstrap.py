#!/usr/bin/env python3
"""Paired context-level bootstrap on FINAL (post-listwise-rerank) accuracy.

The daggers in tab:ablation mark retrieval-stage contrasts only. The Final Acc. column has no
significance test at all, so "LLM-only beats hybrid end to end" is currently an unqualified
point difference. This supplies the test, using the same unit, seed and resample count as the
rest of the paper (source context, 20260724, 2000).

Paired: every arm reranks the same facts, so the same context resample is applied to both
arms rather than resampling them independently.
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
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from run_fintagging_grounding_baseline import normalize_tag

RUN = Path("runs_ags_verifier_ablation/qwen3_32b")
BASE = "hybrid_full"
ARMS = ["hybrid_full", "no_llm", "no_determ", "llm_only", "no_verifier"]
SEED, SAMPLES = 20260724, 2000

def per_context(arm):
    num, den = defaultdict(int), defaultdict(int)
    p = RUN / f"rerank_{arm}" / "qwen_rerank_predictions.jsonl"
    if not p.exists():
        return None
    for line in p.open(encoding="utf-8"):
        r = json.loads(line)
        c = int(r["context_id"]); den[c] += 1
        gold = {normalize_tag(t) for t in (r.get("gold_tags") or [])}
        s = r.get("selected_tag")
        if s and normalize_tag(s) in gold:
            num[c] += 1
    return num, den

data = {a: per_context(a) for a in ARMS}
data = {a: v for a, v in data.items() if v}
ctx = sorted(data[BASE][1])
rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(ctx), size=(SAMPLES, len(ctx)))
den = np.array([data[BASE][1][c] for c in ctx], dtype=float)

def acc_draws(arm):
    n = np.array([data[arm][0].get(c, 0) for c in ctx], dtype=float)
    return n[idx].sum(axis=1) / den[idx].sum(axis=1), n.sum() / den.sum()

base_draws, base_acc = acc_draws(BASE)
print(f"contexts={len(ctx)}  baseline={BASE} acc={base_acc:.4f}\n")
print(f"{'arm':<13} {'acc':>7} {'delta':>8} {'95% CI':>20} {'excl.0':>7}")
for arm in ARMS:
    if arm not in data:
        continue
    draws, acc = acc_draws(arm)
    d = draws - base_draws
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"{arm:<13} {acc:>7.4f} {acc-base_acc:>+8.4f}  [{lo:>+7.4f},{hi:>+7.4f}] {str(not (lo<=0<=hi)):>7}")
