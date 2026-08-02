#!/usr/bin/env python3
"""Loaders for the Table 5 ablation (ags_table5_ablation_spec.md section 0).

Two independent sources, neither requiring new generation or new retrieval for the offline
rows:

  test split   runs_fintagging_grounding_baseline/qwen3_32b_frozen_ags/bm25_candidates.jsonl
               AGS's own test-split run. Its persisted `rounds` field already carries full
               per-hypothesis, per-rendering candidate lists with the full candidate object
               (documentation, retrieval_text) -- exactly what section 0 requires, verbatim.

  dev sample   runs_ags_specialized_generators/qwen3_32b/{hypotheses,retrievals}.jsonl
               Task A's `stoch:0` / `stoch:1` slots are frozen AGS's own two stochastic
               hypotheses (same J=2, K=200, w_cov=1.0 config), generated on the frozen
               661-fact / 70-context development sample. This is the AGS dev run the ablation
               needs for its beta re-sweeps -- it was never logged under that name because
               Task A had a different purpose, but the config is identical.

               One wrinkle: Task A's persisted retrievals.jsonl is compacted (see
               run_ags_coverage_pilot.compact_candidates) and drops `documentation` /
               `retrieval_text`, which agree()'s symbolic profile parsing reads. Trusting the
               compacted log would make the dev beta sweep score a weaker symbolic signal
               than generation time actually used. So this module replays retrieval from the
               logged query strings (BM25 is deterministic; same query, same taxonomy file,
               same retriever settings reproduces the original ranking exactly, full fields
               included) rather than reading retrievals.jsonl at all. It is retrieval-only,
               CPU, and reuses queries verbatim -- no new generation, matching the spirit of
               section 3.10's "reuse the logged queries" pattern.
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

import json
import sys
from pathlib import Path
from typing import Any, Iterator

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from verifier.core import FactRecord  # noqa: E402
from run_ags_coverage_pilot import row_to_example  # noqa: E402
from run_fintagging_grounding_baseline import (  # noqa: E402
    DEFAULT_TAXONOMY_JSONL,
    Concept,
    TaxonomyRetriever,
    load_taxonomy,
    normalize_tag,
    retrieval_query_from_grounding,
    retrieve_candidates,
)


SCRIPT_DIR = _PARENT
DEFAULT_TEST_TRACE = FHS_ROOT / "runs" / "runs_fintagging_grounding_baseline" / "qwen3_32b_frozen_ags" / "bm25_candidates.jsonl"
DEFAULT_DEV_SAMPLE_FACTS = FHS_ROOT / "data" / "dev" / "sample_facts.jsonl"
DEFAULT_DEV_HYPOTHESES = FHS_ROOT / "runs" / "runs_ags_specialized_generators" / "qwen3_32b" / "hypotheses.jsonl"
DEV_SLOT_TO_HYPOTHESIS_IDX = {"stoch:0": 0, "stoch:1": 1}

# Candidate fields evaluate()/agree() actually read. `references` (ASC citation list) is
# dropped -- candidate_text() never touches it -- to keep the ~2500-fact test trace's ~2M
# candidate rows in memory during a single ablation pass.
_CANDIDATE_FIELDS = (
    "rank",
    "tag",
    "type",
    "standard_label",
    "documentation",
    "retrieval_text",
    "bm25_score",
    "bm25_normalized_score",
    "label_coverage",
    "query_label_coverage",
    "retrieval_score",
)


def _compact(candidate: dict[str, Any]) -> dict[str, Any]:
    return {field: candidate.get(field) for field in _CANDIDATE_FIELDS if field in candidate}


def stream_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(stream_jsonl(path))


def load_test_facts(path: Path = DEFAULT_TEST_TRACE, limit: int | None = None) -> dict[int, FactRecord]:
    """AGS's test-split trace -> one FactRecord per fact, rankings keyed (hyp_idx, rendering).

    `limit` stops streaming early (for smoke tests) rather than loading the full ~3.6GB trace
    and truncating afterward.
    """
    facts: dict[int, FactRecord] = {}
    for row in stream_jsonl(path):
        if limit is not None and len(facts) >= limit:
            break
        fact_id = int(row["example_idx"])
        hypotheses = {
            int(hypothesis["hypothesis_idx"]): {
                "dimensions": hypothesis.get("dimensions", {}),
                "operators": hypothesis.get("operators", []),
                "retrieval_query": hypothesis.get("retrieval_query", ""),
            }
            for hypothesis in row.get("frozen_ags_hypotheses", [])
        }
        rankings: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for round_record in row.get("rounds", []):
            if round_record.get("label_render_skipped"):
                continue
            key = (int(round_record["hypothesis_idx"]), round_record["rendering"])
            rankings[key] = [_compact(candidate) for candidate in round_record.get("candidates", [])]
        facts[fact_id] = FactRecord(
            fact_id=fact_id,
            context_id=row.get("context_id"),
            modality=row.get("input_type", ""),
            datatype=row.get("type", ""),
            gold_tags=[normalize_tag(tag) for tag in row.get("gold_tags", [])],
            hypotheses=hypotheses,
            rankings=rankings,
        )
    return facts


def load_dev_facts(
    taxonomy: list[Concept] | None = None,
    sample_facts_path: Path = DEFAULT_DEV_SAMPLE_FACTS,
    hypotheses_path: Path = DEFAULT_DEV_HYPOTHESES,
    top_k: int = 200,
    log_every: int = 200,
) -> dict[int, FactRecord]:
    """Full-fidelity replay of frozen AGS's own dev-sample hypotheses (see module docstring).

    Deterministic: same taxonomy file, same retriever settings (w_cov=1.0, pool_multiplier=0,
    the frozen defaults), same query string that generation time used -> identical BM25
    ranking, this time with the full candidate object.
    """
    if taxonomy is None:
        taxonomy = load_taxonomy(DEFAULT_TAXONOMY_JSONL)
    retriever = TaxonomyRetriever(
        taxonomy, type_filter=True, label_coverage_weight=1.0, label_coverage_pool_multiplier=0
    )
    examples = {int(row["fact_id"]): row_to_example(row) for row in stream_jsonl(sample_facts_path)}

    facts: dict[int, FactRecord] = {}
    processed = 0
    for row in stream_jsonl(hypotheses_path):
        slot = row.get("slot")
        if slot not in DEV_SLOT_TO_HYPOTHESIS_IDX:
            continue
        fact_id = int(row["fact_id"])
        hyp_idx = DEV_SLOT_TO_HYPOTHESIS_IDX[slot]
        example = examples[fact_id]
        fact = facts.setdefault(
            fact_id,
            FactRecord(
                fact_id=fact_id,
                context_id=example.context_id,
                modality=example.input_type,
                datatype=example.entity_type,
                gold_tags=[normalize_tag(tag) for tag in example.gold_tags],
                hypotheses={},
                rankings={},
            ),
        )
        fact.hypotheses[hyp_idx] = {
            "dimensions": row.get("dimensions", {}),
            "operators": row.get("operators", []),
            "retrieval_query": row.get("retrieval_query", ""),
        }

        query_def = retrieval_query_from_grounding(example, row.get("query_def", "") or "")
        fact.rankings[(hyp_idx, "def")] = [
            _compact(candidate) for candidate in retrieve_candidates(retriever, query_def, example.entity_type, top_k)
        ]

        query_lab_raw = row.get("query_lab")
        if query_lab_raw:
            query_lab = retrieval_query_from_grounding(example, query_lab_raw)
            fact.rankings[(hyp_idx, "lab")] = [
                _compact(candidate)
                for candidate in retrieve_candidates(retriever, query_lab, example.entity_type, top_k)
            ]

        processed += 1
        if log_every and processed % log_every == 0:
            print(f"dev replay: {processed} (fact, slot) retrievals done", flush=True)

    return facts


def dev_replay_sanity_check(
    facts: dict[int, FactRecord],
    retrievals_path: Path = FHS_ROOT / "runs" / "runs_ags_specialized_generators" / "qwen3_32b" / "retrievals.jsonl",
    sample: int = 200,
) -> dict[str, Any]:
    """Spot-check the replay against Task A's own logged gold_rank for the same slots --
    catches a query-construction or retriever-settings mismatch before any beta sweep trusts
    the replay."""
    checked = 0
    mismatches: list[dict[str, Any]] = []
    for row in stream_jsonl(retrievals_path):
        slot = row.get("slot")
        if slot not in DEV_SLOT_TO_HYPOTHESIS_IDX:
            continue
        if checked >= sample:
            break
        fact_id = int(row["fact_id"])
        hyp_idx = DEV_SLOT_TO_HYPOTHESIS_IDX[slot]
        rendering = row["rendering"]
        fact = facts.get(fact_id)
        if fact is None:
            continue
        replayed = fact.rankings.get((hyp_idx, rendering))
        checked += 1
        logged_gold_rank = row.get("gold_rank")
        replayed_gold_rank = None
        if replayed:
            gold = set(fact.gold_tags)
            for candidate in replayed:
                if normalize_tag(candidate["tag"]) in gold:
                    replayed_gold_rank = candidate["rank"]
                    break
        if logged_gold_rank != replayed_gold_rank:
            mismatches.append(
                {
                    "fact_id": fact_id,
                    "slot": slot,
                    "rendering": rendering,
                    "logged_gold_rank": logged_gold_rank,
                    "replayed_gold_rank": replayed_gold_rank,
                }
            )
    return {"checked": checked, "mismatches": mismatches, "mismatch_rate": round(len(mismatches) / checked, 6) if checked else None}
