#!/usr/bin/env python3
"""Coverage headroom pilot for AGS-style grounding.

This is a self-contained pilot over the train/development split. It freezes a
context-level sample, runs recall-only retrieval arms, and reports paired
within-sample differences without touching the main comparison outputs.
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
import random
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from run_fintagging_grounding_baseline import (
    DEFAULT_TAXONOMY_JSONL,
    SCRIPT_DIR,
    Example,
    QueryGenerator,
    TaxonomyRetriever,
    build_direct_query,
    build_parallel_sampling_messages,
    build_prompt_under_query_budget,
    build_query_description_messages,
    first_gold_rank,
    fuse_round_candidates,
    html_to_visible_text,
    llm_call_record,
    load_taxonomy,
    messages_to_prompt,
    normalize_space,
    normalize_tag,
    parse_json_value,
    parse_query_description,
    retrieval_query_from_grounding,
    retrieve_candidates,
    scalar_text,
    serialize_evidence,
    write_jsonl,
)


ARM_ORDER = ("A", "B", "B_prime", "C")
ARM_NAMES = {
    "A": "one_pass_reference",
    "B": "temperature_one_pass",
    "B_prime": "diversity_prompt_sampling",
    "C": "dimension_directed",
}
DIMENSION_ASSIGNMENTS = (
    ("FAMILY", "a different accounting family for the same value"),
    ("TEMPORAL", "a different period interpretation such as instant versus duration or relative period"),
    ("QUALIFIER_AGGREGATION", "a different gross/net, total/component, or before/after-adjustment reading"),
    ("SCOPE", "a different consolidated, segment, class, plan, or other scoped reading"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FHS_ROOT / "runs" / "runs_ags_coverage_pilot" / "qwen3_32b")
    parser.add_argument("--original-dir", type=Path, default=FHS_ROOT / "data" / "source" / "FinTagging_800_200_HF")
    parser.add_argument("--table-context-dir", type=Path, default=FHS_ROOT / "data" / "source" / "FinTagging_800_200_table_context_HF")
    parser.add_argument("--text-context-dir", type=Path, default=FHS_ROOT / "data" / "source" / "FinTagging_800_200_text_context_HF")
    parser.add_argument("--taxonomy-jsonl", type=Path, default=DEFAULT_TAXONOMY_JSONL)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--table-contexts", type=int, default=30)
    parser.add_argument("--text-contexts", type=int, default=40)
    parser.add_argument("--target-facts", type=int, default=600)
    parser.add_argument("--target-facts-weight", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=20260723)
    parser.add_argument("--sample-attempts", type=int, default=2000)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--depths", default="10,50,200")
    parser.add_argument("--rrf-kappa", type=float, default=60.0)
    parser.add_argument("--type-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--dry-run-no-llm", action="store_true")

    parser.add_argument("--query-generation-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--query-generation-backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--query-context-max-chars", type=int, default=12000)
    parser.add_argument("--query-max-input-tokens", type=int, default=16000)
    parser.add_argument("--query-max-new-tokens", type=int, default=128)
    parser.add_argument("--arm-c-max-new-tokens", type=int, default=512)
    parser.add_argument("--query-top-p", type=float, default=1.0)
    parser.add_argument("--arm-a-temperature", type=float, default=0.0)
    parser.add_argument("--arm-b-temperature-schedule", default="0.8,1.0,1.2")
    parser.add_argument("--arm-b-jaccard-threshold", type=float, default=0.8)
    parser.add_argument("--arm-bprime-temperature", type=float, default=0.0)
    parser.add_argument("--arm-c-temperature", type=float, default=0.0)
    parser.add_argument("--probe-min-r200", type=float, default=0.95)
    parser.add_argument("--arm-a-reference-r200", type=float, default=0.69)
    parser.add_argument("--arm-a-reference-tolerance", type=float, default=0.10)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--vllm-batch-size", type=int, default=32)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--query-temperature", type=float, default=0.0)
    return parser.parse_args()


def parse_depths(value: str, top_k: int) -> list[int]:
    depths = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not depths:
        raise ValueError("--depths must contain at least one integer")
    if any(depth <= 0 for depth in depths):
        raise ValueError("--depths values must be positive")
    if max(depths) > top_k:
        raise ValueError(f"--depths cannot exceed --top-k={top_k}")
    return depths


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("temperature schedule cannot be empty")
    return values


def expand_temperature_schedule(schedule: list[float], count: int) -> list[float]:
    if count <= 0:
        return []
    if len(schedule) >= count:
        return schedule[:count]
    return schedule + [schedule[-1]] * (count - len(schedule))


def iter_records(value: object) -> Iterable[dict[str, Any]]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        for entry in value:
            if isinstance(entry, dict):
                yield entry


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def input_group_key(input_type: str, input_fields: dict[str, Any]) -> str:
    return json.dumps({"input_type": input_type, "input_fields": input_fields}, ensure_ascii=False, sort_keys=True)


def load_original_rows(original_dir: Path, split: str) -> dict[int, dict[str, Any]]:
    df = pd.read_parquet(original_dir / "data" / f"{split}.parquet")
    rows: dict[int, dict[str, Any]] = {}
    for row_idx, row in df.iterrows():
        source_sample_idx = int(row.get("source_sample_idx", row_idx))
        rows[source_sample_idx] = {
            "context_id": int(row.get("context_id", source_sample_idx)),
            "original_context": str(row.get("text", "")),
            "numeric_entities": [to_jsonable(item) for item in iter_records(row.get("numeric_entities", []))],
        }
    return rows


def concept_for_entity(original_rows: dict[int, dict[str, Any]], source_sample_idx: int, entity_index: int) -> str:
    entities = original_rows[source_sample_idx]["numeric_entities"]
    if entity_index < 0 or entity_index >= len(entities):
        raise IndexError(f"entity_index={entity_index} is invalid for source_sample_idx={source_sample_idx}")
    return normalize_tag(entities[entity_index].get("concept"))


def build_input_fields(input_type: str, output_entity: dict[str, Any], original_context: str) -> dict[str, Any]:
    if input_type == "table":
        return {
            "numeric_entity": output_entity.get("numeric_entity"),
            "datatype": output_entity.get("datatype"),
            "row_context": output_entity.get("row_context"),
            "column_context": output_entity.get("column_context"),
            "original_context": original_context,
        }
    if input_type == "text":
        return {
            "numeric_entity": output_entity.get("numeric_entity"),
            "datatype": output_entity.get("datatype"),
            "sentence_context": output_entity.get("sentence_context"),
            "original_context": original_context,
        }
    raise ValueError(f"Unsupported input_type={input_type}")


def add_context_rows(
    grouped: dict[str, dict[str, Any]],
    context_df: pd.DataFrame,
    input_type: str,
    original_rows: dict[int, dict[str, Any]],
) -> None:
    for _, row in context_df.iterrows():
        source_sample_idx = int(row["source_sample_idx"])
        if source_sample_idx not in original_rows:
            continue
        context_id = int(row["context_id"])
        original_context = original_rows[source_sample_idx]["original_context"]
        output_entities = [to_jsonable(item) for item in iter_records(row.get("output_entities", []))]
        entity_metadata = [to_jsonable(item) for item in iter_records(row.get("entity_metadata", []))]
        if len(output_entities) != len(entity_metadata):
            raise ValueError(f"Output/entity metadata length mismatch for source_sample_idx={source_sample_idx}")

        for output_entity, metadata in zip(output_entities, entity_metadata, strict=True):
            entity_index = int(metadata["entity_index"])
            concept = concept_for_entity(original_rows, source_sample_idx, entity_index)
            input_fields = build_input_fields(input_type, output_entity, original_context)
            key = input_group_key(input_type, input_fields)
            if key not in grouped:
                grouped[key] = {
                    "source_sample_idx": source_sample_idx,
                    "context_id": context_id,
                    "split": "train",
                    "input_type": input_type,
                    "input": json.dumps(input_fields, ensure_ascii=False),
                    "input_fields": input_fields,
                    "ground_truth_concepts": [],
                    "source_entity_indices": [],
                    "source_match_statuses": [],
                }
            grouped[key]["ground_truth_concepts"].append(concept)
            grouped[key]["source_entity_indices"].append(entity_index)
            grouped[key]["source_match_statuses"].append(metadata.get("match_status"))


def load_grounding_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    original_rows = load_original_rows(args.original_dir, args.split)
    table_df = pd.read_parquet(args.table_context_dir / "data" / f"{args.split}.parquet")
    text_df = pd.read_parquet(args.text_context_dir / "data" / f"{args.split}.parquet")
    grouped: dict[str, dict[str, Any]] = {}
    add_context_rows(grouped, table_df, "table", original_rows)
    add_context_rows(grouped, text_df, "text", original_rows)

    rows: list[dict[str, Any]] = []
    for grouped_row in grouped.values():
        concepts = ordered_unique(grouped_row["ground_truth_concepts"])
        row = {
            **grouped_row,
            "ground_truth_concepts": concepts,
            "output": json.dumps(concepts, ensure_ascii=False),
            "ground_truth_count": len(concepts),
            "source_entity_indices": [int(value) for value in grouped_row["source_entity_indices"]],
            "source_occurrence_count": len(grouped_row["source_entity_indices"]),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["source_sample_idx"],
            row["input_type"],
            min(row["source_entity_indices"]) if row["source_entity_indices"] else -1,
            row["input"],
        )
    )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fact_type_counts = Counter(row["input_type"] for row in rows)
    context_keys = {(row["input_type"], int(row["source_sample_idx"])) for row in rows}
    context_type_counts = Counter(input_type for input_type, _ in context_keys)
    facts_by_context: Counter[str] = Counter(
        f"{row['input_type']}:{int(row['source_sample_idx'])}" for row in rows
    )
    datatype_counts = Counter(row["input_fields"].get("datatype") for row in rows)
    concept_counts: Counter[str] = Counter()
    for row in rows:
        concept_counts.update(row.get("ground_truth_concepts", []))

    fact_count = len(rows)
    context_count = len(context_keys)
    facts_per_context = list(facts_by_context.values())
    by_input_type: dict[str, Any] = {}
    for input_type in ("table", "text"):
        type_rows = [row for row in rows if row["input_type"] == input_type]
        type_contexts = {int(row["source_sample_idx"]) for row in type_rows}
        type_facts_by_context = Counter(int(row["source_sample_idx"]) for row in type_rows)
        type_fpc = list(type_facts_by_context.values())
        type_concepts: Counter[str] = Counter()
        for row in type_rows:
            type_concepts.update(row.get("ground_truth_concepts", []))
        by_input_type[input_type] = {
            "fact_count": len(type_rows),
            "context_count": len(type_contexts),
            "fact_ratio": round(len(type_rows) / fact_count, 6) if fact_count else 0.0,
            "facts_per_context": {
                "mean": round(sum(type_fpc) / len(type_fpc), 6) if type_fpc else 0.0,
                "min": min(type_fpc) if type_fpc else 0,
                "max": max(type_fpc) if type_fpc else 0,
            },
            "unique_concept_count": len(type_concepts),
            "unique_concepts_per_context": round(len(type_concepts) / len(type_contexts), 6)
            if type_contexts
            else 0.0,
        }
    return {
        "fact_count": fact_count,
        "context_count": context_count,
        "context_type_counts": dict(sorted(context_type_counts.items())),
        "fact_type_counts": dict(sorted(fact_type_counts.items())),
        "fact_type_ratio": {
            key: round(value / fact_count, 6) for key, value in sorted(fact_type_counts.items())
        }
        if fact_count
        else {},
        "datatype_counts": {str(key): int(value) for key, value in sorted(datatype_counts.items())},
        "datatype_ratio": {
            str(key): round(value / fact_count, 6) for key, value in sorted(datatype_counts.items())
        }
        if fact_count
        else {},
        "unique_concept_count": len(concept_counts),
        "unique_concept_rate": round(len(concept_counts) / fact_count, 6) if fact_count else 0.0,
        "facts_per_context": {
            "mean": round(sum(facts_per_context) / len(facts_per_context), 6) if facts_per_context else 0.0,
            "min": min(facts_per_context) if facts_per_context else 0,
            "max": max(facts_per_context) if facts_per_context else 0,
        },
        "by_input_type": by_input_type,
        "multi_tag_fact_count": sum(1 for row in rows if len(row.get("ground_truth_concepts", [])) > 1),
    }


def distribution_l1(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys)


def sample_quality(sample_summary: dict[str, Any], reference_summary: dict[str, Any]) -> dict[str, Any]:
    table_delta = abs(
        float(sample_summary.get("fact_type_ratio", {}).get("table", 0.0))
        - float(reference_summary.get("fact_type_ratio", {}).get("table", 0.0))
    )
    datatype_l1 = distribution_l1(
        sample_summary.get("datatype_ratio", {}),
        reference_summary.get("datatype_ratio", {}),
    )
    fpc_by_type = {}
    for input_type in ("table", "text"):
        reference_fpc = float(
            reference_summary.get("by_input_type", {}).get(input_type, {}).get("facts_per_context", {}).get("mean", 0.0)
        )
        sample_fpc = float(
            sample_summary.get("by_input_type", {}).get(input_type, {}).get("facts_per_context", {}).get("mean", 0.0)
        )
        fpc_by_type[input_type] = abs(sample_fpc - reference_fpc) / reference_fpc if reference_fpc else 0.0
    fpc_rel_delta = max(fpc_by_type.values()) if fpc_by_type else 0.0
    unique_reference = float(reference_summary.get("unique_concept_count", 0.0)) / max(
        float(reference_summary.get("context_count", 0.0)),
        1.0,
    )
    unique_sample = float(sample_summary.get("unique_concept_count", 0.0)) / max(
        float(sample_summary.get("context_count", 0.0)),
        1.0,
    )
    unique_delta = abs(unique_sample - unique_reference) / unique_reference if unique_reference else 0.0
    flags = []
    if table_delta > 0.08:
        flags.append("table_fact_ratio_delta_gt_0.08")
    if datatype_l1 > 0.15:
        flags.append("datatype_l1_delta_gt_0.15")
    if fpc_rel_delta > 0.35:
        flags.append("by_modality_facts_per_context_relative_delta_gt_0.35")
    if unique_delta > 0.60:
        flags.append("unique_concepts_per_context_relative_delta_gt_0.60")
    return {
        "table_fact_ratio_abs_delta": round(table_delta, 6),
        "datatype_ratio_l1_delta": round(datatype_l1, 6),
        "facts_per_context_relative_delta_by_input_type": {
            key: round(value, 6) for key, value in sorted(fpc_by_type.items())
        },
        "max_facts_per_context_relative_delta": round(fpc_rel_delta, 6),
        "unique_concepts_per_context_relative_delta": round(unique_delta, 6),
        "flags": flags,
        "passed_soft_checks": not flags,
    }


def choose_context_sample(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_context: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_context[(row["input_type"], int(row["source_sample_idx"]))].append(row)
    table_keys = sorted(key for key in by_context if key[0] == "table")
    text_keys = sorted(key for key in by_context if key[0] == "text")
    if args.table_contexts > len(table_keys) or args.text_contexts > len(text_keys):
        raise ValueError(
            f"Requested {args.table_contexts} table and {args.text_contexts} text contexts, "
            f"but only {len(table_keys)} table and {len(text_keys)} text contexts are available."
        )

    reference = summarize_rows(rows)
    rng = random.Random(args.sample_seed)
    best_rows: list[dict[str, Any]] | None = None
    best_quality: dict[str, Any] | None = None
    best_score: float | None = None

    for _ in range(max(args.sample_attempts, 1)):
        selected = set(rng.sample(table_keys, args.table_contexts) + rng.sample(text_keys, args.text_contexts))
        candidate_rows = [row for key in sorted(selected) for row in by_context[key]]
        summary = summarize_rows(candidate_rows)
        quality = sample_quality(summary, reference)
        score = (
            quality["table_fact_ratio_abs_delta"]
            + 0.5 * quality["datatype_ratio_l1_delta"]
            + 0.2 * quality["max_facts_per_context_relative_delta"]
            + 0.2 * quality["unique_concepts_per_context_relative_delta"]
        )
        if args.target_facts > 0:
            score += args.target_facts_weight * abs(summary["fact_count"] - args.target_facts) / args.target_facts
        if best_score is None or score < best_score:
            best_rows = candidate_rows
            best_quality = quality
            best_score = score
            if quality["passed_soft_checks"]:
                break

    if best_rows is None or best_quality is None:
        raise RuntimeError("Failed to draw a context sample")

    sample_rows = []
    for fact_id, row in enumerate(best_rows):
        sample_rows.append({"fact_id": fact_id, **row})

    sample_summary = summarize_rows(sample_rows)
    sample_summary.update(
        {
            "split": args.split,
            "sample_seed": args.sample_seed,
            "sample_attempts": args.sample_attempts,
            "target_facts": args.target_facts,
            "target_facts_weight": args.target_facts_weight,
            "requested_contexts": {"table": args.table_contexts, "text": args.text_contexts},
            "reference_train_or_split_summary": reference,
            "quality_vs_reference": best_quality,
        }
    )
    return sample_rows, sample_summary


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_to_example(row: dict[str, Any]) -> Example:
    fields = row["input_fields"]
    original_context = normalize_space(fields.get("original_context", ""))
    return Example(
        example_idx=int(row["fact_id"]),
        context_id=row.get("context_id"),
        source_sample_idx=row.get("source_sample_idx"),
        input_type=normalize_space(row.get("input_type", "")),
        entity=normalize_space(fields.get("numeric_entity", "")),
        entity_type=normalize_space(fields.get("datatype", "")),
        row_context=normalize_space(fields.get("row_context", "")),
        column_context=normalize_space(fields.get("column_context", "")),
        original_context=original_context,
        query_context=html_to_visible_text(original_context),
        gold_tags=[normalize_tag(tag) for tag in row.get("ground_truth_concepts", []) if normalize_tag(tag)],
    )


def prepare_sample(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[Example], dict[str, Any]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "sample_facts.jsonl"
    summary_path = args.output_dir / "sample_summary.json"
    if args.resume and sample_path.exists() and summary_path.exists() and not args.overwrite:
        sample_rows = load_jsonl(sample_path)
        sample_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        all_rows = load_grounding_rows(args)
        sample_rows, sample_summary = choose_context_sample(all_rows, args)
        write_jsonl(sample_path, sample_rows)
        summary_path.write_text(json.dumps(sample_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    examples = [row_to_example(row) for row in sample_rows]
    return sample_rows, examples, sample_summary


@contextmanager
def temporary_generation_settings(
    generator: QueryGenerator,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> Iterable[None]:
    old_temperature = generator.args.query_temperature
    old_top_p = generator.args.query_top_p
    old_max_new_tokens = generator.args.query_max_new_tokens
    generator.args.query_temperature = temperature
    generator.args.query_top_p = top_p
    generator.args.query_max_new_tokens = max_new_tokens
    try:
        yield
    finally:
        generator.args.query_temperature = old_temperature
        generator.args.query_top_p = old_top_p
        generator.args.query_max_new_tokens = old_max_new_tokens


def build_dimension_directed_messages(example: Example, budget: int, context_max_chars: int) -> list[dict[str, str]]:
    evidence = serialize_evidence(example, context_max_chars)
    assignments = "\n".join(
        f"{idx}. {dimension}: commit to {description}."
        for idx, (dimension, description) in enumerate(DIMENSION_ASSIGNMENTS[:budget], start=1)
    )
    schema_items = ", ".join(
        f'{{"hypothesis_idx": {idx - 1}, "varied_dimension": "{dimension}", "query": "complete retrieval description"}}'
        for idx, (dimension, _) in enumerate(DIMENSION_ASSIGNMENTS[:budget], start=1)
    )
    user = f"""Generate {budget} complete, self-consistent semantic retrieval hypotheses for the financial evidence.

Each hypothesis must be a full interpretation that could independently retrieve the correct US-GAAP XBRL concept. The hypotheses should compete with one another as plausible readings; do not emit fragments that must be combined.

Dimension assignments:
{assignments}

Return JSON only with this schema:
{{"hypotheses": [{schema_items}]}}

Evidence:
{evidence}

Rules:
- Every query must describe the entity, accounting concept, temporal reading, qualifier or aggregation, and scope as far as the evidence supports them.
- The named varied_dimension is the dimension to deliberately explore differently.
- Hold the other dimensions as your best inference from the evidence.
- Do not name a specific US-GAAP tag unless it is explicitly present in the source context.
- Do not include explanations or markdown."""
    return [
        {
            "role": "system",
            "content": "You generate complete dimension-directed US-GAAP retrieval hypotheses.",
        },
        {"role": "user", "content": user},
    ]


def parse_dimension_hypotheses(raw_output: str, fallback: str, budget: int) -> tuple[list[dict[str, str]], bool]:
    parsed, parse_ok = parse_json_value(raw_output)
    hypotheses: list[dict[str, str]] = []
    values: Any = None
    if isinstance(parsed, dict):
        values = parsed.get("hypotheses") or parsed.get("queries") or parsed.get("interpretations")
    elif isinstance(parsed, list):
        values = parsed

    if isinstance(values, list):
        for idx, item in enumerate(values[:budget]):
            if isinstance(item, dict):
                query = scalar_text(item.get("query") or item.get("retrieval_query") or item.get("description"))
                dimension = scalar_text(item.get("varied_dimension") or item.get("dimension") or item.get("focus"))
            else:
                query = scalar_text(item)
                dimension = ""
            default_dimension = DIMENSION_ASSIGNMENTS[idx][0] if idx < len(DIMENSION_ASSIGNMENTS) else f"HYPOTHESIS_{idx}"
            hypotheses.append(
                {
                    "query": query or fallback,
                    "varied_dimension": dimension or default_dimension,
                }
            )

    if not hypotheses:
        cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL | re.IGNORECASE).strip()
        for line in cleaned.splitlines():
            line = normalize_space(re.sub(r"^(?:hypothesis|query|interpretation)\s*\d+\s*[:.)-]\s*", "", line, flags=re.I))
            if line:
                idx = len(hypotheses)
                default_dimension = DIMENSION_ASSIGNMENTS[idx][0] if idx < len(DIMENSION_ASSIGNMENTS) else f"HYPOTHESIS_{idx}"
                hypotheses.append({"query": line, "varied_dimension": default_dimension})
            if len(hypotheses) >= budget:
                break

    while len(hypotheses) < budget:
        idx = len(hypotheses)
        default_dimension = DIMENSION_ASSIGNMENTS[idx][0] if idx < len(DIMENSION_ASSIGNMENTS) else f"FALLBACK_{idx}"
        hypotheses.append({"query": fallback, "varied_dimension": default_dimension})

    return hypotheses[:budget], bool(parse_ok and len(hypotheses) >= budget)


def prompt_under_budget(
    generator: QueryGenerator | None,
    args: argparse.Namespace,
    message_builder: Any,
) -> tuple[str, int, int]:
    if generator is None:
        return "", 0, args.query_context_max_chars
    return build_prompt_under_query_budget(
        generator.tokenizer,
        message_builder,
        context_max_chars=args.query_context_max_chars,
        max_input_tokens=args.query_max_input_tokens,
    )


def fake_one_pass_output(example: Example, hypothesis_idx: int = 0) -> str:
    query = build_direct_query(example)
    if hypothesis_idx:
        query = f"{query} alternative interpretation {hypothesis_idx}"
    return json.dumps({"query": query}, ensure_ascii=False)


def fake_dimension_output(example: Example, budget: int) -> str:
    hypotheses = []
    base = build_direct_query(example)
    for idx, (dimension, _) in enumerate(DIMENSION_ASSIGNMENTS[:budget]):
        hypotheses.append(
            {
                "hypothesis_idx": idx,
                "varied_dimension": dimension,
                "query": f"{base} dimension-directed {dimension.lower()} interpretation",
            }
        )
    return json.dumps({"hypotheses": hypotheses}, ensure_ascii=False)


def compact_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": int(candidate["rank"]),
            "tag": candidate["tag"],
            "type": candidate.get("type", ""),
            "standard_label": candidate.get("standard_label", ""),
            "bm25_score": candidate.get("bm25_score"),
            "bm25_normalized_score": candidate.get("bm25_normalized_score"),
            "label_coverage": candidate.get("label_coverage"),
            "query_label_coverage": candidate.get("query_label_coverage"),
            "retrieval_score": candidate.get("retrieval_score"),
        }
        for candidate in candidates
    ]


def retrieve_hypothesis(
    retriever: TaxonomyRetriever,
    example: Example,
    query_text: str,
    top_k: int,
) -> tuple[str, list[dict[str, Any]]]:
    retrieval_query = retrieval_query_from_grounding(example, query_text)
    candidates = compact_candidates(retrieve_candidates(retriever, retrieval_query, example.entity_type, top_k))
    return retrieval_query, candidates


def hypothesis_record(
    arm: str,
    example: Example,
    hypothesis_idx: int,
    varied_dimension: str | None,
    query_text: str,
    retrieval_query: str,
    candidates: list[dict[str, Any]],
    llm_call: dict[str, Any],
    generation_temperature: float,
) -> dict[str, Any]:
    candidate_ids = [candidate["tag"] for candidate in candidates]
    candidate_scores = [candidate.get("bm25_score") for candidate in candidates]
    return {
        "fact_id": example.example_idx,
        "context_id": example.context_id,
        "source_sample_idx": example.source_sample_idx,
        "modality": example.input_type,
        "arm": arm,
        "arm_name": ARM_NAMES[arm],
        "hypothesis_idx": hypothesis_idx,
        "varied_dimension": varied_dimension,
        "query_text": query_text,
        "retrieval_query": retrieval_query,
        "candidate_ids": candidate_ids,
        "candidate_scores": candidate_scores,
        "candidates": candidates,
        "gold_concept_ids": example.gold_tags,
        "gold_rank": first_gold_rank(candidate_ids, example.gold_tags),
        "generation_temperature": generation_temperature,
        "llm_call": llm_call,
    }


def run_one_prompt_arm(
    args: argparse.Namespace,
    arm: str,
    examples: list[Example],
    retriever: TaxonomyRetriever,
    generator: QueryGenerator | None,
    output_path: Path,
    hypotheses_per_fact: int,
    temperature: float,
    prompt_kind: str,
    hypothesis_temperatures: list[float] | None = None,
) -> list[dict[str, Any]]:
    if args.resume and output_path.exists() and not args.overwrite:
        return load_jsonl(output_path)

    records: list[dict[str, Any]] = []
    prompts: list[str] = []
    meta: list[tuple[Example, int, int, int, float]] = []
    if hypothesis_temperatures is None:
        hypothesis_temperatures = [temperature] * hypotheses_per_fact
    if len(hypothesis_temperatures) != hypotheses_per_fact:
        raise ValueError(
            f"Expected {hypotheses_per_fact} hypothesis temperatures, got {len(hypothesis_temperatures)}"
        )
    for example in examples:
        for hypothesis_idx in range(hypotheses_per_fact):
            generation_temperature = hypothesis_temperatures[hypothesis_idx]
            if prompt_kind == "one_pass":
                prompt, prompt_tokens, used_context_chars = prompt_under_budget(
                    generator,
                    args,
                    lambda ctx_chars, ex=example: build_query_description_messages(ex, ctx_chars),
                )
            elif prompt_kind == "diversity":
                prompt, prompt_tokens, used_context_chars = prompt_under_budget(
                    generator,
                    args,
                    lambda ctx_chars, ex=example, idx=hypothesis_idx: build_parallel_sampling_messages(
                        ex,
                        idx + 1,
                        hypotheses_per_fact,
                        ctx_chars,
                    ),
                )
            else:
                raise ValueError(f"Unsupported prompt_kind={prompt_kind}")
            prompts.append(prompt)
            meta.append((example, hypothesis_idx, prompt_tokens, used_context_chars, generation_temperature))

    if args.dry_run_no_llm:
        raw_outputs = [
            fake_one_pass_output(example, hypothesis_idx)
            for example, hypothesis_idx, _, _, _ in meta
        ]
    else:
        assert generator is not None
        raw_outputs = [""] * len(prompts)
        indices_by_temperature: dict[float, list[int]] = defaultdict(list)
        for prompt_idx, (*_, generation_temperature) in enumerate(meta):
            indices_by_temperature[generation_temperature].append(prompt_idx)
        for generation_temperature, prompt_indices in sorted(indices_by_temperature.items()):
            batch_prompts = [prompts[prompt_idx] for prompt_idx in prompt_indices]
            with temporary_generation_settings(
                generator,
                generation_temperature,
                args.query_top_p,
                args.query_max_new_tokens,
            ):
                batch_outputs = generator.generate_many(batch_prompts)
            for prompt_idx, raw_output in zip(prompt_indices, batch_outputs, strict=True):
                raw_outputs[prompt_idx] = raw_output

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for output_idx, (
            raw_output,
            (example, hypothesis_idx, prompt_tokens, used_context_chars, generation_temperature),
        ) in enumerate(
            zip(raw_outputs, meta, strict=True),
            start=1,
        ):
            fallback = build_direct_query(example)
            query_text, parse_ok = parse_query_description(raw_output, fallback)
            llm_call = llm_call_record(
                prompt_kind,
                raw_output=raw_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=0 if generator is None else generator.count_text_tokens(raw_output),
                parse_ok=parse_ok,
                backend="dry_run" if generator is None else generator.backend,
                model_name="dry_run" if generator is None else generator.model_name,
                extra_fields={
                    "used_context_max_chars": used_context_chars,
                    "hypothesis_idx": hypothesis_idx,
                    "generation_temperature": generation_temperature,
                },
            )
            retrieval_query, candidates = retrieve_hypothesis(retriever, example, query_text, args.top_k)
            record = hypothesis_record(
                arm=arm,
                example=example,
                hypothesis_idx=hypothesis_idx,
                varied_dimension=None,
                query_text=query_text,
                retrieval_query=retrieval_query,
                candidates=candidates,
                llm_call=llm_call,
                generation_temperature=generation_temperature,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            if output_idx % max(args.log_every, 1) == 0 or output_idx == len(raw_outputs):
                print(f"Built arm {arm} hypotheses {output_idx}/{len(raw_outputs)}")
    return records


def run_dimension_arm(
    args: argparse.Namespace,
    examples: list[Example],
    retriever: TaxonomyRetriever,
    generator: QueryGenerator | None,
    output_path: Path,
) -> list[dict[str, Any]]:
    if args.resume and output_path.exists() and not args.overwrite:
        return load_jsonl(output_path)

    prompts: list[str] = []
    meta: list[tuple[Example, int, int]] = []
    for example in examples:
        prompt, prompt_tokens, used_context_chars = prompt_under_budget(
            generator,
            args,
            lambda ctx_chars, ex=example: build_dimension_directed_messages(ex, args.budget, ctx_chars),
        )
        prompts.append(prompt)
        meta.append((example, prompt_tokens, used_context_chars))

    if args.dry_run_no_llm:
        raw_outputs = [fake_dimension_output(example, args.budget) for example, _, _ in meta]
    else:
        assert generator is not None
        with temporary_generation_settings(generator, args.arm_c_temperature, args.query_top_p, args.arm_c_max_new_tokens):
            raw_outputs = generator.generate_many(prompts)

    records: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for output_idx, (raw_output, (example, prompt_tokens, used_context_chars)) in enumerate(
            zip(raw_outputs, meta, strict=True),
            start=1,
        ):
            fallback = build_direct_query(example)
            hypotheses, parse_ok = parse_dimension_hypotheses(raw_output, fallback, args.budget)
            shared_call = llm_call_record(
                "dimension_directed",
                raw_output=raw_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=0 if generator is None else generator.count_text_tokens(raw_output),
                parse_ok=parse_ok,
                backend="dry_run" if generator is None else generator.backend,
                model_name="dry_run" if generator is None else generator.model_name,
                extra_fields={"used_context_max_chars": used_context_chars, "hypothesis_count": args.budget},
            )
            for hypothesis_idx, hypothesis in enumerate(hypotheses):
                query_text = hypothesis["query"]
                retrieval_query, candidates = retrieve_hypothesis(retriever, example, query_text, args.top_k)
                record = hypothesis_record(
                    arm="C",
                    example=example,
                    hypothesis_idx=hypothesis_idx,
                    varied_dimension=hypothesis.get("varied_dimension"),
                    query_text=query_text,
                    retrieval_query=retrieval_query,
                    candidates=candidates,
                    llm_call=shared_call,
                    generation_temperature=args.arm_c_temperature,
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
            if output_idx % max(args.log_every, 1) == 0 or output_idx == len(raw_outputs):
                print(f"Built arm C facts {output_idx}/{len(raw_outputs)}")
    return records


def mean_pairwise_jaccard(candidate_lists: list[list[str]], depth: int) -> float | None:
    if len(candidate_lists) < 2:
        return None
    values: list[float] = []
    for left_idx in range(len(candidate_lists)):
        left = set(candidate_lists[left_idx][:depth])
        for right_idx in range(left_idx + 1, len(candidate_lists)):
            right = set(candidate_lists[right_idx][:depth])
            union = left | right
            values.append((len(left & right) / len(union)) if union else 0.0)
    return sum(values) / len(values) if values else None


def records_by_arm_fact(records: Iterable[dict[str, Any]]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record["arm"]][int(record["fact_id"])].append(record)
    for arm_records in grouped.values():
        for fact_records in arm_records.values():
            fact_records.sort(key=lambda record: int(record["hypothesis_idx"]))
    return grouped


def fact_context_key(example: Example) -> str:
    return f"{example.input_type}:{example.source_sample_idx}"


def compute_fact_metrics(
    arm_records: dict[str, dict[int, list[dict[str, Any]]]],
    examples: list[Example],
    depths: list[int],
    rrf_kappa: float,
) -> dict[str, dict[int, dict[int, dict[str, Any]]]]:
    metrics: dict[str, dict[int, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    example_by_id = {example.example_idx: example for example in examples}
    max_depth = max(depths)

    for arm, records_by_fact in arm_records.items():
        for fact_id, fact_records in records_by_fact.items():
            example = example_by_id[fact_id]
            rounds = [
                {
                    "round": idx + 1,
                    "candidates": record["candidates"],
                }
                for idx, record in enumerate(fact_records)
            ]
            fused = fuse_round_candidates(rounds, max_depth, rrf_kappa)
            fused_ids = [candidate["tag"] for candidate in fused]
            candidate_lists = [record["candidate_ids"] for record in fact_records]
            for depth in depths:
                union_ids: set[str] = set()
                marginal: dict[int, bool] = {}
                for idx, candidate_ids in enumerate(candidate_lists, start=1):
                    union_ids.update(candidate_ids[:depth])
                    marginal[idx] = any(normalize_tag(tag) in set(example.gold_tags) for tag in union_ids)
                gold = set(example.gold_tags)
                coverage = any(normalize_tag(tag) in gold for tag in union_ids)
                rrf_rank = first_gold_rank(fused_ids[:depth], example.gold_tags)
                round1_rank = first_gold_rank(candidate_lists[0][:depth], example.gold_tags) if candidate_lists else None
                metrics[arm][fact_id][depth] = {
                    "coverage": bool(coverage),
                    "rrf_recall": rrf_rank is not None,
                    "rrf_rank": rrf_rank,
                    "round1_recall": round1_rank is not None,
                    "round1_rank": round1_rank,
                    "mean_jaccard": mean_pairwise_jaccard(candidate_lists, depth),
                    "marginal_coverage": marginal,
                    "hypothesis_count": len(candidate_lists),
                }
    return metrics


def selected_fact_ids(examples: list[Example], modality: str) -> list[int]:
    if modality == "pooled":
        return [example.example_idx for example in examples]
    return [example.example_idx for example in examples if example.input_type == modality]


def context_count_for_facts(examples_by_id: dict[int, Example], fact_ids: Iterable[int]) -> int:
    return len({fact_context_key(examples_by_id[fact_id]) for fact_id in fact_ids})


def aggregate_arm_metrics(
    fact_metrics: dict[str, dict[int, dict[int, dict[str, Any]]]],
    examples: list[Example],
    depths: list[int],
) -> list[dict[str, Any]]:
    examples_by_id = {example.example_idx: example for example in examples}
    rows: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        if arm not in fact_metrics:
            continue
        for modality in ("pooled", "table", "text"):
            fact_ids = [fact_id for fact_id in selected_fact_ids(examples, modality) if fact_id in fact_metrics[arm]]
            for depth in depths:
                values = [fact_metrics[arm][fact_id][depth] for fact_id in fact_ids]
                n = len(values)
                jaccards = [value["mean_jaccard"] for value in values if value["mean_jaccard"] is not None]
                rows.append(
                    {
                        "arm": arm,
                        "arm_name": ARM_NAMES[arm],
                        "modality": modality,
                        "k": depth,
                        "fact_count": n,
                        "context_count": context_count_for_facts(examples_by_id, fact_ids),
                        "hypothesis_count": max((value["hypothesis_count"] for value in values), default=0),
                        "coverage_at_4": round(sum(value["coverage"] for value in values) / n, 6) if n else 0.0,
                        "rrf_recall_at_k": round(sum(value["rrf_recall"] for value in values) / n, 6) if n else 0.0,
                        "round1_recall_at_k": round(sum(value["round1_recall"] for value in values) / n, 6) if n else 0.0,
                        "mean_jaccard": round(sum(jaccards) / len(jaccards), 6) if jaccards else None,
                    }
                )
    return rows


def aggregate_marginal_coverage(
    fact_metrics: dict[str, dict[int, dict[int, dict[str, Any]]]],
    examples: list[Example],
    depths: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        if arm not in fact_metrics:
            continue
        for modality in ("pooled", "table", "text"):
            fact_ids = [fact_id for fact_id in selected_fact_ids(examples, modality) if fact_id in fact_metrics[arm]]
            for depth in depths:
                max_hypotheses = max(
                    (fact_metrics[arm][fact_id][depth]["hypothesis_count"] for fact_id in fact_ids),
                    default=0,
                )
                for hypothesis_count in range(1, max_hypotheses + 1):
                    values = [
                        fact_metrics[arm][fact_id][depth]["marginal_coverage"].get(hypothesis_count, False)
                        for fact_id in fact_ids
                    ]
                    n = len(values)
                    rows.append(
                        {
                            "arm": arm,
                            "arm_name": ARM_NAMES[arm],
                            "modality": modality,
                            "k": depth,
                            "hypotheses": hypothesis_count,
                            "fact_count": n,
                            "coverage": round(sum(values) / n, 6) if n else 0.0,
                        }
                    )
    return rows


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_context_ci(
    values_by_fact: dict[int, float],
    examples_by_id: dict[int, Example],
    fact_ids: list[int],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if not fact_ids:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "iterations": iterations}
    by_context: dict[str, list[int]] = defaultdict(list)
    for fact_id in fact_ids:
        by_context[fact_context_key(examples_by_id[fact_id])].append(fact_id)
    context_keys = sorted(by_context)
    observed_values = [values_by_fact[fact_id] for fact_id in fact_ids]
    observed = sum(observed_values) / len(observed_values)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled_values: list[float] = []
        for context_key in (rng.choice(context_keys) for _ in context_keys):
            sampled_values.extend(values_by_fact[fact_id] for fact_id in by_context[context_key])
        samples.append(sum(sampled_values) / len(sampled_values) if sampled_values else 0.0)
    return {
        "mean": round(observed, 6),
        "ci_low": round(percentile(samples, 0.025), 6),
        "ci_high": round(percentile(samples, 0.975), 6),
        "iterations": iterations,
        "context_count": len(context_keys),
        "fact_count": len(fact_ids),
    }


def paired_difference_rows(
    fact_metrics: dict[str, dict[int, dict[int, dict[str, Any]]]],
    examples: list[Example],
    depths: list[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    examples_by_id = {example.example_idx: example for example in examples}
    rows: list[dict[str, Any]] = []
    quantities = {
        "controller_room_C_minus_B_prime": lambda fid, k: float(fact_metrics["C"][fid][k]["coverage"])
        - float(fact_metrics["B_prime"][fid][k]["coverage"]),
        "controller_room_C_minus_B_aux": lambda fid, k: float(fact_metrics["C"][fid][k]["coverage"])
        - float(fact_metrics["B"][fid][k]["coverage"]),
        "B_prime_minus_B_coverage": lambda fid, k: float(fact_metrics["B_prime"][fid][k]["coverage"])
        - float(fact_metrics["B"][fid][k]["coverage"]),
    }
    for arm in ("B", "B_prime", "C"):
        quantities[f"total_headroom_{arm}_minus_A"] = (
            lambda fid, k, arm=arm: float(fact_metrics[arm][fid][k]["coverage"])
            - float(fact_metrics["A"][fid][k]["rrf_recall"])
        )
        quantities[f"fusion_loss_{arm}"] = (
            lambda fid, k, arm=arm: float(fact_metrics[arm][fid][k]["coverage"])
            - float(fact_metrics[arm][fid][k]["rrf_recall"])
        )

    for modality in ("pooled", "table", "text"):
        base_fact_ids = selected_fact_ids(examples, modality)
        for depth in depths:
            available = [
                fact_id
                for fact_id in base_fact_ids
                if all(fact_id in fact_metrics.get(arm, {}) for arm in ARM_ORDER)
                and all(depth in fact_metrics[arm][fact_id] for arm in ARM_ORDER)
            ]
            for idx, (quantity, func) in enumerate(quantities.items()):
                values = {fact_id: func(fact_id, depth) for fact_id in available}
                ci = bootstrap_context_ci(
                    values,
                    examples_by_id,
                    available,
                    iterations=bootstrap_samples,
                    seed=bootstrap_seed + idx + depth,
                )
                rows.append({"quantity": quantity, "modality": modality, "k": depth, **ci})
    return rows


def run_probe(
    args: argparse.Namespace,
    examples: list[Example],
    retriever: TaxonomyRetriever,
    taxonomy_by_tag: dict[str, Any],
    output_path: Path,
) -> list[dict[str, Any]]:
    if args.resume and output_path.exists() and not args.overwrite:
        return load_jsonl(output_path)
    records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for offset, example in enumerate(examples, start=1):
            best_rank: int | None = None
            best_gold: str | None = None
            per_gold = []
            for gold_tag in example.gold_tags:
                concept = taxonomy_by_tag.get(normalize_tag(gold_tag))
                if concept is None:
                    continue
                query = normalize_space(f"{concept.standard_label}. {concept.documentation}")
                candidates = compact_candidates(retrieve_candidates(retriever, query, example.entity_type, args.top_k))
                candidate_ids = [candidate["tag"] for candidate in candidates]
                rank = first_gold_rank(candidate_ids, [gold_tag])
                per_gold.append(
                    {
                        "gold_concept_id": gold_tag,
                        "query_text": query,
                        "candidate_ids": candidate_ids,
                        "candidate_scores": [candidate.get("bm25_score") for candidate in candidates],
                        "gold_rank": rank,
                    }
                )
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_gold = gold_tag
            record = {
                "fact_id": example.example_idx,
                "context_id": example.context_id,
                "source_sample_idx": example.source_sample_idx,
                "modality": example.input_type,
                "gold_concept_ids": example.gold_tags,
                "best_gold_concept_id": best_gold,
                "best_gold_rank": best_rank,
                "per_gold": per_gold,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            if offset % max(args.log_every, 1) == 0 or offset == len(examples):
                print(f"Built probe records {offset}/{len(examples)}")
    return records


def aggregate_probe_metrics(
    probe_records: list[dict[str, Any]],
    examples: list[Example],
    depths: list[int],
) -> list[dict[str, Any]]:
    by_fact = {int(record["fact_id"]): record for record in probe_records}
    rows: list[dict[str, Any]] = []
    examples_by_id = {example.example_idx: example for example in examples}
    for modality in ("pooled", "table", "text"):
        fact_ids = [fact_id for fact_id in selected_fact_ids(examples, modality) if fact_id in by_fact]
        for depth in depths:
            values = [
                by_fact[fact_id].get("best_gold_rank") is not None
                and int(by_fact[fact_id]["best_gold_rank"]) <= depth
                for fact_id in fact_ids
            ]
            n = len(values)
            rows.append(
                {
                    "modality": modality,
                    "k": depth,
                    "fact_count": n,
                    "context_count": context_count_for_facts(examples_by_id, fact_ids),
                    "probe_recall_at_k": round(sum(values) / n, 6) if n else 0.0,
                }
            )
    return rows


def first_row(rows: list[dict[str, Any]], **matches: Any) -> dict[str, Any] | None:
    for row in rows:
        if all(row.get(key) == value for key, value in matches.items()):
            return row
    return None


def build_probe_check(args: argparse.Namespace, probe_rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    probe_at_200 = {
        modality: (
            first_row(probe_rows, modality=modality, k=args.top_k) or {}
        ).get("probe_recall_at_k")
        for modality in ("pooled", "table", "text")
    }
    pooled_probe = float(probe_at_200.get("pooled") or 0.0)
    if pooled_probe < args.probe_min_r200:
        failures.append(
            {
                "check": "probe_r200",
                "message": (
                    "Probe R@200 is below the index-ceiling threshold; "
                    "pilot arm results are not interpretable until retrieval serialization is diagnosed."
                ),
                "observed": round(pooled_probe, 6),
                "threshold": args.probe_min_r200,
            }
        )
    return {
        "read_order": [
            "1. Confirm probe R@200 is near 1.0.",
            "2. Confirm Arm A one-pass R@200 is near the expected reference.",
            "3. Then inspect coverage, fusion loss, and controller_room.",
        ],
        "probe_r200": probe_at_200,
        "probe_min_r200": args.probe_min_r200,
        "failures": failures,
        "passed": not failures,
    }


def build_pre_interpretation_checks(
    args: argparse.Namespace,
    probe_rows: list[dict[str, Any]],
    report_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    probe_check = build_probe_check(args, probe_rows)
    failures = list(probe_check["failures"])
    arm_a_row = first_row(report_rows, arm="A", modality="pooled", k=args.top_k) or {}
    arm_a_r200 = float(arm_a_row.get("rrf_recall_at_k") or 0.0)
    arm_a_delta = arm_a_r200 - args.arm_a_reference_r200
    if abs(arm_a_delta) > args.arm_a_reference_tolerance:
        failures.append(
            {
                "check": "arm_a_reference_r200",
                "message": (
                    "Arm A one-pass R@200 diverges from the expected reference; "
                    "treat this as a split or preprocessing consistency issue before interpreting effects."
                ),
                "observed": round(arm_a_r200, 6),
                "reference": args.arm_a_reference_r200,
                "absolute_tolerance": args.arm_a_reference_tolerance,
            }
        )

    return {
        "read_order": probe_check["read_order"],
        "probe_r200": probe_check["probe_r200"],
        "probe_min_r200": probe_check["probe_min_r200"],
        "arm_a_r200": round(arm_a_r200, 6),
        "arm_a_reference_r200": args.arm_a_reference_r200,
        "arm_a_reference_tolerance": args.arm_a_reference_tolerance,
        "arm_a_reference_check_note": (
            "Gross-misconfiguration check only. Passing this check does not validate the setup "
            "or support any effect claim; some divergence is expected because this pilot uses a "
            "different split and sample."
        ),
        "failures": failures,
        "passed": not failures,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def compact_probe_fact_summaries(
    probe_records: list[dict[str, Any]],
    examples_by_id: dict[int, Example],
    top_k: int,
) -> list[dict[str, Any]]:
    summaries = []
    for record in probe_records:
        fact_id = int(record["fact_id"])
        example = examples_by_id.get(fact_id)
        best_rank = record.get("best_gold_rank")
        summaries.append(
            {
                "fact_id": fact_id,
                "context_id": record.get("context_id"),
                "source_sample_idx": record.get("source_sample_idx"),
                "modality": record.get("modality"),
                "datatype": example.entity_type if example is not None else None,
                "gold_concept_ids": record.get("gold_concept_ids", []),
                "best_gold_concept_id": record.get("best_gold_concept_id"),
                "best_gold_rank": best_rank,
                "hit_at_top_k": best_rank is not None and int(best_rank) <= top_k,
            }
        )
    return summaries


def summarize_probe_misses(
    probe_records: list[dict[str, Any]],
    examples_by_id: dict[int, Example],
    top_k: int,
) -> dict[str, Any]:
    summaries = compact_probe_fact_summaries(probe_records, examples_by_id, top_k)
    misses = [summary for summary in summaries if not summary["hit_at_top_k"]]
    by_modality = Counter(str(summary.get("modality")) for summary in misses)
    by_datatype = Counter(str(summary.get("datatype")) for summary in misses)
    by_gold: Counter[str] = Counter()
    for summary in misses:
        by_gold.update(summary.get("gold_concept_ids", []))
    return {
        "top_k": top_k,
        "miss_count": len(misses),
        "miss_rate": round(len(misses) / len(summaries), 6) if summaries else 0.0,
        "misses_by_modality": dict(sorted(by_modality.items())),
        "misses_by_datatype": dict(sorted(by_datatype.items())),
        "most_common_missed_gold_concepts": by_gold.most_common(25),
        "missed_fact_ids": [summary["fact_id"] for summary in misses[:100]],
    }


def compact_arm_fact_summaries(
    records: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    summaries = []
    for record in records:
        gold_rank = record.get("gold_rank")
        summaries.append(
            {
                "fact_id": int(record["fact_id"]),
                "context_id": record.get("context_id"),
                "source_sample_idx": record.get("source_sample_idx"),
                "modality": record.get("modality"),
                "arm": record.get("arm"),
                "hypothesis_idx": record.get("hypothesis_idx"),
                "generation_temperature": record.get("generation_temperature"),
                "gold_concept_ids": record.get("gold_concept_ids", []),
                "gold_rank": gold_rank,
                "hit_at_top_k": gold_rank is not None and int(gold_rank) <= top_k,
            }
        )
    return summaries


def write_abort_metrics(
    args: argparse.Namespace,
    sample_summary: dict[str, Any],
    depths: list[int],
    probe_rows: list[dict[str, Any]],
    checks: dict[str, Any],
    abort_stage: str,
    probe_records: list[dict[str, Any]] | None = None,
    examples_by_id: dict[int, Example] | None = None,
    report_rows: list[dict[str, Any]] | None = None,
    arm_records: list[dict[str, Any]] | None = None,
) -> None:
    probe_records = probe_records or []
    examples_by_id = examples_by_id or {}
    metrics = {
        "aborted": True,
        "abort_stage": abort_stage,
        "abort_reason": checks.get("failures", []),
        "artifact_paths": {
            "sample_facts": str(args.output_dir / "sample_facts.jsonl"),
            "sample_summary": str(args.output_dir / "sample_summary.json"),
            "probe_candidates": str(args.output_dir / "probe_candidates.jsonl"),
            "probe_metrics": str(args.output_dir / "probe_metrics.csv"),
            "arm_a_candidates": str(args.output_dir / "arm_A_hypothesis_candidates.jsonl"),
            "report_table": str(args.output_dir / "report_table.csv"),
        },
        "sample_summary": sample_summary,
        "pre_interpretation_checks": checks,
        "depths": depths,
        "top_k": args.top_k,
        "budget": args.budget,
        "probe": probe_rows,
        "probe_fact_summaries": compact_probe_fact_summaries(probe_records, examples_by_id, args.top_k)
        if probe_records
        else [],
        "probe_miss_summary_at_top_k": summarize_probe_misses(probe_records, examples_by_id, args.top_k)
        if probe_records
        else {},
        "arms": report_rows or [],
        "arm_fact_summaries": compact_arm_fact_summaries(arm_records or [], args.top_k),
        "decision_quantity_note": (
            "Run aborted before downstream arms because a hard pre-interpretation check failed."
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def arm_b_jaccard_diagnostic(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    examples: list[Example],
    depths: list[int],
    hypothesis_temperatures: list[float],
    records_path: Path,
) -> dict[str, Any]:
    grouped = records_by_arm_fact(records)
    fact_metrics = compute_fact_metrics(grouped, examples, depths, args.rrf_kappa)
    pooled_k200 = [
        fact_metrics["B"][example.example_idx][args.top_k]["mean_jaccard"]
        for example in examples
        if fact_metrics["B"][example.example_idx][args.top_k]["mean_jaccard"] is not None
    ]
    mean_jaccard = sum(pooled_k200) / len(pooled_k200) if pooled_k200 else 0.0
    return {
        "temperature_schedule": parse_float_list(args.arm_b_temperature_schedule),
        "hypothesis_temperatures": hypothesis_temperatures,
        "records_path": str(records_path),
        "mean_pairwise_jaccard_at_200": round(mean_jaccard, 6),
        "jaccard_warning": mean_jaccard > args.arm_b_jaccard_threshold,
        "jaccard_warning_threshold": args.arm_b_jaccard_threshold,
        "warning_message": (
            "Arm B free temperature diversity collapsed; report this as a finding, "
            "but do not use it to gate B_prime or C."
        )
        if mean_jaccard > args.arm_b_jaccard_threshold
        else None,
        "description": (
            "Arm B uses the one-pass prompt with a mixed per-hypothesis temperature schedule. "
            "Per-hypothesis temperatures are logged in generation_temperature."
        ),
    }


def run_arm_b_records(
    args: argparse.Namespace,
    examples: list[Example],
    retriever: TaxonomyRetriever,
    generator: QueryGenerator | None,
    depths: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records_path = args.output_dir / "arm_B_temperature_mixture_hypothesis_candidates.jsonl"
    diagnostic_path = args.output_dir / "arm_B_diagnostics.json"
    hypothesis_temperatures = expand_temperature_schedule(
        parse_float_list(args.arm_b_temperature_schedule),
        args.budget,
    )
    records = run_one_prompt_arm(
        args,
        arm="B",
        examples=examples,
        retriever=retriever,
        generator=generator,
        output_path=records_path,
        hypotheses_per_fact=args.budget,
        temperature=hypothesis_temperatures[0],
        prompt_kind="one_pass",
        hypothesis_temperatures=hypothesis_temperatures,
    )
    diagnostic = arm_b_jaccard_diagnostic(
        args,
        records,
        examples,
        depths,
        hypothesis_temperatures,
        records_path,
    )
    diagnostic_path.write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if diagnostic["jaccard_warning"]:
        print(f"WARNING: {diagnostic['warning_message']} mean_jaccard@200={diagnostic['mean_pairwise_jaccard_at_200']}")
    return records, diagnostic


def main() -> None:
    args = parse_args()
    depths = parse_depths(args.depths, args.top_k)
    args.max_new_tokens = max(args.max_new_tokens, args.arm_c_max_new_tokens)
    if args.budget != 4:
        raise ValueError("This pilot currently expects --budget=4 to match the AGS coverage spec")
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows, examples, sample_summary = prepare_sample(args)
    examples_by_id = {example.example_idx: example for example in examples}
    print(json.dumps({"sample": sample_summary}, ensure_ascii=False, indent=2, sort_keys=True))

    taxonomy = load_taxonomy(args.taxonomy_jsonl)
    taxonomy_by_tag = {concept.tag: concept for concept in taxonomy}
    retriever = TaxonomyRetriever(taxonomy, type_filter=args.type_filter)
    probe_records = run_probe(
        args,
        examples,
        retriever,
        taxonomy_by_tag,
        args.output_dir / "probe_candidates.jsonl",
    )
    probe_rows = aggregate_probe_metrics(probe_records, examples, depths)
    write_csv(args.output_dir / "probe_metrics.csv", probe_rows)
    probe_check = build_probe_check(args, probe_rows)
    print(
        json.dumps(
            {"probe_first_check": {"rows": probe_rows, "check": probe_check}},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if not probe_check["passed"]:
        write_abort_metrics(
            args,
            sample_summary,
            depths,
            probe_rows,
            probe_check,
            abort_stage="probe",
            probe_records=probe_records,
            examples_by_id=examples_by_id,
        )
        raise RuntimeError(
            "Aborting AGS coverage pilot: probe R@200 failed the hard index-ceiling check. "
            f"Failures: {probe_check['failures']}"
        )

    generator = None if args.dry_run_no_llm else QueryGenerator(args)
    try:
        records_a = run_one_prompt_arm(
            args,
            arm="A",
            examples=examples,
            retriever=retriever,
            generator=generator,
            output_path=args.output_dir / "arm_A_hypothesis_candidates.jsonl",
            hypotheses_per_fact=1,
            temperature=args.arm_a_temperature,
            prompt_kind="one_pass",
        )
        arm_a_grouped = records_by_arm_fact(records_a)
        arm_a_metrics = compute_fact_metrics(arm_a_grouped, examples, depths, args.rrf_kappa)
        arm_a_report_rows = aggregate_arm_metrics(arm_a_metrics, examples, depths)
        write_csv(args.output_dir / "report_table.csv", arm_a_report_rows)
        arm_a_check = build_pre_interpretation_checks(args, probe_rows, arm_a_report_rows)
        print(json.dumps({"arm_a_second_check": arm_a_check}, ensure_ascii=False, indent=2, sort_keys=True))
        if not arm_a_check["passed"]:
            write_abort_metrics(
                args,
                sample_summary,
                depths,
                probe_rows,
                arm_a_check,
                abort_stage="arm_a_reference",
                probe_records=probe_records,
                examples_by_id=examples_by_id,
                report_rows=arm_a_report_rows,
                arm_records=records_a,
            )
            raise RuntimeError(
                "Aborting AGS coverage pilot: Arm A diverged from the one-pass R@200 reference. "
                f"Failures: {arm_a_check['failures']}"
            )

        records_b, b_diagnostic = run_arm_b_records(args, examples, retriever, generator, depths)
        records_bprime = run_one_prompt_arm(
            args,
            arm="B_prime",
            examples=examples,
            retriever=retriever,
            generator=generator,
            output_path=args.output_dir / "arm_B_prime_hypothesis_candidates.jsonl",
            hypotheses_per_fact=args.budget,
            temperature=args.arm_bprime_temperature,
            prompt_kind="diversity",
        )
        records_c = run_dimension_arm(
            args,
            examples=examples,
            retriever=retriever,
            generator=generator,
            output_path=args.output_dir / "arm_C_hypothesis_candidates.jsonl",
        )
    finally:
        if generator is not None:
            generator.close()

    all_records = records_a + records_b + records_bprime + records_c
    write_jsonl(args.output_dir / "all_hypothesis_candidates.jsonl", all_records)

    grouped = records_by_arm_fact(all_records)
    fact_metrics = compute_fact_metrics(grouped, examples, depths, args.rrf_kappa)
    report_rows = aggregate_arm_metrics(fact_metrics, examples, depths)
    marginal_rows = aggregate_marginal_coverage(fact_metrics, examples, depths)
    paired_rows = paired_difference_rows(
        fact_metrics,
        examples,
        depths,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    pre_interpretation_checks = build_pre_interpretation_checks(args, probe_rows, report_rows)
    warnings = []
    if b_diagnostic.get("jaccard_warning"):
        warnings.append(
            {
                "check": "arm_b_temperature_diversity",
                "message": b_diagnostic.get("warning_message"),
                "observed_mean_pairwise_jaccard_at_200": b_diagnostic.get("mean_pairwise_jaccard_at_200"),
                "threshold": b_diagnostic.get("jaccard_warning_threshold"),
            }
        )

    write_csv(args.output_dir / "report_table.csv", report_rows)
    write_csv(args.output_dir / "marginal_coverage.csv", marginal_rows)
    write_csv(args.output_dir / "paired_differences.csv", paired_rows)

    metrics = {
        "aborted": False,
        "sample_summary": sample_summary,
        "pre_interpretation_checks": pre_interpretation_checks,
        "warnings": warnings,
        "arm_b_temperature_diagnostic": b_diagnostic,
        "depths": depths,
        "top_k": args.top_k,
        "budget": args.budget,
        "rrf_kappa": args.rrf_kappa,
        "arms": report_rows,
        "marginal_coverage": marginal_rows,
        "paired_differences": paired_rows,
        "probe": probe_rows,
        "decision_quantity_note": "controller_room is C coverage@4 minus B_prime coverage@4, per pilot approval.",
        "arm_a_reference_check_note": (
            "The Arm A reference gate is only a gross-misconfiguration check. Passing it does not validate "
            "the setup or support any effect claim; divergence from 0.69 is expected because this pilot uses "
            "a different split and sample."
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"metrics_path": str(args.output_dir / "metrics.json")}, indent=2))


if __name__ == "__main__":
    main()
