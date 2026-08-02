#!/usr/bin/env python3
"""Create a labeled FinTagging train/test subset with covered test labels.

The split uses only samples with at least one numeric entity. It stratifies by
the rarest datatype present in each sample, which keeps rare datatypes visible
while still allowing the train/test datatype mix to stay similar.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATA_PATH = "FinTagging_Original/data/test-00000-of-00001.parquet"
DEFAULT_OUTPUT_DIR = "FinTagging_Original_800_200_split"


@dataclass(frozen=True)
class SampleInfo:
    sample_idx: int
    context_id: int
    concepts: frozenset[str]
    datatypes: frozenset[str]
    datatype_counts: Counter[str]
    stratum: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample FinTagging train/test subsets from labeled rows only. "
            "The test split is restricted to concepts and datatypes already "
            "present in the train split."
        )
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Input FinTagging parquet file. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for train/test files and report. Default: %(default)s",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=800,
        help="Target number of train samples. Default: %(default)s",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=200,
        help="Target number of test samples. Default: %(default)s",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed. Default: %(default)s",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=200,
        help="Number of candidate splits to try before choosing the best. Default: %(default)s",
    )
    parser.add_argument(
        "--max-stratum-pct-diff",
        type=float,
        default=0.05,
        help=(
            "Maximum allowed absolute train/test percentage-point difference "
            "for rarest-datatype strata. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected split report without writing output files.",
    )
    return parser.parse_args()


def iter_numeric_entities(value: object) -> Iterable[dict[str, Any]]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        for entity in value:
            if isinstance(entity, dict):
                yield entity


def build_labeled_samples(df: pd.DataFrame) -> list[SampleInfo]:
    sample_rows: list[tuple[int, int, frozenset[str], Counter[str]]] = []
    datatype_presence_counts: Counter[str] = Counter()

    for sample_idx, row in df.iterrows():
        concepts: set[str] = set()
        datatype_counts: Counter[str] = Counter()

        for entity in iter_numeric_entities(row["numeric_entities"]):
            concept = entity.get("concept")
            datatype = entity.get("type")
            if isinstance(concept, str) and concept:
                concepts.add(concept)
            if isinstance(datatype, str) and datatype:
                datatype_counts[datatype] += 1

        if not concepts or not datatype_counts:
            continue

        datatypes = frozenset(datatype_counts)
        datatype_presence_counts.update(datatypes)
        context_id = int(row["context_id"]) if "context_id" in row else int(sample_idx)
        sample_rows.append((int(sample_idx), context_id, frozenset(concepts), datatype_counts))

    samples: list[SampleInfo] = []
    for sample_idx, context_id, concepts, datatype_counts in sample_rows:
        datatypes = frozenset(datatype_counts)
        stratum = min(datatypes, key=lambda item: (datatype_presence_counts[item], item))
        samples.append(
            SampleInfo(
                sample_idx=sample_idx,
                context_id=context_id,
                concepts=concepts,
                datatypes=datatypes,
                datatype_counts=datatype_counts,
                stratum=stratum,
            )
        )

    return samples


def group_by_stratum(samples: Iterable[SampleInfo]) -> dict[str, list[SampleInfo]]:
    groups: dict[str, list[SampleInfo]] = defaultdict(list)
    for sample in samples:
        groups[sample.stratum].append(sample)
    return dict(groups)


def allocate_quotas(
    weights: dict[str, int],
    target_total: int,
    caps: dict[str, int] | None = None,
    min_per_group: bool = True,
) -> dict[str, int]:
    """Allocate an integer quota by largest proportional deficit."""
    if target_total < 0:
        raise ValueError("target_total must be non-negative")

    groups = sorted(group for group, weight in weights.items() if weight > 0)
    caps = {group: max(0, int((caps or weights).get(group, 0))) for group in groups}
    target_total = min(target_total, sum(caps.values()))
    quotas = {group: 0 for group in groups}

    positive_cap_groups = [group for group in groups if caps[group] > 0]
    if min_per_group and target_total >= len(positive_cap_groups):
        for group in positive_cap_groups:
            quotas[group] = 1

    weight_total = sum(weights[group] for group in groups if caps[group] > 0)
    if weight_total == 0:
        return quotas

    raw_targets = {
        group: (weights[group] / weight_total) * target_total if caps[group] > 0 else 0.0
        for group in groups
    }

    while sum(quotas.values()) < target_total:
        candidates = [group for group in groups if quotas[group] < caps[group]]
        if not candidates:
            break
        best_group = max(
            candidates,
            key=lambda group: (raw_targets[group] - quotas[group], weights[group], group),
        )
        quotas[best_group] += 1

    return quotas


def stratified_sample(
    samples: Iterable[SampleInfo],
    quotas: dict[str, int],
    rng: random.Random,
) -> list[SampleInfo]:
    groups = group_by_stratum(samples)
    selected: list[SampleInfo] = []
    for stratum, quota in quotas.items():
        if quota <= 0:
            continue
        available = groups.get(stratum, [])
        if quota > len(available):
            raise ValueError(f"Quota {quota} exceeds available {len(available)} for {stratum}")
        selected.extend(rng.sample(available, quota))
    return selected


def summarize(samples: Iterable[SampleInfo]) -> dict[str, Any]:
    sample_list = list(samples)
    concept_set: set[str] = set()
    datatype_set: set[str] = set()
    datatype_entity_counts: Counter[str] = Counter()
    datatype_sample_presence_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()

    for sample in sample_list:
        concept_set.update(sample.concepts)
        datatype_set.update(sample.datatypes)
        datatype_entity_counts.update(sample.datatype_counts)
        datatype_sample_presence_counts.update(sample.datatypes)
        stratum_counts[sample.stratum] += 1

    return {
        "sample_count": len(sample_list),
        "entity_count": int(sum(datatype_entity_counts.values())),
        "unique_concept_count": len(concept_set),
        "unique_datatype_count": len(datatype_set),
        "datatype_entity_counts": dict(sorted(datatype_entity_counts.items())),
        "datatype_sample_presence_counts": dict(sorted(datatype_sample_presence_counts.items())),
        "rarest_datatype_stratum_counts": dict(sorted(stratum_counts.items())),
    }


def proportions(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {key: 0.0 for key in counts}
    return {key: value / total for key, value in counts.items()}


def max_abs_prop_diff(left: dict[str, int], right: dict[str, int]) -> float:
    left_props = proportions(left)
    right_props = proportions(right)
    keys = set(left_props) | set(right_props)
    return max((abs(left_props.get(key, 0.0) - right_props.get(key, 0.0)) for key in keys), default=0.0)


def validate_split(
    train_samples: list[SampleInfo],
    test_samples: list[SampleInfo],
    max_stratum_pct_diff: float,
) -> dict[str, Any]:
    train_ids = {sample.sample_idx for sample in train_samples}
    test_ids = {sample.sample_idx for sample in test_samples}
    train_context_ids = {sample.context_id for sample in train_samples}
    test_context_ids = {sample.context_id for sample in test_samples}

    train_concepts = set().union(*(sample.concepts for sample in train_samples))
    test_concepts = set().union(*(sample.concepts for sample in test_samples))
    train_datatypes = set().union(*(sample.datatypes for sample in train_samples))
    test_datatypes = set().union(*(sample.datatypes for sample in test_samples))

    train_summary = summarize(train_samples)
    test_summary = summarize(test_samples)
    stratum_pct_diff = max_abs_prop_diff(
        train_summary["rarest_datatype_stratum_counts"],
        test_summary["rarest_datatype_stratum_counts"],
    )

    validation = {
        "sample_idx_overlap_count": len(train_ids & test_ids),
        "context_id_overlap_count": len(train_context_ids & test_context_ids),
        "missing_test_concepts_in_train_count": len(test_concepts - train_concepts),
        "missing_test_datatypes_in_train_count": len(test_datatypes - train_datatypes),
        "max_rarest_datatype_stratum_pct_diff": stratum_pct_diff,
        "max_allowed_rarest_datatype_stratum_pct_diff": max_stratum_pct_diff,
    }
    validation["passed"] = (
        validation["sample_idx_overlap_count"] == 0
        and validation["context_id_overlap_count"] == 0
        and validation["missing_test_concepts_in_train_count"] == 0
        and validation["missing_test_datatypes_in_train_count"] == 0
        and stratum_pct_diff <= max_stratum_pct_diff
    )
    return validation


def candidate_score(
    train_samples: list[SampleInfo],
    test_samples: list[SampleInfo],
    train_target: int,
    test_target: int,
) -> float:
    train_summary = summarize(train_samples)
    test_summary = summarize(test_samples)
    train_size_penalty = abs(len(train_samples) - train_target) / max(train_target, 1)
    test_size_penalty = abs(len(test_samples) - test_target) / max(test_target, 1)
    stratum_diff = max_abs_prop_diff(
        train_summary["rarest_datatype_stratum_counts"],
        test_summary["rarest_datatype_stratum_counts"],
    )
    sample_presence_diff = max_abs_prop_diff(
        train_summary["datatype_sample_presence_counts"],
        test_summary["datatype_sample_presence_counts"],
    )
    entity_type_diff = max_abs_prop_diff(
        train_summary["datatype_entity_counts"],
        test_summary["datatype_entity_counts"],
    )
    return (
        10.0 * test_size_penalty
        + 5.0 * train_size_penalty
        + 4.0 * stratum_diff
        + 2.0 * sample_presence_diff
        + entity_type_diff
    )


def choose_split(
    samples: list[SampleInfo],
    train_size: int,
    test_size: int,
    seed: int,
    max_attempts: int,
    max_stratum_pct_diff: float,
) -> tuple[list[SampleInfo], list[SampleInfo], dict[str, Any]]:
    stratum_sizes = {stratum: len(group) for stratum, group in group_by_stratum(samples).items()}
    train_quotas = allocate_quotas(stratum_sizes, train_size)

    best: tuple[
        float,
        int,
        list[SampleInfo],
        list[SampleInfo],
        dict[str, Any],
        dict[str, int],
        int,
    ] | None = None

    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt)
        train_samples = stratified_sample(samples, train_quotas, rng)
        train_ids = {sample.sample_idx for sample in train_samples}
        train_concepts = set().union(*(sample.concepts for sample in train_samples))
        train_datatypes = set().union(*(sample.datatypes for sample in train_samples))

        eligible_test_samples = [
            sample
            for sample in samples
            if sample.sample_idx not in train_ids
            and sample.concepts.issubset(train_concepts)
            and sample.datatypes.issubset(train_datatypes)
        ]
        eligible_test_caps = {
            stratum: len(group) for stratum, group in group_by_stratum(eligible_test_samples).items()
        }
        test_quotas = allocate_quotas(stratum_sizes, test_size, caps=eligible_test_caps)
        test_samples = stratified_sample(eligible_test_samples, test_quotas, rng)
        validation = validate_split(train_samples, test_samples, max_stratum_pct_diff)
        score = candidate_score(train_samples, test_samples, train_size, test_size)

        candidate = (
            score,
            attempt,
            train_samples,
            test_samples,
            validation,
            test_quotas,
            len(eligible_test_samples),
        )
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    if best is None:
        raise RuntimeError("Could not produce any candidate split")

    _, best_attempt, train_samples, test_samples, validation, test_quotas, eligible_test_count = best
    validation = validate_split(train_samples, test_samples, max_stratum_pct_diff)
    if not validation["passed"]:
        raise RuntimeError(f"Best split failed validation: {json.dumps(validation, indent=2)}")

    metadata = {
        "seed": seed,
        "selected_attempt": best_attempt,
        "requested_train_size": train_size,
        "requested_test_size": test_size,
        "train_quotas": dict(sorted(train_quotas.items())),
        "test_quotas": dict(sorted(test_quotas.items())),
        "test_eligible_sample_count": eligible_test_count,
        "validation": validation,
    }
    return train_samples, test_samples, metadata


def samples_to_frame(df: pd.DataFrame, samples: list[SampleInfo], split_name: str) -> pd.DataFrame:
    ordered = sorted(samples, key=lambda sample: sample.sample_idx)
    indices = [sample.sample_idx for sample in ordered]
    out = df.loc[indices].copy()
    out.insert(0, "source_sample_idx", indices)
    out.insert(1, "split", split_name)
    return out.reset_index(drop=True)


def write_outputs(
    df: pd.DataFrame,
    train_samples: list[SampleInfo],
    test_samples: list[SampleInfo],
    report: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = samples_to_frame(df, train_samples, "train")
    test_df = samples_to_frame(df, test_samples, "test")
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    pd.DataFrame(
        {
            "source_sample_idx": train_df["source_sample_idx"],
            "context_id": train_df["context_id"],
        }
    ).to_csv(output_dir / "train_ids.csv", index=False)
    pd.DataFrame(
        {
            "source_sample_idx": test_df["source_sample_idx"],
            "context_id": test_df["context_id"],
        }
    ).to_csv(output_dir / "test_ids.csv", index=False)

    with (output_dir / "split_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)

    df = pd.read_parquet(data_path)
    samples = build_labeled_samples(df)
    if not samples:
        raise RuntimeError("No labeled samples found")

    if args.train_size + args.test_size > len(samples):
        raise ValueError(
            f"Requested {args.train_size + args.test_size} samples, but only {len(samples)} labeled samples exist"
        )

    train_samples, test_samples, metadata = choose_split(
        samples=samples,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        max_attempts=args.max_attempts,
        max_stratum_pct_diff=args.max_stratum_pct_diff,
    )

    report = {
        "data_path": str(data_path),
        "labeled_sample_count": len(samples),
        "split_metadata": metadata,
        "train": summarize(train_samples),
        "test": summarize(test_samples),
    }

    if not args.dry_run:
        write_outputs(df, train_samples, test_samples, report, output_dir)

    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.dry_run:
        print(f"\nWrote split files to: {output_dir}")


if __name__ == "__main__":
    main()
