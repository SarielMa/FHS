#!/usr/bin/env python3
"""Compute train/test label statistics for a FinTagging split directory."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SPLIT_DIR = "FinTagging_Original_800_200_split"
DEFAULT_OUTPUT_DIR = "FinTagging_Original_800_200_split_stats"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze FinTagging train/test split labels. Datatype means "
            "numeric_entities[].type; tag means numeric_entities[].concept."
        )
    )
    parser.add_argument(
        "--split-dir",
        default=DEFAULT_SPLIT_DIR,
        help="Directory containing train.parquet and test.parquet. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV/JSON statistics. Default: %(default)s",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of most frequent datatypes/tags to print. Default: %(default)s",
    )
    return parser.parse_args()


def iter_numeric_entities(value: object) -> Iterable[dict[str, Any]]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        for entity in value:
            if isinstance(entity, dict):
                yield entity


def entity_count(value: object) -> int:
    return sum(1 for _ in iter_numeric_entities(value))


def load_split(split_dir: Path, split_name: str) -> pd.DataFrame:
    path = split_dir / f"{split_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_parquet(path)
    if "split" not in df.columns:
        df.insert(0, "split", split_name)
    if "source_sample_idx" not in df.columns:
        df.insert(0, "source_sample_idx", df.index)
    return df


def build_entity_table(split_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, df in split_frames.items():
        for row_idx, row in df.iterrows():
            source_sample_idx = int(row["source_sample_idx"])
            context_id = int(row["context_id"]) if "context_id" in row else source_sample_idx

            for entity_idx, entity in enumerate(iter_numeric_entities(row["numeric_entities"])):
                rows.append(
                    {
                        "split": split_name,
                        "row_idx": int(row_idx),
                        "source_sample_idx": source_sample_idx,
                        "context_id": context_id,
                        "entity_idx": entity_idx,
                        "concept": entity.get("concept"),
                        "type": entity.get("type"),
                        "value": entity.get("value"),
                    }
                )

    return pd.DataFrame(rows)


def count_with_pct(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    counts = (
        df.groupby(group_cols, dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(group_cols[:-1] + ["count"], ascending=[True] * (len(group_cols) - 1) + [False])
    )
    totals = counts.groupby(group_cols[:-1])["count"].transform("sum")
    counts["pct"] = counts["count"] / totals
    return counts.rename(columns={value_col: value_col})


def sample_presence_counts(entity_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    return (
        entity_df[["split", "source_sample_idx", label_col]]
        .drop_duplicates()
        .groupby(["split", label_col], dropna=False)
        .size()
        .rename("sample_presence_count")
        .reset_index()
        .sort_values(["split", "sample_presence_count"], ascending=[True, False])
    )


def split_summary(df: pd.DataFrame, entity_df: pd.DataFrame) -> dict[str, Any]:
    sample_entity_counts = df["numeric_entities"].map(entity_count)
    return {
        "sample_count": int(len(df)),
        "labeled_sample_count": int((sample_entity_counts > 0).sum()),
        "empty_sample_count": int((sample_entity_counts == 0).sum()),
        "entity_count": int(len(entity_df)),
        "unique_datatype_count": int(entity_df["type"].nunique(dropna=True)),
        "unique_tag_count": int(entity_df["concept"].nunique(dropna=True)),
        "datatype_entity_counts": {
            str(key): int(value)
            for key, value in entity_df["type"].value_counts(dropna=False).sort_index().items()
        },
        "datatype_sample_presence_counts": {
            str(key): int(value)
            for key, value in (
                entity_df[["source_sample_idx", "type"]]
                .drop_duplicates()
                ["type"]
                .value_counts(dropna=False)
                .sort_index()
                .items()
            )
        },
    }


def coverage_summary(split_frames: dict[str, pd.DataFrame], entity_df: pd.DataFrame) -> dict[str, Any]:
    train_entities = entity_df[entity_df["split"] == "train"]
    test_entities = entity_df[entity_df["split"] == "test"]
    train_ids = set(split_frames["train"]["source_sample_idx"])
    test_ids = set(split_frames["test"]["source_sample_idx"])
    train_context_ids = set(split_frames["train"]["context_id"])
    test_context_ids = set(split_frames["test"]["context_id"])
    train_concepts = set(train_entities["concept"].dropna())
    test_concepts = set(test_entities["concept"].dropna())
    train_types = set(train_entities["type"].dropna())
    test_types = set(test_entities["type"].dropna())

    return {
        "sample_idx_overlap_count": len(train_ids & test_ids),
        "context_id_overlap_count": len(train_context_ids & test_context_ids),
        "missing_test_concepts_in_train_count": len(test_concepts - train_concepts),
        "missing_test_datatypes_in_train_count": len(test_types - train_types),
        "train_only_concept_count": len(train_concepts - test_concepts),
        "shared_concept_count": len(train_concepts & test_concepts),
        "test_unique_concept_count": len(test_concepts),
        "train_unique_concept_count": len(train_concepts),
        "passed": (
            len(train_ids & test_ids) == 0
            and len(train_context_ids & test_context_ids) == 0
            and len(test_concepts - train_concepts) == 0
            and len(test_types - train_types) == 0
        ),
    }


def print_split_report(
    split_name: str,
    summary: dict[str, Any],
    datatype_counts: pd.DataFrame,
    tag_counts: pd.DataFrame,
    top_k: int,
) -> None:
    print(f"{split_name.upper()} split")
    print(f"Total samples: {summary['sample_count']:,}")
    print(f"Samples with at least one entity: {summary['labeled_sample_count']:,}")
    print(f"Samples with no entities: {summary['empty_sample_count']:,}")
    print(f"Total entity triples: {summary['entity_count']:,}")
    print(f"Unique datatypes: {summary['unique_datatype_count']:,}")
    print(f"Unique tags/concepts: {summary['unique_tag_count']:,}")
    print()

    print(f"Top {top_k} datatypes:")
    print(datatype_counts[datatype_counts["split"] == split_name].head(top_k).to_string(index=False))
    print()

    print(f"Top {top_k} tags/concepts:")
    print(tag_counts[tag_counts["split"] == split_name].head(top_k).to_string(index=False))
    print()


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)

    split_frames = {
        "train": load_split(split_dir, "train"),
        "test": load_split(split_dir, "test"),
    }
    entity_df = build_entity_table(split_frames)
    if entity_df.empty:
        raise RuntimeError("No numeric entities found in split files")

    output_dir.mkdir(parents=True, exist_ok=True)

    datatype_counts = count_with_pct(entity_df, ["split", "type"], "type")
    tag_counts = count_with_pct(entity_df, ["split", "concept"], "concept")
    datatype_sample_presence = sample_presence_counts(entity_df, "type")
    tag_sample_presence = sample_presence_counts(entity_df, "concept")

    summaries = {
        split_name: split_summary(df, entity_df[entity_df["split"] == split_name])
        for split_name, df in split_frames.items()
    }
    report = {
        "split_dir": str(split_dir),
        "coverage": coverage_summary(split_frames, entity_df),
        "splits": summaries,
    }

    datatype_counts.to_csv(output_dir / "datatype_counts_by_split.csv", index=False)
    tag_counts.to_csv(output_dir / "tag_counts_by_split.csv", index=False)
    datatype_sample_presence.to_csv(output_dir / "datatype_sample_presence_by_split.csv", index=False)
    tag_sample_presence.to_csv(output_dir / "tag_sample_presence_by_split.csv", index=False)
    entity_df.to_csv(output_dir / "entities_flat_by_split.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Split directory: {split_dir}")
    print()
    for split_name in ["train", "test"]:
        print_split_report(
            split_name=split_name,
            summary=summaries[split_name],
            datatype_counts=datatype_counts,
            tag_counts=tag_counts,
            top_k=args.top_k,
        )

    print("Coverage checks:")
    for key, value in report["coverage"].items():
        print(f"{key}: {value}")
    print()
    print(f"Wrote: {output_dir / 'summary.json'}")
    print(f"Wrote: {output_dir / 'datatype_counts_by_split.csv'}")
    print(f"Wrote: {output_dir / 'tag_counts_by_split.csv'}")
    print(f"Wrote: {output_dir / 'datatype_sample_presence_by_split.csv'}")
    print(f"Wrote: {output_dir / 'tag_sample_presence_by_split.csv'}")
    print(f"Wrote: {output_dir / 'entities_flat_by_split.csv'}")


if __name__ == "__main__":
    main()
