#!/usr/bin/env python3
"""Validate full CodiEsp relocation outputs before launching GPU sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_nonempty_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts-jsonl", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--relocations-jsonl", type=Path, required=True)
    parser.add_argument("--spotcheck-tsv", type=Path, required=True)
    parser.add_argument("--docs-txt", type=Path, required=True)
    parser.add_argument("--expected-facts", type=int, default=3431)
    parser.add_argument("--expected-docs", type=int, default=250)
    parser.add_argument("--min-exact-rate", type=float, default=0.95)
    parser.add_argument("--min-parse-rate", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [
        args.facts_jsonl,
        args.stats_json,
        args.relocations_jsonl,
        args.spotcheck_tsv,
        args.docs_txt,
    ]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise SystemExit("Missing or empty full relocation output(s): " + ", ".join(missing))

    stats = json.loads(args.stats_json.read_text(encoding="utf-8"))
    fact_rows = count_nonempty_lines(args.facts_jsonl)
    relocation_rows = count_nonempty_lines(args.relocations_jsonl)
    spotcheck_rows = count_nonempty_lines(args.spotcheck_tsv)
    doc_rows = count_nonempty_lines(args.docs_txt)

    stats_facts = int(stats.get("target_facts", -1))
    if stats_facts != args.expected_facts or fact_rows != args.expected_facts or relocation_rows != args.expected_facts:
        raise SystemExit(
            f"Expected {args.expected_facts} full facts/relocations; "
            f"stats={stats_facts} facts_rows={fact_rows} relocation_rows={relocation_rows}"
        )

    stats_docs = int(stats.get("selected_documents", -1))
    if stats_docs != args.expected_docs or doc_rows != args.expected_docs:
        raise SystemExit(f"Expected {args.expected_docs} full test documents; stats={stats_docs} docs_rows={doc_rows}")

    if spotcheck_rows < 51:
        raise SystemExit(f"Spotcheck TSV has too few rows: {spotcheck_rows}")

    exact_rate = float(stats.get("relocation_exact_substring_rate", 0.0))
    parse_rate = float(stats.get("relocation_parse_ok_rate", 0.0))
    counts = stats.get("relocation_counts", {})
    fallback_document = int(counts.get("fallback_document", 0))
    if exact_rate < args.min_exact_rate or parse_rate < args.min_parse_rate or fallback_document:
        raise SystemExit(
            "Relocation quality gate failed: "
            f"exact_rate={exact_rate} parse_rate={parse_rate} fallback_document={fallback_document}"
        )

    print("Full relocation outputs passed validation:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
