#!/usr/bin/env python3
"""Self-retrieval sanity probe for the generated ICD-10-CM index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = ROOT / "codiesp_pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from run_fintagging_grounding_baseline import TaxonomyRetriever, load_taxonomy, normalize_tag  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--taxonomy-jsonl",
        type=Path,
        default=ROOT / "index" / "icd10cm_fy2018" / "icd10cm_fy2018_retrieval.jsonl",
    )
    parser.add_argument(
        "--gold-jsonl",
        type=Path,
        default=None,
        help="Optional grounding JSONL. If set, probe only concepts used as gold labels here.",
    )
    parser.add_argument("--sample", type=int, default=0, help="0 means all concepts.")
    parser.add_argument("--output", type=Path, default=ROOT / "index" / "icd10cm_fy2018" / "self_retrieval_probe.json")
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.taxonomy_jsonl)
    concept_by_tag = {normalize_tag(concept.tag): concept for concept in taxonomy}
    if args.gold_jsonl:
        target_tags = []
        with args.gold_jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    target_tags.extend(normalize_tag(tag) for tag in row.get("ground_truth_concepts", []))
        probe_concepts = [concept_by_tag[tag] for tag in sorted(set(target_tags)) if tag in concept_by_tag]
    elif args.sample > 0:
        probe_concepts = taxonomy[: args.sample]
    else:
        probe_concepts = taxonomy
    retriever = TaxonomyRetriever(taxonomy, type_filter=True, label_coverage_weight=1.0, label_coverage_pool_multiplier=0)
    failures = []
    for concept in probe_concepts:
        query = " ".join(part for part in [concept.standard_label, concept.documentation] if part)
        ranked = retriever.retrieve(query, concept.entity_type, 1)
        top_tag = normalize_tag(ranked[0][0].tag) if ranked else ""
        if top_tag != normalize_tag(concept.tag):
            failures.append(
                {
                    "tag": concept.tag,
                    "standard_label": concept.standard_label,
                    "top_tag": top_tag,
                    "top_label": ranked[0][0].standard_label if ranked else "",
                }
            )
    report = {
        "checked": len(probe_concepts),
        "top1_failures": len(failures),
        "top1_failure_rate": round(len(failures) / len(probe_concepts), 6) if probe_concepts else 0.0,
        "failed": len(failures) > max(1, int(0.05 * len(probe_concepts))),
        "first_failures": failures[:50],
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
