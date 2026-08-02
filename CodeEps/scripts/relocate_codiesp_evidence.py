#!/usr/bin/env python3
"""Relocate CodiEsp Spanish evidence spans into English MT text.

The preparation script can structurally filter CodiEsp facts, but the spec asks for
English evidence spans because retrieval is English.  This script runs that alignment
step as an auditable, resumable pass:

1. Build diagnosis facts from the official test split at document level.
2. Ask the selected LLM for an exact English substring copied from the candidate
   English text.
3. Materialize a grounding JSONL using the verified substring when possible and a
   logged fallback otherwise.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = ROOT / "codiesp_pipeline"
SHARED_RUNNER = PIPELINE_ROOT / "run_fintagging_grounding_baseline.py"
DATA_DIR = ROOT / "data" / "codiesp"
INDEX_JSONL = ROOT / "index" / "icd10cm_fy2018" / "icd10cm_fy2018_retrieval.jsonl"

sys.path.insert(0, str(ROOT / "scripts"))
import prepare_codiesp_domain as prep  # noqa: E402


def load_shared_runner() -> Any:
    sys.path.insert(0, str(PIPELINE_ROOT))
    spec = importlib.util.spec_from_file_location("run_fintagging_grounding_baseline", SHARED_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import shared runner from {SHARED_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_fintagging_grounding_baseline"] = module
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def inventory_display_by_raw() -> dict[str, str]:
    return {
        prep.normalize_code(row["tag"]): row["tag"]
        for row in load_jsonl(INDEX_JSONL)
    }


def selected_rows(target_facts: int | None, seed: int) -> list[dict[str, str]]:
    display_by_raw = inventory_display_by_raw()
    rows = [
        row
        for row in prep.load_x_rows("test")
        if row["label"] == "DIAGNOSTICO" and row["raw_code"] in display_by_raw
    ]
    by_doc: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["article_id"], row["position"], row["raw_code"])
        if key in seen:
            continue
        seen.add(key)
        by_doc[row["article_id"]].append(row)

    doc_ids = sorted(by_doc)
    if target_facts is not None and target_facts > 0:
        import random

        rng = random.Random(seed)
        rng.shuffle(doc_ids)
        selected_docs: list[str] = []
        fact_total = 0
        for doc_id in doc_ids:
            selected_docs.append(doc_id)
            fact_total += len(by_doc[doc_id])
            if fact_total >= target_facts:
                break
        doc_ids = sorted(selected_docs)

    selected: list[dict[str, str]] = []
    for doc_id in doc_ids:
        selected.extend(by_doc[doc_id])
    return selected


def target_records(target_facts: int | None, seed: int) -> list[dict[str, Any]]:
    display_by_raw = inventory_display_by_raw()
    records: list[dict[str, Any]] = []
    for relocation_id, row in enumerate(selected_rows(target_facts, seed)):
        locus = prep.locate_english_locus(row["article_id"], row["position"], row["reference"], "test")
        candidate_text = locus["english_locus"] if locus["locus_level"] == "aligned_sentence" else locus["english_context"]
        candidate_scope = "aligned_sentence" if locus["locus_level"] == "aligned_sentence" else "document"
        records.append(
            {
                "relocation_id": relocation_id,
                "article_id": row["article_id"],
                "label": row["label"],
                "raw_code": row["raw_code"],
                "display_code": display_by_raw[row["raw_code"]],
                "reference": row["reference"],
                "position": row["position"],
                "candidate_scope": candidate_scope,
                "candidate_text": normalize_space(candidate_text),
                "alignment": locus,
            }
        )
    return records


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    user = f"""Align one Spanish clinical evidence mention to the English machine-translated text.

Return JSON only with this schema:
{{"substring": "exact English substring copied from the English text"}}

Rules:
- The substring must be copied verbatim from the English text below.
- Prefer the shortest clinically meaningful phrase matching the Spanish evidence.
- If no exact English substring can be identified, return {{"substring": null}}.
- Do not include explanations or markdown.

Spanish evidence:
{record["reference"]}

English text:
{record["candidate_text"]}"""
    return [
        {"role": "system", "content": "You align Spanish clinical mentions to exact English text spans."},
        {"role": "user", "content": user},
    ]


def parse_substring(raw_output: str, runner: Any) -> tuple[str, bool]:
    parsed, parse_ok = runner.parse_json_object(raw_output)
    value = parsed.get("substring")
    if value is None:
        return "", parse_ok
    return normalize_space(value), parse_ok and isinstance(value, str)


def validate_substring(substring: str, candidate_text: str) -> tuple[str, bool]:
    substring = normalize_space(substring)
    candidate_text = normalize_space(candidate_text)
    if substring and substring in candidate_text:
        return substring, True
    return "", False


def load_existing_relocations(path: Path) -> dict[int, dict[str, Any]]:
    return {
        int(row["relocation_id"]): row
        for row in load_jsonl(path)
        if "relocation_id" in row
    }


def run_relocation(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runner = load_shared_runner()
    existing = load_existing_relocations(args.relocations_jsonl) if args.resume else {}
    pending = [record for record in records if int(record["relocation_id"]) not in existing]
    if not pending:
        return [existing[int(record["relocation_id"])] for record in records]

    if args.backend != "vllm":
        raise SystemExit("Only --backend vllm is currently implemented for relocation.")

    from vllm import SamplingParams

    tokenizer, llm = runner.load_vllm_engine(args, args.model)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    mode = "a" if args.resume and args.relocations_jsonl.exists() else "w"
    with args.relocations_jsonl.open(mode, encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            prompts = [runner.messages_to_prompt(tokenizer, build_messages(record)) for record in batch]
            outputs = llm.generate(prompts, sampling_params)
            for record, output in zip(batch, outputs, strict=True):
                raw_output = output.outputs[0].text.strip() if output.outputs else ""
                parsed_substring, parse_ok = parse_substring(raw_output, runner)
                substring, substring_found = validate_substring(parsed_substring, record["candidate_text"])
                out = {
                    **record,
                    "raw_output": raw_output,
                    "parse_ok": parse_ok,
                    "parsed_substring": parsed_substring,
                    "selected_substring": substring,
                    "substring_found": substring_found,
                    "backend": args.backend,
                    "model": args.model,
                }
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
                existing[int(record["relocation_id"])] = out
            handle.flush()
            processed = min(start + len(batch), len(pending))
            if processed % args.log_every == 0 or processed == len(pending):
                print(f"Relocated {processed}/{len(pending)} pending evidence spans", flush=True)

    runner.release_model_handles(llm, tokenizer)
    return [existing[int(record["relocation_id"])] for record in records]


def materialize_facts(records: list[dict[str, Any]], output_jsonl: Path, stats_json: Path, docs_txt: Path) -> None:
    facts: list[dict[str, Any]] = []
    diagnostics = Counter()
    selected_docs = sorted({record["article_id"] for record in records})
    for record in records:
        alignment = dict(record["alignment"])
        substring_found = bool(record.get("substring_found"))
        entity = normalize_space(record.get("selected_substring") if substring_found else record["candidate_text"])
        locus_level = "exact_substring" if substring_found else f"fallback_{record['candidate_scope']}"
        alignment.update(
            {
                "relocation_candidate_scope": record["candidate_scope"],
                "relocation_parse_ok": bool(record.get("parse_ok")),
                "relocation_substring_found": substring_found,
                "relocation_selected_substring": normalize_space(record.get("selected_substring", "")),
                "relocation_locus_level": locus_level,
            }
        )
        diagnostics["facts"] += 1
        diagnostics[f"relocation_{locus_level}"] += 1
        diagnostics["parse_ok"] += int(bool(record.get("parse_ok")))
        diagnostics["substring_found"] += int(substring_found)
        diagnostics[f"candidate_scope_{record['candidate_scope']}"] += 1

        input_fields = {
            "entity": entity,
            "type": "diagnosis",
            "row_context": "",
            "column_context": "",
            "original_context": normalize_space(record["candidate_text"]),
            "document_context": normalize_space(alignment["english_context"]),
            "spanish_reference_text": record["reference"],
            "spanish_offset_text": alignment["spanish_offset_text"],
            "located_english_span": entity,
            "codiesp_position": record["position"],
            "document_id": record["article_id"],
        }
        facts.append(
            {
                "context_id": record["article_id"],
                "ground_truth_concepts": [record["display_code"]],
                "ground_truth_count": 1,
                "input": json.dumps(input_fields, ensure_ascii=False),
                "input_fields": input_fields,
                "input_type": "text",
                "output": json.dumps([record["display_code"]]),
                "source_sample_idx": record["article_id"],
                "split": "test",
                "codiesp_label": record["label"],
                "alignment": alignment,
            }
        )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    docs_txt.write_text("\n".join(selected_docs) + "\n", encoding="utf-8")
    write_jsonl(output_jsonl, facts)
    stats = {
        "selected_documents": len(selected_docs),
        "source_contexts": len(selected_docs),
        "target_facts": len(facts),
        "unique_gold_concepts": len({fact["ground_truth_concepts"][0] for fact in facts}),
        "facts_per_context": round(len(facts) / len(selected_docs), 4) if selected_docs else 0.0,
        "relocation_parse_ok_rate": round(diagnostics["parse_ok"] / diagnostics["facts"], 6) if diagnostics["facts"] else 0.0,
        "relocation_exact_substring_rate": round(diagnostics["substring_found"] / diagnostics["facts"], 6) if diagnostics["facts"] else 0.0,
        "relocation_counts": {
            key.replace("relocation_", ""): value
            for key, value in diagnostics.items()
            if key.startswith("relocation_")
        },
        "candidate_scope_counts": {
            key.replace("candidate_scope_", ""): value
            for key, value in diagnostics.items()
            if key.startswith("candidate_scope_")
        },
    }
    stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_spotcheck(facts_jsonl: Path, spotcheck_tsv: Path, seed: int, sample_size: int = 50) -> None:
    import random

    facts = load_jsonl(facts_jsonl)
    rng = random.Random(seed)
    rng.shuffle(facts)
    rows = facts[:sample_size]
    label_by_code = {row["tag"]: row["standard_label"] for row in load_jsonl(INDEX_JSONL)}
    with spotcheck_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "document_id",
                "spanish_original",
                "located_english_span",
                "english_locus",
                "gold_code",
                "gold_description",
                "relocation_locus_level",
                "candidate_scope",
                "parse_ok",
            ]
        )
        for fact in rows:
            alignment = fact["alignment"]
            code = fact["ground_truth_concepts"][0]
            writer.writerow(
                [
                    fact["context_id"],
                    alignment["spanish_reference_text"],
                    alignment["relocation_selected_substring"] or fact["input_fields"]["located_english_span"],
                    fact["input_fields"]["original_context"],
                    code,
                    label_by_code.get(code, ""),
                    alignment["relocation_locus_level"],
                    alignment["relocation_candidate_scope"],
                    alignment["relocation_parse_ok"],
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-facts", type=int, default=0, help="<=0 means full CodiEsp test diagnosis set.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--relocations-jsonl", type=Path, default=DATA_DIR / "evidence_relocations_full.jsonl")
    parser.add_argument("--facts-jsonl", type=Path, default=DATA_DIR / "facts_test_full.jsonl")
    parser.add_argument("--stats-json", type=Path, default=DATA_DIR / "stats_full.json")
    parser.add_argument("--docs-txt", type=Path, default=DATA_DIR / "test_docs_full.txt")
    parser.add_argument("--spotcheck-tsv", type=Path, default=DATA_DIR / "spotcheck_50_full.tsv")
    parser.add_argument("--emit-requests-only", action="store_true")
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", choices=["vllm"], default="vllm")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=12000)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--query-max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = None if args.target_facts <= 0 else args.target_facts
    records = target_records(target, args.seed)
    if args.limit is not None:
        records = records[: args.limit]
    print(f"Prepared {len(records)} relocation target(s)", flush=True)

    if args.emit_requests_only:
        request_path = args.relocations_jsonl.with_suffix(".requests.jsonl")
        write_jsonl(request_path, records)
        print(f"Wrote relocation requests: {request_path}", flush=True)
        return

    if args.materialize_only:
        relocations = load_existing_relocations(args.relocations_jsonl)
        missing = [record["relocation_id"] for record in records if int(record["relocation_id"]) not in relocations]
        if missing:
            raise SystemExit(f"Missing {len(missing)} relocation record(s); first missing ids: {missing[:10]}")
        final_records = [relocations[int(record["relocation_id"])] for record in records]
    else:
        final_records = run_relocation(args, records)

    materialize_facts(final_records, args.facts_jsonl, args.stats_json, args.docs_txt)
    write_spotcheck(args.facts_jsonl, args.spotcheck_tsv, args.seed)
    print(f"Wrote facts: {args.facts_jsonl}", flush=True)
    print(f"Wrote stats: {args.stats_json}", flush=True)
    print(f"Wrote spotcheck: {args.spotcheck_tsv}", flush=True)


if __name__ == "__main__":
    main()
