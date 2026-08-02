#!/usr/bin/env python3
"""Prepare the CodiEsp / ICD-10-CM second-domain grounding data.

This script intentionally writes only under second_domain.  It adapts the
downloaded CodiEsp and ICD-10-CM FY2018 artifacts into the narrow JSONL/index
interfaces consumed by codiesp_pipeline/run_fintagging_grounding_baseline.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
RAW_CODIESP = ROOT / "raw" / "codiesp" / "extracted" / "final_dataset_v4_to_publish"
RAW_VALID_CODES = ROOT / "raw" / "valid_codes" / "extracted" / "codiesp_codes" / "codiesp-D_codes.tsv"
RAW_ICD = ROOT / "raw" / "icd10cm_fy2018" / "extracted"
TABULAR_XML = RAW_ICD / "icd10cm_tabular_2018.xml"
INDEX_XML = RAW_ICD / "icd10cm_index_2018.xml"
CODES_TXT = RAW_ICD / "icd10cm_codes_2018.txt"

DATA_DIR = ROOT / "data" / "codiesp"
INDEX_DIR = ROOT / "index" / "icd10cm_fy2018"
SCHEMA_DIR = ROOT / "schema" / "icd10cm"
RESULTS_DIR = ROOT / "results" / "codiesp"

CODE_RE = re.compile(r"^[A-Z][0-9][A-Z0-9](?:\.?[A-Z0-9]{1,4})?$")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass
class VerifyReport:
    codiesp_root: str
    codiesp_splits: dict[str, dict[str, Any]]
    valid_codes: dict[str, Any]
    icd10cm: dict[str, Any]
    discrepancies: list[str] = field(default_factory=list)


@dataclass
class DiagInfo:
    code: str
    desc: str
    chapter_number: str
    chapter_desc: str
    section_id: str
    section_desc: str
    parent_path: list[str]
    inclusion_terms: list[str]
    inherited_includes: list[str]
    excludes1: list[str]
    excludes2: list[str]
    extensions: dict[str, str]


def clean(text: Any) -> str:
    return re.sub(r"\s+", " ", "" if text is None else str(text)).strip()


def normalize_code(code: str) -> str:
    return clean(code).upper().replace(".", "")


def dotted_from_raw(raw: str) -> str:
    raw = normalize_code(raw)
    if len(raw) <= 3:
        return raw
    return f"{raw[:3]}.{raw[3:]}"


def text_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def text_all(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def note_texts(elem: ET.Element | None) -> list[str]:
    if elem is None:
        return []
    return [clean(" ".join(note.itertext())) for note in elem.findall(".//note") if clean(" ".join(note.itertext()))]


def direct_child_text(elem: ET.Element, tag: str) -> str:
    child = elem.find(tag)
    return clean("".join(child.itertext())) if child is not None else ""


def direct_child_note_texts(elem: ET.Element, tag: str) -> list[str]:
    child = elem.find(tag)
    return note_texts(child)


def load_valid_codes(path: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    display_by_raw: dict[str, str] = {}
    es_by_raw: dict[str, str] = {}
    en_by_raw: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 3:
                raise ValueError(f"Expected 3 columns in {path}, got {len(row)}: {row[:5]}")
            code, es_desc, en_desc = row
            raw = normalize_code(code)
            display_by_raw[raw] = clean(code).upper()
            es_by_raw[raw] = clean(es_desc)
            en_by_raw[raw] = clean(en_desc)
    return display_by_raw, es_by_raw, en_by_raw


def load_billable_code_descriptions(path: Path) -> dict[str, str]:
    desc_by_raw: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line:
                continue
            raw = normalize_code(line[:8])
            desc = clean(line[8:])
            if raw and desc:
                desc_by_raw[raw] = desc
    return desc_by_raw


def collect_index_terms(path: Path) -> dict[str, list[str]]:
    root = ET.parse(path).getroot()
    terms_by_code: dict[str, set[str]] = defaultdict(set)

    def walk_terms(elem: ET.Element, trail: list[str]) -> None:
        title = direct_child_text(elem, "title")
        next_trail = trail + ([title] if title else [])
        code = direct_child_text(elem, "code")
        if code:
            terms_by_code[normalize_code(code)].add(" > ".join(next_trail))
        for child in elem.findall("term"):
            walk_terms(child, next_trail)

    for main in root.findall("./letter/mainTerm"):
        walk_terms(main, [])
    return {code: sorted(values) for code, values in terms_by_code.items()}


def collect_tabular(path: Path) -> tuple[dict[str, DiagInfo], list[dict[str, str]], list[dict[str, str]]]:
    root = ET.parse(path).getroot()
    by_code: dict[str, DiagInfo] = {}
    chapters: list[dict[str, str]] = []
    raw_extensions: list[dict[str, str]] = []

    def walk_diag(
        diag: ET.Element,
        chapter_number: str,
        chapter_desc: str,
        section_id: str,
        section_desc: str,
        parent_path: list[str],
        inherited_includes: list[str],
    ) -> None:
        code = normalize_code(direct_child_text(diag, "name"))
        desc = direct_child_text(diag, "desc")
        label = f"{dotted_from_raw(code)} {desc}" if code else desc
        local_includes = direct_child_note_texts(diag, "includes")
        inclusion_terms = direct_child_note_texts(diag, "inclusionTerm")
        excludes1 = direct_child_note_texts(diag, "excludes1")
        excludes2 = direct_child_note_texts(diag, "excludes2")
        extensions = {
            clean(ext.attrib.get("char", "")): clean("".join(ext.itertext()))
            for ext in diag.findall("./sevenChrDef/extension")
            if clean(ext.attrib.get("char", "")) and clean("".join(ext.itertext()))
        }
        for char, definition in extensions.items():
            raw_extensions.append({"code": dotted_from_raw(code), "char": char, "definition": definition})
        if code:
            by_code[code] = DiagInfo(
                code=dotted_from_raw(code),
                desc=desc,
                chapter_number=chapter_number,
                chapter_desc=chapter_desc,
                section_id=section_id,
                section_desc=section_desc,
                parent_path=parent_path,
                inclusion_terms=inclusion_terms,
                inherited_includes=inherited_includes,
                excludes1=excludes1,
                excludes2=excludes2,
                extensions=extensions,
            )
        next_inherited = inherited_includes + local_includes
        next_parent_path = parent_path + ([label] if label else [])
        for child in diag.findall("./diag"):
            walk_diag(
                child,
                chapter_number,
                chapter_desc,
                section_id,
                section_desc,
                next_parent_path,
                next_inherited,
            )

    for chapter in root.findall("./chapter"):
        chapter_number = direct_child_text(chapter, "name")
        chapter_desc = direct_child_text(chapter, "desc")
        chapters.append({"number": chapter_number, "description": chapter_desc})
        chapter_includes = direct_child_note_texts(chapter, "includes")
        for section in chapter.findall("./section"):
            section_id = clean(section.attrib.get("id", ""))
            section_desc = direct_child_text(section, "desc")
            section_includes = direct_child_note_texts(section, "includes")
            inherited = chapter_includes + section_includes
            for diag in section.findall("./diag"):
                walk_diag(
                    diag,
                    chapter_number,
                    chapter_desc,
                    section_id,
                    section_desc,
                    [],
                    inherited,
                )

    return by_code, chapters, raw_extensions


def role_for_code(raw: str) -> str:
    raw = normalize_code(raw)
    first = raw[0]
    if first == "C" or raw[:3] in {"D00", "D01", "D02", "D03", "D04", "D05", "D06", "D07", "D09"}:
        return "neoplasm"
    if first == "R":
        return "symptom-sign-abnormal-finding"
    if first in {"S", "T"}:
        return "injury-poisoning"
    if first in {"V", "W", "X", "Y"}:
        return "external-cause"
    if first == "Z":
        return "health-status-factor"
    if first == "O":
        return "pregnancy-related"
    if first == "P":
        return "perinatal"
    if first == "Q":
        return "congenital"
    return "disease-disorder"


def temporal_category(definition: str) -> str:
    text = definition.lower()
    if "routine healing" in text:
        return "subsequent-routine-healing"
    if "delayed healing" in text:
        return "subsequent-delayed-healing"
    if "nonunion" in text:
        return "subsequent-nonunion"
    if "malunion" in text:
        return "subsequent-malunion"
    if "subsequent" in text:
        return "subsequent-encounter"
    if "initial" in text:
        return "initial-encounter"
    if "sequela" in text:
        return "sequela"
    if "stage unspecified" in text or "unspecified" in text:
        return "unspecified"
    if "not applicable" in text:
        return "not-applicable"
    if "fetus" in text:
        return "fetus-specific"
    if "stage" in text:
        return "stage"
    return "other-extension"


def build_taxonomy() -> dict[str, Any]:
    display_by_raw, es_by_raw, valid_en_by_raw = load_valid_codes(RAW_VALID_CODES)
    descriptions = load_billable_code_descriptions(CODES_TXT)
    tabular, chapters, raw_extensions = collect_tabular(TABULAR_XML)
    index_terms = collect_index_terms(INDEX_XML)
    inventory = sorted(set(display_by_raw) & set(descriptions))
    taxonomy_rows = []
    metadata_rows = []

    for raw in inventory:
        display = display_by_raw[raw]
        info = tabular.get(raw)
        canonical_label = descriptions[raw]
        if info:
            definition_parts = []
            definition_parts.extend(info.inclusion_terms)
            definition_parts.extend(info.inherited_includes)
            hierarchy = " > ".join(
                part for part in [info.chapter_desc, info.section_desc, *info.parent_path] if part
            )
            if hierarchy:
                definition_parts.append(hierarchy)
            definition_parts.extend(index_terms.get(raw, [])[:20])
            definition = clean(". ".join(part for part in definition_parts if part))
            excludes = {"excludes1": info.excludes1, "excludes2": info.excludes2}
            structural = {
                "chapter_number": info.chapter_number,
                "chapter_desc": info.chapter_desc,
                "section_range": info.section_id,
                "code_character_length": len(raw),
                "seven_character_extension": raw[-1] if len(raw) == 7 else "",
                "seven_character_extension_definition": info.extensions.get(raw[-1], "") if len(raw) == 7 else "",
                "role": role_for_code(raw),
            }
        else:
            definition = ""
            excludes = {"excludes1": [], "excludes2": []}
            structural = {
                "chapter_number": "",
                "chapter_desc": "",
                "section_range": "",
                "code_character_length": len(raw),
                "seven_character_extension": raw[-1] if len(raw) == 7 else "",
                "seven_character_extension_definition": "",
                "role": role_for_code(raw),
            }
        retrieval_text = clean(f"{display}. {canonical_label}. {definition}")
        taxonomy_rows.append(
            {
                "tag": display,
                "type": "diagnosis",
                "standard_label": canonical_label,
                "documentation": definition,
                "references": [],
                "retrieval_text": retrieval_text,
                "codiesp_spanish_description": es_by_raw.get(raw, ""),
                "codiesp_english_description": valid_en_by_raw.get(raw, ""),
                "structural_metadata": structural,
                "excludes": excludes,
            }
        )
        metadata_rows.append({"code": display, **structural})

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(INDEX_DIR / "icd10cm_fy2018_retrieval.jsonl", taxonomy_rows)
    (INDEX_DIR / "inventory_manifest.json").write_text(
        json.dumps(
            {
                "inventory_size": len(inventory),
                "valid_diagnosis_codes": len(display_by_raw),
                "fy2018_billable_codes": len(descriptions),
                "intersection_rule": "CodiEsp diagnosis valid codes intersected with ICD-10-CM FY2018 billable codes.",
                "definition_substitution": [
                    "current-code inclusionTerm notes",
                    "inherited includes notes from chapter/section/ancestor diag",
                    "hierarchy path text: chapter desc > section desc > parent diag desc",
                    "alphabetic-index lead/sub-term paths resolving to the code",
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(INDEX_DIR / "code_metadata.jsonl", metadata_rows)
    return {
        "taxonomy_rows": taxonomy_rows,
        "inventory": set(inventory),
        "display_by_raw": display_by_raw,
        "chapters": chapters,
        "raw_extensions": raw_extensions,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def category_phrases(name: str) -> list[str]:
    return [name, name.replace("-", " "), name.replace("-", "/")]


def build_schema(taxonomy: dict[str, Any]) -> None:
    chapters = taxonomy["chapters"]
    rows = taxonomy["taxonomy_rows"]
    raw_extensions = taxonomy["raw_extensions"]
    family_vocab = {
        f"chapter-{chapter['number']}": [chapter["description"]]
        for chapter in chapters
        if chapter.get("number") and chapter.get("description")
    }
    role_names = sorted({row["structural_metadata"]["role"] for row in rows})
    role_vocab = {role: category_phrases(role) for role in role_names}
    qualifier_vocab = {
        "acute": ["acute"],
        "chronic": ["chronic"],
        "acute-on-chronic": ["acute on chronic", "acute-on-chronic"],
        "with-complication": ["with complication", "complicated"],
        "without-complication": ["without complication", "uncomplicated"],
        "mild": ["mild"],
        "moderate": ["moderate"],
        "severe": ["severe"],
        "displaced": ["displaced"],
        "nondisplaced": ["nondisplaced", "non displaced"],
        "primary": ["primary"],
        "secondary": ["secondary"],
        "malignant": ["malignant"],
        "benign": ["benign"],
        "in-situ": ["in situ", "in-situ"],
        "uncertain-behavior": ["uncertain behavior"],
        "unspecified": ["unspecified"],
        "type-1": ["type 1", "type I"],
        "type-2": ["type 2", "type II"],
        "open": ["open"],
        "closed": ["closed"],
    }
    scope_vocab = {
        "right": ["right", "right side", "right eye", "right arm", "right leg"],
        "left": ["left", "left side", "left eye", "left arm", "left leg"],
        "bilateral": ["bilateral", "both sides"],
        "unspecified-side": ["unspecified side", "unspecified eye", "unspecified limb"],
        "not-applicable": ["not applicable", "without laterality"],
    }
    raw_temporal = [
        {"definition": key, "count": value}
        for key, value in sorted(Counter(item["definition"] for item in raw_extensions).items())
    ]
    clustered_temporal: dict[str, set[str]] = defaultdict(set)
    for item in raw_temporal:
        clustered_temporal[temporal_category(item["definition"])].add(item["definition"])
    clustered_temporal["not-applicable"].add("not applicable")
    temporal_vocab = {key: sorted(values) for key, values in sorted(clustered_temporal.items())}
    normalization_map = {
        "version": "codiesp-icd10cm-fy2018-2026-07-29",
        "description": "CodiEsp / ICD-10-CM normalization map for the second-domain FHS transfer experiment.",
        "dimensions": {
            "family": family_vocab,
            "role": {},
            "event": {},
            "qualifier": qualifier_vocab,
            "scope": scope_vocab,
            "temporal": temporal_vocab,
            "aggregation": {},
        },
        "metadata_aliases": {
            "datatype": ["type", "datatype", "entity_type"],
            "chapter": ["chapter_number", "chapter_desc"],
            "section": ["section_range"],
        },
    }

    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in [
        ("vocab_family.json", family_vocab),
        ("vocab_role.json", role_vocab),
        ("vocab_temporal_raw.json", raw_temporal),
        ("vocab_temporal.json", temporal_vocab),
        ("vocab_qualifier.json", qualifier_vocab),
        ("vocab_scope.json", scope_vocab),
        ("normalization_map.json", normalization_map),
    ]:
        (SCHEMA_DIR / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_x_rows(split: str) -> list[dict[str, str]]:
    path = RAW_CODIESP / split / f"{split}X.tsv"
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 5:
                raise ValueError(f"Expected 5 columns in {path}, got {len(row)}: {row[:5]}")
            article_id, label, code, reference, position = row
            rows.append(
                {
                    "article_id": article_id,
                    "label": label,
                    "code": code.upper(),
                    "raw_code": normalize_code(code),
                    "reference": clean(reference),
                    "position": position,
                }
            )
    return rows


def parse_ranges(position: str) -> list[tuple[int, int]]:
    ranges = []
    for part in position.split(";"):
        fields = part.strip().split()
        if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit():
            ranges.append((int(fields[0]), int(fields[1])))
    return ranges


def line_spans(text: str) -> list[tuple[int, int, str]]:
    spans = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        line_text = line.rstrip("\r\n")
        spans.append((cursor, cursor + len(line_text), line_text))
        cursor += len(line)
    if not spans and text:
        spans.append((0, len(text), text))
    return spans


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    idx = 0
    while idx < len(text):
        char = text[idx]
        boundary = False
        if char in ".!?":
            next_idx = idx + 1
            while next_idx < len(text) and text[next_idx].isspace():
                next_idx += 1
            prev = text[idx - 1] if idx > 0 else ""
            next_char = text[next_idx] if next_idx < len(text) else ""
            is_initial_abbrev = char == "." and prev.isupper() and idx >= 2 and text[idx - 2] == "."
            boundary = (
                not is_initial_abbrev
                and (next_idx >= len(text) or next_char.isupper() or next_char.isdigit() or next_char in "\"¿¡")
            )
        if boundary:
            sent = clean(text[start : idx + 1])
            if sent:
                spans.append((start, idx + 1, sent))
            start = idx + 1
        idx += 1
    tail = clean(text[start:])
    if tail:
        spans.append((start, len(text), tail))
    return spans


def sentence_indices_for_ranges(span_lines: list[tuple[int, int, str]], ranges: list[tuple[int, int]]) -> list[int]:
    indices = set()
    for start, end in ranges:
        for idx, (line_start, line_end, _) in enumerate(span_lines):
            if start < line_end and end > line_start:
                indices.add(idx)
    return sorted(indices)


def extract_span(text: str, ranges: list[tuple[int, int]]) -> str:
    parts = []
    for start, end in ranges:
        if 0 <= start <= end <= len(text):
            parts.append(text[start:end])
    return clean(" ".join(parts))


def locate_english_locus(article_id: str, position: str, reference: str, split: str) -> dict[str, Any]:
    es_path = RAW_CODIESP / split / "text_files" / f"{article_id}.txt"
    en_path = RAW_CODIESP / split / "text_files_en" / f"{article_id}.txt"
    es_text = text_all(es_path)
    en_lines = text_lines(en_path)
    es_sentence_spans = sentence_spans(es_text)
    ranges = parse_ranges(position)
    spanish_from_offsets = extract_span(es_text, ranges)
    sentence_indices = sentence_indices_for_ranges(es_sentence_spans, ranges)
    alignment_ok = len(es_sentence_spans) == len(en_lines)
    fallback_reason = ""
    if alignment_ok and sentence_indices and max(sentence_indices) < len(en_lines):
        locus = clean(" ".join(en_lines[idx] for idx in sentence_indices))
        locus_level = "aligned_sentence"
    else:
        locus = clean("\n".join(en_lines))
        locus_level = "document"
        fallback_reason = "sentence_count_mismatch_or_offset_out_of_range"
    return {
        "english_locus": locus,
        "located_english_span": locus,
        "english_context": clean("\n".join(en_lines)),
        "spanish_offset_text": spanish_from_offsets,
        "spanish_reference_text": reference,
        "sentence_indices": sentence_indices,
        "spanish_sentence_count": len(es_sentence_spans),
        "english_sentence_count": len(en_lines),
        "alignment_ok": alignment_ok,
        "locus_level": locus_level,
        "fallback_reason": fallback_reason,
    }


def build_test_facts(taxonomy: dict[str, Any], target_facts: int, seed: int) -> dict[str, Any]:
    inventory = taxonomy["inventory"]
    display_by_raw = taxonomy["display_by_raw"]
    x_rows = [
        row
        for row in load_x_rows("test")
        if row["label"] == "DIAGNOSTICO" and row["raw_code"] in inventory
    ]
    by_doc: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen = set()
    for row in x_rows:
        key = (row["article_id"], row["position"], row["raw_code"])
        if key in seen:
            continue
        seen.add(key)
        by_doc[row["article_id"]].append(row)
    doc_ids = sorted(by_doc)
    rng = random.Random(seed)
    rng.shuffle(doc_ids)
    selected_docs = []
    fact_total = 0
    for doc_id in doc_ids:
        selected_docs.append(doc_id)
        fact_total += len(by_doc[doc_id])
        if fact_total >= target_facts:
            break
    selected_docs = sorted(selected_docs)

    facts = []
    diagnostics = Counter()
    for doc_id in selected_docs:
        for row in by_doc[doc_id]:
            locus = locate_english_locus(doc_id, row["position"], row["reference"], "test")
            diagnostics["facts"] += 1
            diagnostics["alignment_ok"] += int(locus["alignment_ok"])
            diagnostics[f"locus_{locus['locus_level']}"] += 1
            display_code = display_by_raw[row["raw_code"]]
            input_fields = {
                "entity": locus["located_english_span"],
                "type": "diagnosis",
                "row_context": "",
                "column_context": "",
                "original_context": locus["english_locus"],
                "document_context": locus["english_context"],
                "spanish_reference_text": row["reference"],
                "spanish_offset_text": locus["spanish_offset_text"],
                "located_english_span": locus["located_english_span"],
                "codiesp_position": row["position"],
                "document_id": doc_id,
            }
            facts.append(
                {
                    "context_id": doc_id,
                    "ground_truth_concepts": [display_code],
                    "ground_truth_count": 1,
                    "input": json.dumps(input_fields, ensure_ascii=False),
                    "input_fields": input_fields,
                    "input_type": "text",
                    "output": json.dumps([display_code]),
                    "source_sample_idx": doc_id,
                    "split": "test",
                    "codiesp_label": row["label"],
                    "alignment": locus,
                }
            )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "test_docs.txt").write_text("\n".join(selected_docs) + "\n", encoding="utf-8")
    write_jsonl(DATA_DIR / "facts_test.jsonl", facts)
    write_spotcheck(facts, DATA_DIR / "spotcheck_50.tsv", seed)
    stats = {
        "seed": seed,
        "selected_documents": len(selected_docs),
        "source_contexts": len(selected_docs),
        "target_facts": len(facts),
        "unique_gold_concepts": len({fact["ground_truth_concepts"][0] for fact in facts}),
        "facts_per_context": round(len(facts) / len(selected_docs), 4) if selected_docs else 0.0,
        "alignment_ok_rate": round(diagnostics["alignment_ok"] / diagnostics["facts"], 6) if diagnostics["facts"] else 0.0,
        "sentence_locus_fallback_rate": round(diagnostics["locus_document"] / diagnostics["facts"], 6) if diagnostics["facts"] else 0.0,
        "locus_counts": {key.replace("locus_", ""): value for key, value in diagnostics.items() if key.startswith("locus_")},
    }
    (DATA_DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"facts": facts, "stats": stats}


def write_spotcheck(facts: list[dict[str, Any]], path: Path, seed: int) -> None:
    sample = facts[:]
    random.Random(seed).shuffle(sample)
    rows = sample[:50]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "document_id",
                "spanish_original",
                "english_locus",
                "located_english_span",
                "gold_code",
                "gold_description",
                "locus_level",
                "spanish_sentence_count",
                "english_sentence_count",
            ]
        )
        label_by_code = {row["tag"]: row["standard_label"] for row in load_jsonl(INDEX_DIR / "icd10cm_fy2018_retrieval.jsonl")}
        for fact in rows:
            alignment = fact["alignment"]
            code = fact["ground_truth_concepts"][0]
            writer.writerow(
                [
                    fact["context_id"],
                    alignment["spanish_reference_text"],
                    alignment["english_locus"],
                    alignment["located_english_span"],
                    code,
                    label_by_code.get(code, ""),
                    alignment["locus_level"],
                    alignment["spanish_sentence_count"],
                    alignment["english_sentence_count"],
                ]
            )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def verify_inputs() -> VerifyReport:
    discrepancies: list[str] = []
    splits = {}
    for split in ("train", "dev", "test"):
        split_dir = RAW_CODIESP / split
        d_path = split_dir / f"{split}D.tsv"
        x_path = split_dir / f"{split}X.tsv"
        text_files = split_dir / "text_files"
        text_files_en = split_dir / "text_files_en"
        d_cols = sorted({len(row) for row in csv.reader(d_path.open(encoding="utf-8"), delimiter="\t")})
        x_cols = sorted({len(row) for row in csv.reader(x_path.open(encoding="utf-8"), delimiter="\t")})
        text_count = len(list(text_files.glob("*.txt")))
        text_en_count = len(list(text_files_en.glob("*.txt")))
        if d_cols != [2]:
            discrepancies.append(f"{split}D.tsv column counts are {d_cols}, expected [2].")
        if x_cols != [5]:
            discrepancies.append(f"{split}X.tsv column counts are {x_cols}, expected [5].")
        if text_count != text_en_count:
            discrepancies.append(f"{split} text_files count {text_count} != text_files_en count {text_en_count}.")
        splits[split] = {
            "D_columns": d_cols,
            "X_columns": x_cols,
            "text_files": text_count,
            "text_files_en": text_en_count,
            "has_discontinuous_offsets": any(";" in row["position"] for row in load_x_rows(split)),
        }
    report = VerifyReport(
        codiesp_root=str(RAW_CODIESP.relative_to(ROOT)),
        codiesp_splits=splits,
        valid_codes={"D_columns": [3], "path": str(RAW_VALID_CODES.relative_to(ROOT))},
        icd10cm={
            "tabular_xml": str(TABULAR_XML.relative_to(ROOT)),
            "index_xml": str(INDEX_XML.relative_to(ROOT)),
            "codes_txt": str(CODES_TXT.relative_to(ROOT)),
            "observed_elements": [
                "chapter",
                "section",
                "diag",
                "name",
                "desc",
                "inclusionTerm",
                "includes",
                "excludes1",
                "excludes2",
                "sevenChrDef",
                "extension",
            ],
        },
        discrepancies=discrepancies,
    )
    (ROOT / "verify_report.json").write_text(json.dumps(report.__dict__, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def write_notes(report: VerifyReport, taxonomy: dict[str, Any], test_data: dict[str, Any]) -> None:
    manifest = json.loads((INDEX_DIR / "inventory_manifest.json").read_text(encoding="utf-8"))
    stats = test_data["stats"]
    lines = [
        "# CodiEsp / ICD-10-CM Transfer Notes",
        "",
        "## Sources",
        "",
        "- CodiEsp corpus: Zenodo DOI 10.5281/zenodo.3837305, file `codiesp.zip`, downloaded for this run.",
        "- CodiEsp valid code list: Zenodo DOI 10.5281/zenodo.3706838, file `codiesp_codes.zip`, downloaded for this run.",
        "- ICD-10-CM FY2018: CDC archive files `ICD-10-CM-Codes-Tables-and-Index-2018.zip` and `2018-ICD-10-CM-Codes-File.zip`, downloaded for this run.",
        "",
        "## License Notes",
        "",
        "- CodiEsp record is listed as Creative Commons Attribution 4.0 International on the European Language Grid mirror; verify exact Zenodo landing-page license before publication.",
        "- ICD-10-CM files are distributed by CDC/NCHS; verify final public-domain wording before publication.",
        "",
        "## Verify-Before-Trust Findings",
        "",
        f"- CodiEsp root directory: `{report.codiesp_root}`.",
        "- CodiEsp D files have 2 columns; X files have 5 columns.",
        "- CodiEsp X offsets include semicolon-separated discontinuous spans.",
        "- English machine-translated `text_files_en` directories are present.",
        "- Spanish and English line/sentence counts are not always equal; alignment fallbacks are logged.",
        "- ICD XML contains the expected `chapter`, `section`, `diag`, `name`, `desc`, `inclusionTerm`, `includes`, `excludes1`, `excludes2`, `sevenChrDef`, and `extension` elements.",
        "",
        "## ICD Definition Substitution",
        "",
        "ICD-10-CM lacks a US-GAAP-style per-code definition paragraph. The `documentation` field is generated by concatenating, in order:",
        "1. Current-code `inclusionTerm` notes.",
        "2. Inherited `includes` notes from chapter/section/ancestor diagnoses.",
        "3. Hierarchy path text: chapter description > section description > parent diagnosis descriptions.",
        "4. Alphabetic-index lead/sub-term paths that resolve to the code.",
        "",
        "`excludes1` and `excludes2` are stored in a separate `excludes` field and are not included in `documentation`.",
        "",
        "## Counts",
        "",
        f"- Inventory size: {manifest['inventory_size']}.",
        f"- Selected test documents: {stats['selected_documents']} with seed {stats['seed']}.",
        f"- Target facts: {stats['target_facts']}.",
        f"- Unique gold concepts: {stats['unique_gold_concepts']}.",
        f"- Facts per context: {stats['facts_per_context']}.",
        f"- Exact sentence-count alignment rate: {stats['alignment_ok_rate']}.",
        f"- Document-locus fallback rate: {stats['sentence_locus_fallback_rate']}.",
        "",
    ]
    if report.discrepancies:
        lines.extend(["## Discrepancies", ""])
        lines.extend(f"- {item}" for item in report.discrepancies)
        lines.append("")
    (ROOT / "NOTES.md").write_text("\n".join(lines), encoding="utf-8")


def write_results_stub(test_data: dict[str, Any]) -> None:
    stats = test_data["stats"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "diagnostics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (RESULTS_DIR / "main_table.md").write_text(
        "| Method | R@10 | R@50 | MRR | Acc. | std |\n|---|---:|---:|---:|---:|---:|\n",
        encoding="utf-8",
    )
    (RESULTS_DIR / "specificity_table.md").write_text(
        "| Group | Method | n | R@10 | MRR | Acc. |\n|---|---|---:|---:|---:|---:|\n",
        encoding="utf-8",
    )
    (ROOT / "RESULTS.md").write_text(
        "# CodiEsp Results\n\nFull GPU runs have not been executed yet. Data preparation diagnostics are in `results/codiesp/diagnostics.json`.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-facts", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for required in (RAW_CODIESP, RAW_VALID_CODES, TABULAR_XML, INDEX_XML, CODES_TXT):
        if not required.exists():
            raise SystemExit(f"Missing required input: {required}")
    report = verify_inputs()
    taxonomy = build_taxonomy()
    build_schema(taxonomy)
    test_data = build_test_facts(taxonomy, target_facts=args.target_facts, seed=args.seed)
    write_notes(report, taxonomy, test_data)
    write_results_stub(test_data)
    print(
        json.dumps(
            {
                "inventory_size": len(taxonomy["inventory"]),
                "selected_documents": test_data["stats"]["selected_documents"],
                "target_facts": test_data["stats"]["target_facts"],
                "alignment_ok_rate": test_data["stats"]["alignment_ok_rate"],
                "sentence_locus_fallback_rate": test_data["stats"]["sentence_locus_fallback_rate"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
