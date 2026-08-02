#!/usr/bin/env python3
"""Build an enriched US-GAAP 2024 retrieval JSONL from the taxonomy package."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


NS = {
    "xs": "http://www.w3.org/2001/XMLSchema",
    "link": "http://www.xbrl.org/2003/linkbase",
    "xlink": "http://www.w3.org/1999/xlink",
    "ref": "http://www.xbrl.org/2006/ref",
    "codification": "http://fasb.org/codification-part/2024",
}

XLINK_LABEL = f"{{{NS['xlink']}}}label"
XLINK_HREF = f"{{{NS['xlink']}}}href"
XLINK_FROM = f"{{{NS['xlink']}}}from"
XLINK_TO = f"{{{NS['xlink']}}}to"
XLINK_ROLE = f"{{{NS['xlink']}}}role"

STANDARD_LABEL_ROLE = "http://www.xbrl.org/2003/role/label"
DOCUMENTATION_ROLE = "http://www.xbrl.org/2003/role/documentation"


def normalize_space(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def clean_label(label: str) -> str:
    return normalize_space(label.replace(",", " "))


def strip_namespace(value: str | None) -> str:
    if not value:
        return ""
    return value.rsplit(":", 1)[-1]


def tag_from_href(href: str | None) -> str | None:
    if not href or "#" not in href:
        return None
    fragment = href.rsplit("#", 1)[-1]
    if fragment.startswith("us-gaap_"):
        return fragment[len("us-gaap_") :]
    return None


def extract_zip_if_needed(zip_path: Path, extract_dir: Path) -> None:
    if extract_dir.exists():
        return
    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir.parent)


def load_old_bm25_labels(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    labels: dict[str, str] = {}
    types: dict[str, str] = {}
    if not path.exists():
        return labels, types

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            tag = str(item.get("us_gaap_tag") or item.get("tag") or "")
            if not tag:
                continue
            label = normalize_space(str(item.get("text") or ""))
            entity_type = normalize_space(str(item.get("entity_type") or item.get("type") or ""))
            if label:
                labels[tag] = label
            if entity_type:
                types[tag] = entity_type
    return labels, types


def parse_concepts(xsd_path: Path) -> list[dict[str, str]]:
    concepts: list[dict[str, str]] = []
    for _, elem in ET.iterparse(xsd_path, events=("end",)):
        if elem.tag != f"{{{NS['xs']}}}element":
            continue
        tag = elem.attrib.get("name")
        if not tag:
            continue
        concepts.append(
            {
                "tag": tag,
                "type": strip_namespace(elem.attrib.get("type")),
            }
        )
        elem.clear()
    return concepts


def parse_label_linkbase(path: Path, desired_role: str) -> dict[str, str]:
    values: dict[str, str] = {}
    root = ET.parse(path).getroot()

    for label_link in root.findall("link:labelLink", NS):
        loc_to_tag = {
            loc.attrib.get(XLINK_LABEL): tag_from_href(loc.attrib.get(XLINK_HREF))
            for loc in label_link.findall("link:loc", NS)
        }
        resources = {
            label.attrib.get(XLINK_LABEL): (
                label.attrib.get(XLINK_ROLE),
                normalize_space("".join(label.itertext())),
            )
            for label in label_link.findall("link:label", NS)
        }

        for arc in label_link.findall("link:labelArc", NS):
            tag = loc_to_tag.get(arc.attrib.get(XLINK_FROM))
            role_text = resources.get(arc.attrib.get(XLINK_TO))
            if not tag or role_text is None:
                continue
            role, text = role_text
            if role == desired_role and text and tag not in values:
                values[tag] = text

    return values


def child_texts_by_localname(element: ET.Element) -> dict[str, list[str]]:
    values: dict[str, list[str]] = defaultdict(list)
    for child in list(element):
        text = normalize_space("".join(child.itertext()))
        if text:
            values[local_name(child.tag)].append(text)
    return values


def first(values: dict[str, list[str]], key: str) -> str:
    vals = values.get(key) or []
    return vals[0] if vals else ""


def format_reference(reference: ET.Element) -> str:
    values = child_texts_by_localname(reference)
    topic = first(values, "Topic")
    subtopic = first(values, "SubTopic")
    section = first(values, "Section")
    paragraph = first(values, "Paragraph")
    name = first(values, "Name")

    if topic and subtopic and section and paragraph:
        return f"ASC {topic}-{subtopic}-{section}-{paragraph}"

    uri = first(values, "URI")
    if uri:
        tail = uri.rstrip("/").rsplit("/", 1)[-1]
        if tail:
            return tail

    parts = []
    for key in ("Name", "Number", "Section", "Paragraph", "Subparagraph"):
        val = first(values, key)
        if val:
            parts.append(val)
    return normalize_space(" ".join(parts) or name)


def parse_references(path: Path) -> dict[str, list[str]]:
    tag_refs: dict[str, list[str]] = defaultdict(list)
    root = ET.parse(path).getroot()

    for reference_link in root.findall("link:referenceLink", NS):
        loc_to_tag = {
            loc.attrib.get(XLINK_LABEL): tag_from_href(loc.attrib.get(XLINK_HREF))
            for loc in reference_link.findall("link:loc", NS)
        }
        resources = {
            reference.attrib.get(XLINK_LABEL): format_reference(reference)
            for reference in reference_link.findall("link:reference", NS)
        }

        for arc in reference_link.findall("link:referenceArc", NS):
            tag = loc_to_tag.get(arc.attrib.get(XLINK_FROM))
            reference = resources.get(arc.attrib.get(XLINK_TO))
            if tag and reference and reference not in tag_refs[tag]:
                tag_refs[tag].append(reference)

    return dict(tag_refs)


def build_retrieval_text(parts: Iterable[str]) -> str:
    return ". ".join(part for part in (normalize_space(p) for p in parts) if part)


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_dataset(args: argparse.Namespace) -> dict[str, object]:
    retrieval_root = Path(args.retrieval_root)
    zip_path = retrieval_root / "us-gaap-2024.zip"
    taxonomy_root = retrieval_root / "us-gaap-2024"
    extract_zip_if_needed(zip_path, taxonomy_root)

    elts_dir = taxonomy_root / "elts"
    xsd_path = elts_dir / "us-gaap-2024.xsd"
    label_path = elts_dir / "us-gaap-lab-2024.xml"
    doc_path = elts_dir / "us-gaap-doc-2024.xml"
    ref_path = elts_dir / "us-gaap-ref-2024.xml"
    old_bm25_path = retrieval_root / "us_gaap_2024_BM25.jsonl"

    old_labels, old_types = load_old_bm25_labels(old_bm25_path)
    concepts = parse_concepts(xsd_path)
    xml_labels = parse_label_linkbase(label_path, STANDARD_LABEL_ROLE)
    docs = parse_label_linkbase(doc_path, DOCUMENTATION_ROLE)
    references = parse_references(ref_path)

    rows: list[dict[str, object]] = []
    for concept in concepts:
        tag = concept["tag"]
        entity_type = old_types.get(tag) or concept["type"]
        standard_label = old_labels.get(tag) or clean_label(xml_labels.get(tag, ""))
        documentation = docs.get(tag, "")
        row = {
            "tag": tag,
            "type": entity_type,
            "standard_label": standard_label,
            "documentation": documentation,
            "references": references.get(tag, []),
            "retrieval_text": build_retrieval_text([tag, standard_label, documentation]),
        }
        rows.append(row)

    output_dir = Path(args.output_dir)
    output_path = output_dir / "us_gaap_2024_enriched_retrieval.jsonl"
    summary_path = output_dir / "summary.json"
    sample_path = output_dir / "sample.json"

    count = write_jsonl(output_path, rows)
    summary = {
        "output_path": str(output_path),
        "records": count,
        "tags_with_standard_label": sum(1 for row in rows if row["standard_label"]),
        "tags_with_documentation": sum(1 for row in rows if row["documentation"]),
        "tags_with_references": sum(1 for row in rows if row["references"]),
        "source_files": {
            "xsd": str(xsd_path),
            "labels": str(label_path),
            "documentation": str(doc_path),
            "references": str(ref_path),
            "old_bm25_labels": str(old_bm25_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sample_tag = args.sample_tag
    sample_row = next((row for row in rows if row["tag"] == sample_tag), rows[0] if rows else {})
    sample_path.write_text(json.dumps(sample_row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-root", default="retrieval_data", help="Directory containing us-gaap-2024.zip.")
    parser.add_argument(
        "--output-dir",
        default="retrieval_data/us_gaap_2024_enriched",
        help="Directory for the enriched JSONL and summary files.",
    )
    parser.add_argument("--sample-tag", default="OperatingLeaseCost")
    return parser.parse_args()


def main() -> None:
    summary = build_dataset(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
