#!/usr/bin/env python3
"""Grounding experiments for FinTagging context-aware tag selection.

The direct retrieval method has two stages:

1. BM25 retrieval over enriched ICD-10-CM diagnosis-code descriptions.
2. Optional Qwen reranking over the retrieved candidates.

The one-pass grounding method first asks an LLM to generate a brief concept
description from the source context, entity, and type, then uses that description
with the entity/type as the BM25 query. Candidate reranking and evaluation are
shared with direct retrieval.

Additional comparison methods change only the candidate-generation stage. They
generate one or more retrieval queries, retrieve with the same BM25 index, fuse
multi-round candidates with RRF, and then use the same reranker and evaluator.

Retrieval-only stages can run on CPU. LLM query generation and reranking require
the model backend selected by the command line.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

try:
    import torch
except ImportError:  # pragma: no cover - torch is only required for model generation.
    torch = None

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_TEST_JSONL = SCRIPT_DIR / "FinTagging_800_200_grounding_test_JSON" / "data" / "test.jsonl"
DEFAULT_TAXONOMY_JSONL = (
    PROJECT_ROOT / "retrieval_data" / "us_gaap_2024_enriched" / "us_gaap_2024_enriched_retrieval.jsonl"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "runs_fintagging_grounding_baseline" / "qwen3_32b_direct_retrieval"

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
TAG_PREFIX = ""
QUERY_MODE_ALIASES = {
    "direct": "direct_retrieval",
    "direct_retrieval": "direct_retrieval",
    "decomposed": "decomposed_retrieval",
    "decomposed_retrieval": "decomposed_retrieval",
    "feedback": "retrieval_feedback_refinement",
    "intrinsic": "intrinsic_self_refinement",
    "intrinsic_self_refinement": "intrinsic_self_refinement",
    "llm_description": "one_pass_grounding",
    "memory": "memory_guided_refinement",
    "memory_guided_refinement": "memory_guided_refinement",
    "one_pass_grounding": "one_pass_grounding",
    "operator": "operator_refinement",
    "operator_refinement": "operator_refinement",
    "parallel": "parallel_sampling",
    "parallel_sampling": "parallel_sampling",
    "parallel_diversity": "parallel_sampling_diversity",
    "parallel_sampling_diversity": "parallel_sampling_diversity",
    "retrieval_feedback": "retrieval_feedback_refinement",
    "retrieval_feedback_refinement": "retrieval_feedback_refinement",
    "self_refinement": "intrinsic_self_refinement",
    "bandit_freeform": "bandit_freeform",
    "bandit_guided_freeform": "bandit_freeform",
    "bandit_freeform_10arm": "bandit_freeform_10arm",
    "bandit_guided_freeform_10arm": "bandit_freeform_10arm",
    "ags": "frozen_ags",
    "fhs": "frozen_ags",
    "frozen_ags": "frozen_ags",
    "frozen_ags_grounding": "frozen_ags",
    "fhs_j1": "fhs_j1",
    "frozen_ags_j1": "fhs_j1",
    "fhs_no_verifier": "fhs_no_verifier",
    "frozen_ags_no_verifier": "fhs_no_verifier",
    "fhs_j3_wcov0": "fhs_j3_wcov0",
    "fhs_j3_wcov1": "fhs_j3_wcov1",
    "fhs_j4_wcov0": "fhs_j4_wcov0",
    "fhs_j4_wcov1": "fhs_j4_wcov1",
    "gold_label_definition": "gold_label_definition_retrieval",
    "gold_label_definition_retrieval": "gold_label_definition_retrieval",
    "oracle_definition": "gold_label_definition_retrieval",
    "oracle_retrieval": "gold_label_definition_retrieval",
    "ags_seq": "ags_seq",
    "ags_sequential": "ags_seq",
    "ags_seq_random": "ags_seq_random",
    "ags_sequential_random": "ags_seq_random",
    "seq_verifier": "seq_verifier",
    "fhs_seq": "seq_verifier",
    "ags_j1": "one_pass_structured",
    "one_pass_structured": "one_pass_structured",
    "one_pass_grounding_structured": "one_pass_structured",
}
MULTI_ROUND_QUERY_MODES = {
    "intrinsic_self_refinement",
    "retrieval_feedback_refinement",
    "parallel_sampling",
    "parallel_sampling_diversity",
    "decomposed_retrieval",
    "operator_refinement",
    "memory_guided_refinement",
    "bandit_freeform",
    "bandit_freeform_10arm",
    "frozen_ags",
    "fhs_j1",
    "fhs_no_verifier",
    "fhs_j3_wcov0",
    "fhs_j3_wcov1",
    "fhs_j4_wcov0",
    "fhs_j4_wcov1",
    "ags_seq",
    "ags_seq_random",
    "seq_verifier",
    "one_pass_structured",
}
LLM_QUERY_MODES = MULTI_ROUND_QUERY_MODES | {"one_pass_grounding"}
STRUCTURED_QUERY_MODES = {"operator_refinement", "memory_guided_refinement"}
# frozen_ags samples J structured hypotheses but runs no refinement loop; it produces its
# own final ranking via internal fuse + rerank (see ags_frozen_grounding).
FROZEN_AGS_QUERY_MODES = {"frozen_ags"}
FHS_J1_QUERY_MODES = {"fhs_j1"}
FHS_NO_VERIFIER_QUERY_MODES = {"fhs_no_verifier"}
FHS_J_SWEEP_QUERY_MODES = {"fhs_j3_wcov0", "fhs_j3_wcov1", "fhs_j4_wcov0", "fhs_j4_wcov1"}
# "One-pass grounding (structured)" = AGS with J=1: the identical frozen code path with one
# greedy hypothesis and beta=0, so the consensus term drops out and the ranking is the
# range-normalized sum-RRF score alone. Shares frozen_ags's startup assertions and record
# builder; only the config constants differ (ags_frozen_grounding._FROZEN_VARIANTS).
ONE_PASS_STRUCTURED_QUERY_MODES = {"one_pass_structured"}
# Everything that runs through ags_frozen_grounding.build_frozen_ags_method_record.
FROZEN_AGS_FAMILY_QUERY_MODES = (
    FROZEN_AGS_QUERY_MODES
    | FHS_J1_QUERY_MODES
    | FHS_NO_VERIFIER_QUERY_MODES
    | FHS_J_SWEEP_QUERY_MODES
    | ONE_PASS_STRUCTURED_QUERY_MODES
)
# The two sequential negative-control arms start from frozen_ags's round one and differ
# from each other only in directive selection (see ags_sequential_arms).
AGS_SEQ_QUERY_MODES = {"ags_seq", "ags_seq_random"}
# The matched control the paper describes: FHS with the parallel fan replaced by a loop, every
# other component including the candidate-level verifier held identical. Separate from
# AGS_SEQ_QUERY_MODES because it shares none of their controller machinery -- no posterior bank,
# no slate scoring, no counterfactual replay. See ags_seq_verifier_arm.
SEQ_VERIFIER_QUERY_MODES = {"seq_verifier"}
DIMENSIONS = ("FAMILY", "ROLE", "EVENT", "QUALIFIER", "SCOPE", "TEMPORAL")
OPERATOR_LIBRARY = (
    "direct_label",
    "row_column",
    "relative_time",
    "roll_forward",
    "dimensional",
    "aggregation",
    "rate",
    "schedule",
)
FREEFORM_REWRITE_ARMS = (
    "paraphrase",
    "expand",
    "simplify",
    "decompose",
    "alternative",
    "freeform_revise",
)
FREEFORM_REWRITE_ARMS_10 = FREEFORM_REWRITE_ARMS + (
    "sharpen",
    "broaden_scope",
    "shift_temporal_framing",
    "restate_as_definition",
)
FREEFORM_REWRITE_INSTRUCTIONS = {
    "paraphrase": "Restate the interpretation in different wording without changing its meaning.",
    "expand": "Add detail or context that the evidence supports but the current grounding omits.",
    "simplify": "Remove specifics and state the interpretation more generally.",
    "decompose": "Split the interpretation into its constituent parts and ground the core part.",
    "alternative": "Propose a different plausible reading of the same evidence.",
    "freeform_revise": "Revise however the retrieved candidates suggest is best.",
    "sharpen": "Make the interpretation more specific while staying supported by the evidence.",
    "broaden_scope": "State the interpretation at a broader scope or less granular level.",
    "shift_temporal_framing": "Try a different plausible temporal framing of the same evidence.",
    "restate_as_definition": "Rewrite the interpretation as a taxonomy-definition-style sentence.",
}
Q_LAB_FORMATTING_INSTRUCTION = (
    "For q_lab, emit compact label-form content words only, no function words, "
    "in canonical-label word order. Use space-separated lowercase words."
)
FREEFORM_FEATURE_NAMES = [
    "bias",
    "is_table",
    "is_text",
    "is_monetary",
    "is_shares",
    "token_count_lt_12",
    "token_count_gt_30",
    "has_temporal_cue",
    "has_aggregation_cue",
    "has_scope_cue",
    "critique_mismatch_flag",
    "neighborhood_novelty",
    "round_idx",
    "prior_arm_count",
]
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "this",
    "to",
    "was",
    "were",
    "which",
    "with",
}


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = normalize_space(data)
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return normalize_space(" ".join(self.parts))


@dataclass(frozen=True)
class Concept:
    tag: str
    raw_tag: str
    entity_type: str
    standard_label: str
    documentation: str
    references: list[str]
    retrieval_text: str


@dataclass(frozen=True)
class Example:
    example_idx: int
    context_id: Any
    source_sample_idx: Any
    input_type: str
    entity: str
    entity_type: str
    row_context: str
    column_context: str
    original_context: str
    query_context: str
    gold_tags: list[str]


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def normalize_tag(tag: Any) -> str:
    text = normalize_space(tag)
    if not text:
        return ""
    return text if text.startswith(TAG_PREFIX) else f"{TAG_PREFIX}{text}"


def raw_tag(tag: Any) -> str:
    text = normalize_space(tag)
    return text[len(TAG_PREFIX) :] if text.startswith(TAG_PREFIX) else text


def canonical_query_mode(value: Any) -> str:
    text = normalize_space(value)
    canonical = QUERY_MODE_ALIASES.get(text)
    if canonical is None:
        raise ValueError(
            f"Unsupported query_mode={text}. Expected one of: {sorted(QUERY_MODE_ALIASES)}"
        )
    return canonical


def tokenize(text: Any) -> list[str]:
    expanded = CAMEL_BOUNDARY_RE.sub(" ", str(text))
    tokens: list[str] = []
    for token in TOKEN_RE.findall(f"{text} {expanded}"):
        normalized = normalize_token(token)
        if normalized and normalized not in STOPWORDS:
            tokens.append(normalized)
    return tokens


def normalize_token(token: str) -> str:
    token = token.lower()
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def html_to_visible_text(text: str) -> str:
    if "<" not in text or ">" not in text:
        return normalize_space(unescape(text))
    parser = VisibleTextParser()
    try:
        parser.feed(text)
        parsed = parser.text()
    except Exception:
        parsed = ""
    return parsed or normalize_space(re.sub(r"<[^>]+>", " ", unescape(text)))


def truncate_text(text: str, max_chars: int) -> str:
    text = normalize_space(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head].rstrip()} ... {text[-tail:].lstrip()}"


def extract_balanced_json(text: str, start_char: str, end_char: str) -> str | None:
    start = text.find(start_char)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def parse_json_object(text: str) -> tuple[dict[str, Any], bool]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = cleaned.strip("`")
    for candidate in (cleaned, extract_balanced_json(cleaned, "{", "}")):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, True
    return {}, False


def parse_json_value(text: str) -> tuple[Any, bool]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = cleaned.strip("`")
    candidates = [
        cleaned,
        extract_balanced_json(cleaned, "{", "}"),
        extract_balanced_json(cleaned, "[", "]"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate), True
        except json.JSONDecodeError:
            continue
    return None, False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_examples(path: Path, limit: int | None = None) -> list[Example]:
    examples: list[Example] = []
    rows = load_jsonl(path)
    if limit is not None:
        rows = rows[:limit]

    for example_idx, row in enumerate(rows):
        fields = row.get("input_fields")
        if not isinstance(fields, dict):
            fields = json.loads(row.get("input", "{}"))
        original_context = normalize_space(fields.get("original_context", ""))
        visible_context = html_to_visible_text(original_context)
        entity = normalize_space(
            fields.get("numeric_entity")
            or fields.get("entity")
            or fields.get("value")
            or fields.get("numeric_value")
            or ""
        )
        entity_type = normalize_space(fields.get("datatype") or fields.get("type") or "")
        row_context = normalize_space(fields.get("row_context", ""))
        column_context = normalize_space(fields.get("column_context", ""))
        gold_tags = [normalize_tag(tag) for tag in row.get("ground_truth_concepts", []) if normalize_tag(tag)]
        examples.append(
            Example(
                example_idx=example_idx,
                context_id=row.get("context_id"),
                source_sample_idx=row.get("source_sample_idx"),
                input_type=normalize_space(row.get("input_type", "")),
                entity=entity,
                entity_type=entity_type,
                row_context=row_context,
                column_context=column_context,
                original_context=original_context,
                query_context=visible_context,
                gold_tags=gold_tags,
            )
        )
    return examples


def load_taxonomy(path: Path) -> list[Concept]:
    concepts: list[Concept] = []
    for row in load_jsonl(path):
        tag = normalize_tag(row["tag"])
        concepts.append(
            Concept(
                tag=tag,
                raw_tag=raw_tag(tag),
                entity_type=normalize_space(row.get("type", "")),
                standard_label=normalize_space(row.get("standard_label", "")),
                documentation=normalize_space(row.get("documentation", "")),
                references=[normalize_space(ref) for ref in row.get("references", []) if normalize_space(ref)],
                retrieval_text=normalize_space(row.get("retrieval_text", "")),
            )
        )
    return concepts


class BM25Index:
    def __init__(
        self,
        concepts: list[Concept],
        k1: float = 1.5,
        b: float = 0.75,
        include_type_in_text: bool = False,
    ) -> None:
        self.concepts = concepts
        self.k1 = k1
        self.b = b
        self.doc_tokens = [
            tokenize(
                f"{concept.entity_type}. {concept.retrieval_text}"
                if include_type_in_text
                else concept.retrieval_text
            )
            for concept in concepts
        ]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_freq: Counter[str] = Counter()

        for doc_idx, tokens in enumerate(self.doc_tokens):
            counts = Counter(tokens)
            doc_freq.update(counts.keys())
            for token, freq in counts.items():
                self.postings[token].append((doc_idx, freq))

        n_docs = len(self.doc_tokens)
        self.idf = {
            token: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }

    def rank(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if not self.doc_tokens or self.avg_doc_length <= 0:
            return []

        scores: dict[int, float] = defaultdict(float)
        for token in sorted(set(tokenize(query))):
            postings = self.postings.get(token)
            if not postings:
                continue
            idf = self.idf.get(token, 0.0)
            for doc_idx, freq in postings:
                doc_len = self.doc_lengths[doc_idx]
                denom = freq + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_length)
                scores[doc_idx] += idf * freq * (self.k1 + 1.0) / denom

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) < min(top_k, len(self.concepts)):
            seen = {idx for idx, _ in ranked}
            ranked.extend((idx, 0.0) for idx in range(len(self.concepts)) if idx not in seen)
        return ranked[:top_k]


class TaxonomyRetriever:
    def __init__(
        self,
        concepts: list[Concept],
        type_filter: bool,
        label_coverage_weight: float = 0.0,
        label_coverage_pool_multiplier: int = 10,
    ) -> None:
        self.type_filter = type_filter
        self.label_coverage_weight = label_coverage_weight
        self.label_coverage_pool_multiplier = label_coverage_pool_multiplier
        self.all_index = BM25Index(concepts, include_type_in_text=True)
        self.by_type: dict[str, list[Concept]] = defaultdict(list)
        for concept in concepts:
            self.by_type[concept.entity_type].append(concept)
        self.index_by_type = {
            entity_type: BM25Index(type_concepts, include_type_in_text=False)
            for entity_type, type_concepts in self.by_type.items()
        }

    def label_coverage(self, query_tokens: set[str], concept: Concept) -> float:
        label_tokens = set(tokenize(concept.standard_label or concept.raw_tag))
        if not label_tokens:
            return 0.0
        return len(query_tokens & label_tokens) / len(label_tokens)

    def query_label_coverage(self, query_tokens: set[str], concept: Concept) -> float:
        label_tokens = set(tokenize(concept.standard_label or concept.raw_tag))
        if not query_tokens or not label_tokens:
            return 0.0
        return len(query_tokens & label_tokens) / len(query_tokens)

    def retrieve(self, query: str, entity_type: str, top_k: int) -> list[tuple[Concept, float, float, float, float, float]]:
        if self.type_filter and entity_type in self.index_by_type:
            index = self.index_by_type[entity_type]
        else:
            index = self.all_index
        if self.label_coverage_weight <= 0.0:
            return [(index.concepts[idx], score, 0.0, 0.0, score, score) for idx, score in index.rank(query, top_k)]

        if self.label_coverage_pool_multiplier <= 0:
            pool_k = len(index.concepts)
        else:
            pool_k = min(len(index.concepts), max(top_k, top_k * self.label_coverage_pool_multiplier))
        query_tokens = set(tokenize(query))
        ranked = index.rank(query, pool_k)
        raw_scores = [score for _, score in ranked]
        lo = min(raw_scores) if raw_scores else 0.0
        hi = max(raw_scores) if raw_scores else 0.0
        rescored = []
        for idx, bm25_score in ranked:
            concept = index.concepts[idx]
            coverage = self.label_coverage(query_tokens, concept)
            query_coverage = self.query_label_coverage(query_tokens, concept)
            normalized_bm25 = (bm25_score - lo) / (hi - lo) if hi > lo else 1.0
            retrieval_score = normalized_bm25 + self.label_coverage_weight * (coverage + query_coverage)
            rescored.append((concept, bm25_score, coverage, query_coverage, normalized_bm25, retrieval_score))
        rescored.sort(
            key=lambda item: (
                -item[5],
                -item[2],
                -item[3],
                -item[1],
                normalize_tag(item[0].tag),
            )
        )
        return rescored[:top_k]


def build_direct_query(example: Example) -> str:
    return normalize_space(f"{example.entity} {example.entity_type} {example.query_context}")


def serialize_evidence(example: Example, context_max_chars: int) -> str:
    lines = [
        f"Entity value: {example.entity}",
        f"Datatype: {example.entity_type}",
        f"Input type: {example.input_type}",
    ]
    if example.row_context:
        lines.append(f"Row context: {example.row_context}")
    if example.column_context:
        lines.append(f"Column context: {example.column_context}")
    lines.append("Source context:")
    lines.append(truncate_text(example.query_context, context_max_chars))
    return "\n".join(lines)


def retrieval_query_from_grounding(example: Example, grounding: str) -> str:
    grounding = normalize_space(grounding)
    if not grounding:
        grounding = build_direct_query(example)
    return normalize_space(f"{example.entity} {example.entity_type} {grounding}")


def build_retrieval_query(
    example: Example,
    query_mode: str,
    query_descriptions: dict[int, str] | None = None,
    concept_by_tag: dict[str, Concept] | None = None,
) -> tuple[str, str | None]:
    query_mode = canonical_query_mode(query_mode)
    if query_mode == "direct_retrieval":
        return build_direct_query(example), None

    if query_mode == "gold_label_definition_retrieval":
        concept = (concept_by_tag or {}).get(example.gold_tags[0] if example.gold_tags else "")
        if concept is None:
            return build_direct_query(example), None
        query = normalize_space(
            f"{concept.raw_tag} {concept.standard_label} {concept.documentation}"
        )
        return query, query

    if query_mode != "one_pass_grounding":
        raise ValueError(f"Unsupported query_mode={query_mode}")

    description = normalize_space((query_descriptions or {}).get(example.example_idx, ""))
    if not description:
        description = build_direct_query(example)
    return normalize_space(f"{example.entity} {example.entity_type} {description}"), description


def concept_to_candidate(
    concept: Concept,
    rank: int,
    bm25_score: float,
    label_coverage: float = 0.0,
    query_label_coverage: float = 0.0,
    bm25_normalized_score: float | None = None,
    retrieval_score: float | None = None,
) -> dict[str, Any]:
    retrieval_score = bm25_score if retrieval_score is None else retrieval_score
    bm25_normalized_score = bm25_score if bm25_normalized_score is None else bm25_normalized_score
    return {
        "rank": rank,
        "tag": concept.tag,
        "type": concept.entity_type,
        "standard_label": concept.standard_label,
        "documentation": concept.documentation,
        "references": concept.references,
        "retrieval_text": concept.retrieval_text,
        "bm25_score": round(bm25_score, 8),
        "bm25_normalized_score": round(bm25_normalized_score, 8),
        "label_coverage": round(label_coverage, 8),
        "query_label_coverage": round(query_label_coverage, 8),
        "retrieval_score": round(retrieval_score, 8),
    }


def first_gold_rank(ranking: list[str], gold_tags: Iterable[str]) -> int | None:
    gold = set(gold_tags)
    for idx, tag in enumerate(ranking, start=1):
        if normalize_tag(tag) in gold:
            return idx
    return None


def metric_row(ranking: list[str], gold_tags: list[str], top_ks: tuple[int, ...]) -> dict[str, Any]:
    rank = first_gold_rank(ranking, gold_tags)
    row: dict[str, Any] = {
        "rank": rank,
        "mrr": 0.0 if rank is None else 1.0 / rank,
        "accuracy": bool(rank == 1),
    }
    for k in top_ks:
        row[f"recall_at_{k}"] = bool(rank is not None and rank <= k)
    return row


def aggregate_metric_rows(rows: list[dict[str, Any]], top_ks: tuple[int, ...]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    metrics: dict[str, Any] = {
        "n": n,
        "accuracy": round(sum(bool(row["accuracy"]) for row in rows) / n, 6),
        "mrr": round(sum(float(row["mrr"]) for row in rows) / n, 6),
    }
    for k in top_ks:
        metrics[f"recall_at_{k}"] = round(
            sum(bool(row[f"recall_at_{k}"]) for row in rows) / n,
            6,
        )
    return metrics


def retrieve_candidates(
    retriever: TaxonomyRetriever,
    query: str,
    entity_type: str,
    top_k: int,
) -> list[dict[str, Any]]:
    return [
        concept_to_candidate(
            concept,
            rank,
            bm25_score,
            label_coverage,
            query_label_coverage,
            bm25_normalized_score,
            retrieval_score,
        )
        for rank, (
            concept,
            bm25_score,
            label_coverage,
            query_label_coverage,
            bm25_normalized_score,
            retrieval_score,
        ) in enumerate(
            retriever.retrieve(query, entity_type, top_k),
            start=1,
        )
    ]


def fuse_round_candidates(
    rounds: list[dict[str, Any]],
    top_k: int | None,
    rrf_kappa: float,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    first_round: dict[str, int] = {}
    best_candidate: dict[str, dict[str, Any]] = {}
    round_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for round_record in rounds:
        round_idx = int(round_record.get("round", 1))
        for candidate in round_record.get("candidates", []):
            tag = normalize_tag(candidate.get("tag"))
            rank = int(candidate.get("rank", 0) or 0)
            if not tag or rank <= 0:
                continue
            scores[tag] += 1.0 / (rrf_kappa + rank)
            if tag not in best_rank or rank < best_rank[tag]:
                best_rank[tag] = rank
                best_candidate[tag] = dict(candidate)
            if tag not in first_round:
                first_round[tag] = round_idx
            round_hits[tag].append(
                {
                    "round": round_idx,
                    "rank": rank,
                    "bm25_score": candidate.get("bm25_score"),
                    "bm25_normalized_score": candidate.get("bm25_normalized_score"),
                    "label_coverage": candidate.get("label_coverage"),
                    "query_label_coverage": candidate.get("query_label_coverage"),
                    "retrieval_score": candidate.get("retrieval_score"),
                }
            )

    ranked_tags = sorted(
        scores,
        key=lambda tag: (-scores[tag], first_round.get(tag, 10**9), best_rank.get(tag, 10**9), tag),
    )
    if top_k is not None and top_k > 0:
        ranked_tags = ranked_tags[:top_k]

    fused: list[dict[str, Any]] = []
    for final_rank, tag in enumerate(ranked_tags, start=1):
        candidate = dict(best_candidate[tag])
        candidate["rank"] = final_rank
        candidate["rrf_score"] = round(scores[tag], 8)
        candidate["best_round_rank"] = best_rank[tag]
        candidate["first_retrieved_round"] = first_round[tag]
        candidate["round_hits"] = round_hits[tag]
        fused.append(candidate)
    return fused


def finalize_candidate_record(
    example: Example,
    query_mode: str,
    rounds: list[dict[str, Any]],
    top_k: int,
    rrf_kappa: float,
    total_llm_calls: int = 0,
    total_prompt_tokens: int = 0,
    total_completion_tokens: int = 0,
    wall_time: float | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_candidates = fuse_round_candidates(rounds, top_k, rrf_kappa)
    candidate_tags = [candidate["tag"] for candidate in final_candidates]
    retrieval_metrics = metric_row(candidate_tags, example.gold_tags, (10, 50, top_k))
    search_coverage = any(tag in set(candidate_tags) for tag in example.gold_tags)
    last_round = rounds[-1] if rounds else {}
    record: dict[str, Any] = {
        "instance_id": example.example_idx,
        "example_idx": example.example_idx,
        "context_id": example.context_id,
        "source_sample_idx": example.source_sample_idx,
        "input_type": example.input_type,
        "entity": example.entity,
        "type": example.entity_type,
        "row_context": example.row_context,
        "column_context": example.column_context,
        "gold_tags": example.gold_tags,
        "gold_concept": example.gold_tags[0] if example.gold_tags else None,
        "method": query_mode,
        "query_mode": query_mode,
        "query": last_round.get("query", ""),
        "query_context": example.query_context,
        "query_description": last_round.get("grounding"),
        "rounds": rounds,
        "final_candidates": final_candidates,
        "candidates": final_candidates,
        "candidate_union_tags": candidate_tags,
        "search_coverage": search_coverage,
        "retrieval_metrics": retrieval_metrics,
        "gold_rank": retrieval_metrics.get("rank"),
        "total_llm_calls": total_llm_calls,
        "total_retrieval_calls": len(rounds),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
    }
    if wall_time is not None:
        record["wall_time"] = round(wall_time, 6)
    if extra_fields:
        record.update(extra_fields)
    return record


def build_candidate_records(
    examples: list[Example],
    taxonomy: list[Concept],
    top_k: int,
    type_filter: bool,
    query_mode: str,
    rrf_kappa: float = 60.0,
    query_descriptions: dict[int, str] | None = None,
    label_coverage_weight: float = 0.0,
    label_coverage_pool_multiplier: int = 10,
) -> list[dict[str, Any]]:
    query_mode = canonical_query_mode(query_mode)
    # Defaults match TaxonomyRetriever's own, so callers that do not pass these keep the exact
    # behaviour every published non-frozen baseline ran with (w_cov=0.0). This is the retriever
    # the non-LLM query modes actually use -- the frozen-family override in the method-record
    # builder never reaches here, which is why direct_retrieval and one_pass_grounding ran with
    # the coverage term off.
    retriever = TaxonomyRetriever(
        taxonomy,
        type_filter=type_filter,
        label_coverage_weight=label_coverage_weight,
        label_coverage_pool_multiplier=label_coverage_pool_multiplier,
    )
    if label_coverage_weight > 0.0:
        print(
            f"build_candidate_records: w_cov={label_coverage_weight} "
            f"pool_multiplier={label_coverage_pool_multiplier}",
            flush=True,
        )
    records: list[dict[str, Any]] = []
    concept_by_tag = {concept.tag: concept for concept in taxonomy}
    for example in examples:
        query, query_description = build_retrieval_query(
            example,
            query_mode,
            query_descriptions,
            concept_by_tag=concept_by_tag,
        )
        candidates = retrieve_candidates(retriever, query, example.entity_type, top_k)
        rounds = [
            {
                "round": 1,
                "grounding": query_description,
                "query": query,
                "candidates": candidates,
            }
        ]
        records.append(
            finalize_candidate_record(
                example,
                query_mode=query_mode,
                rounds=rounds,
                top_k=top_k,
                rrf_kappa=rrf_kappa,
        total_llm_calls=1 if query_mode == "one_pass_grounding" else 0,
            )
        )
    return records


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_predictions(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    predictions: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(path):
        predictions[int(row["example_idx"])] = row
    return predictions


def candidate_query_mode(record: dict[str, Any]) -> str:
    return canonical_query_mode(record.get("query_mode") or "direct_retrieval")


def validate_candidate_records(records: list[dict[str, Any]], query_mode: str, top_k: int) -> None:
    query_mode = canonical_query_mode(query_mode)
    mismatches = [
        record.get("example_idx")
        for record in records
        if candidate_query_mode(record) != query_mode
    ]
    if mismatches:
        raise ValueError(
            f"Existing candidate file has query_mode={candidate_query_mode(records[0])}, "
            f"but requested query_mode={query_mode}. Set REUSE_CANDIDATES=0 or use a separate OUTPUT_DIR."
        )

    wrong_top_k = [
        record.get("example_idx")
        for record in records
        if len(record.get("candidates", [])) > top_k
    ]
    if wrong_top_k:
        raise ValueError(
            f"Existing candidate file contains more than top_k={top_k} candidates in at least one row. "
            "Set REUSE_CANDIDATES=0 to rebuild it."
        )


def format_candidate_for_prompt(candidate: dict[str, Any], doc_max_chars: int) -> str:
    docs = truncate_text(candidate.get("documentation", ""), doc_max_chars)
    refs = ", ".join(candidate.get("references", [])[:3])
    parts = [
        f"[{candidate['rank']}] {candidate['tag']}",
        f"Type: {candidate.get('type', '')}",
        f"Label: {candidate.get('standard_label', '')}",
    ]
    if docs:
        parts.append(f"Definition: {docs}")
    if refs:
        parts.append(f"References: {refs}")
    return "\n".join(parts)


def build_rerank_messages(
    record: dict[str, Any],
    context_max_chars: int,
    doc_max_chars: int,
    rerank_list_size: int,
) -> list[dict[str, str]]:
    context = truncate_text(str(record.get("query_context") or record.get("query", "")), context_max_chars)
    candidate_text = "\n\n".join(
        format_candidate_for_prompt(candidate, doc_max_chars)
        for candidate in record.get("candidates", [])
    )
    user = f"""Select the ICD-10-CM diagnosis code that best matches the clinical evidence.

Input:
Entity: {record.get("entity", "")}
Type: {record.get("type", "")}
Context:
{context}

Candidates:
{candidate_text}

Return JSON only with this schema:
{{"selected_index": 1, "selected_tag": "A00.0", "ranked_indices": [1, 2, 3]}}

Rules:
- Choose only from the candidate list.
- `selected_index` must be the index shown in square brackets.
- `ranked_indices` should contain up to {rerank_list_size} best candidate indices, best first.
- Do not include explanations or markdown."""
    return [
        {
            "role": "system",
            "content": "You are a precise ICD-10-CM diagnosis-code grounding reranker.",
        },
        {"role": "user", "content": user},
    ]


def messages_to_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages) + "\n\nASSISTANT:\n"


def build_prompt_under_token_budget(
    tokenizer: Any,
    record: dict[str, Any],
    context_max_chars: int,
    doc_max_chars: int,
    rerank_list_size: int,
    max_input_tokens: int,
) -> tuple[str, int, int, int]:
    context_options = [context_max_chars, min(context_max_chars, 8000), min(context_max_chars, 5000)]
    doc_options = [doc_max_chars, min(doc_max_chars, 240), min(doc_max_chars, 120), 0]
    tried: set[tuple[int, int]] = set()

    last_prompt = ""
    last_tokens = 0
    for ctx_chars in context_options:
        for doc_chars in doc_options:
            if (ctx_chars, doc_chars) in tried:
                continue
            tried.add((ctx_chars, doc_chars))
            messages = build_rerank_messages(record, ctx_chars, doc_chars, rerank_list_size)
            prompt = messages_to_prompt(tokenizer, messages)
            token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            last_prompt = prompt
            last_tokens = token_count
            if token_count <= max_input_tokens:
                return prompt, token_count, ctx_chars, doc_chars

    return last_prompt, last_tokens, min(context_options), min(doc_options)


def build_query_description_messages(example: Example, context_max_chars: int) -> list[dict[str, str]]:
    context = truncate_text(example.query_context, context_max_chars)
    user = f"""Write a brief retrieval query for finding the correct ICD-10-CM diagnosis code.

Input:
Entity value: {example.entity}
Type: {example.entity_type}
Source context:
{context}

Return JSON only with this schema:
{{"query": "brief concept description"}}

Rules:
- Describe the accounting concept represented by the entity in this context.
- Use wording likely to appear in ICD-10-CM labels, descriptions, or index text.
- Do not name a specific ICD-10-CM code unless it is explicitly present in the source context.
- Do not include explanations or markdown."""
    return [
        {
            "role": "system",
            "content": "You generate concise ICD-10-CM retrieval queries for diagnosis-code grounding.",
        },
        {"role": "user", "content": user},
    ]


def build_query_description_prompt(
    tokenizer: Any,
    example: Example,
    context_max_chars: int,
    max_input_tokens: int,
) -> tuple[str, int, int]:
    context_options = [context_max_chars, min(context_max_chars, 8000), min(context_max_chars, 5000)]
    last_prompt = ""
    last_tokens = 0
    for ctx_chars in context_options:
        messages = build_query_description_messages(example, ctx_chars)
        prompt = messages_to_prompt(tokenizer, messages)
        token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        last_prompt = prompt
        last_tokens = token_count
        if token_count <= max_input_tokens:
            return prompt, token_count, ctx_chars
    return last_prompt, last_tokens, min(context_options)


def clean_model_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return normalize_space(text.strip('"').strip("'"))


def parse_query_description(raw_output: str, fallback: str) -> tuple[str, bool]:
    parsed, parse_ok = parse_json_object(raw_output)
    if parse_ok:
        for key in ("query", "description", "retrieval_query"):
            value = normalize_space(parsed.get(key))
            if value:
                return value, True

    cleaned = clean_model_text(raw_output)
    if cleaned:
        first_line = normalize_space(cleaned.splitlines()[0])
        if first_line:
            return first_line, False
    return fallback, False


def compact_label_query(text: str) -> str:
    return normalize_space(" ".join(tokenize(text)))


def parse_grounding_surfaces(raw_output: str, fallback: str) -> tuple[dict[str, str], bool]:
    parsed, parse_ok = parse_json_object(raw_output)
    grounding = ""
    q_lab = ""
    q_def = ""
    if parse_ok:
        grounding = scalar_text(
            parsed.get("grounding")
            or parsed.get("query")
            or parsed.get("retrieval_query")
            or parsed.get("semantic_description")
            or parsed.get("description")
        )
        q_lab = scalar_text(parsed.get("q_lab") or parsed.get("label_query"))
        q_def = scalar_text(parsed.get("q_def") or parsed.get("definition_query"))

    if not grounding:
        grounding, parse_ok = parse_query_description(raw_output, fallback)
    grounding = grounding or fallback
    q_def = q_def or grounding
    q_lab = q_lab or compact_label_query(grounding)
    return {
        "grounding": normalize_space(grounding),
        "q_lab": normalize_space(q_lab),
        "q_def": normalize_space(q_def),
    }, bool(parse_ok and grounding)


def build_dual_grounding_messages(
    example: Example,
    context_max_chars: int,
    extra_instructions: str = "",
) -> list[dict[str, str]]:
    evidence = serialize_evidence(example, context_max_chars)
    user = f"""Generate a semantic interpretation of clinical evidence for retrieving the correct ICD-10-CM diagnosis code.

Return JSON only with this schema:
{{"grounding": "free-text interpretation of the evidence", "q_lab": "compact label-form query", "q_def": "definition-style retrieval sentence"}}

Formatting:
- grounding should describe what the evidence expresses.
- {Q_LAB_FORMATTING_INSTRUCTION}
- q_def should be one definition-style sentence expressing the same interpretation.

Evidence:
{evidence}
{extra_instructions}

Rules:
- Do not name a specific ICD-10-CM code unless it is explicitly present in the source context.
- Do not include explanations or markdown."""
    return [
        {"role": "system", "content": "You create ICD-10-CM grounding interpretations and retrieval query surfaces."},
        {"role": "user", "content": user},
    ]


def build_parallel_diversity_messages(
    example: Example,
    prior_interpretations: list[str],
    context_max_chars: int,
) -> list[dict[str, str]]:
    if prior_interpretations:
        prior_text = "\n".join(f"- {item}" for item in prior_interpretations)
        diversity_block = f"""
You have already produced the following interpretations of this evidence:
{prior_text}

Produce a new interpretation that differs from all of them in substance, not only in wording. A different interpretation resolves the evidence's ambiguity differently: for example a different reading of what the row and column jointly denote, a different temporal framing, a different aggregation level, or a different measurement basis.

Constraints:
- Do not contradict anything the evidence states explicitly.
- Do not invent attribute values the evidence gives no support for; if a dimension is genuinely unsupported, leave it out of the interpretation rather than guessing.
- The new interpretation must be one you consider plausible, not merely different."""
    else:
        diversity_block = """
This is the first interpretation. Produce the most plausible retrieval interpretation supported by the evidence."""
    return build_dual_grounding_messages(example, context_max_chars, extra_instructions=diversity_block)


def load_existing_query_descriptions(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(path):
        records[int(row["example_idx"])] = row
    return records


def query_description_record(
    example: Example,
    raw_output: str,
    prompt_tokens: int,
    used_context_chars: int,
    backend: str,
    model_name: str,
) -> dict[str, Any]:
    fallback = build_direct_query(example)
    query_description, parse_ok = parse_query_description(raw_output, fallback)
    return {
        "example_idx": example.example_idx,
        "context_id": example.context_id,
        "source_sample_idx": example.source_sample_idx,
        "input_type": example.input_type,
        "entity": example.entity,
        "type": example.entity_type,
        "query_description": query_description,
        "parse_ok": parse_ok,
        "raw_output": raw_output,
        "prompt_tokens": prompt_tokens,
        "used_context_max_chars": used_context_chars,
        "backend": backend,
        "model": model_name,
    }


def load_rerank_model(
    model_name: str,
    bf16: bool,
    trust_remote_code: bool,
    attn_implementation: str | None,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return tokenizer, model


def generate_text(
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    do_sample = temperature > 0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated = model.generate(**inputs, **generation_kwargs)
    new_tokens = generated[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_vllm_engine(args: argparse.Namespace, model_name: str) -> tuple[Any, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_generation_tokens = max(
        args.max_new_tokens,
        args.query_max_new_tokens,
        int(getattr(args, "fhs_verifier_max_new_tokens", 0) or 0),
    )
    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "trust_remote_code": args.trust_remote_code,
        "dtype": "bfloat16" if args.bf16 else "float16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_input_tokens + max_generation_tokens,
        "max_num_seqs": args.max_num_seqs,
    }
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True

    return tokenizer, LLM(**llm_kwargs)


def release_model_handles(*handles: Any) -> None:
    for handle in handles:
        del handle
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


class QueryGenerator:
    def __init__(
        self,
        args: argparse.Namespace,
        tokenizer: Any | None = None,
        llm: Any | None = None,
    ) -> None:
        self.args = args
        self.tokenizer = tokenizer
        self.llm = llm
        self.loaded_here = tokenizer is None or llm is None
        if self.tokenizer is None or self.llm is None:
            if args.query_generation_backend == "vllm":
                self.tokenizer, self.llm = load_vllm_engine(args, args.query_generation_model)
            else:
                self.tokenizer, self.llm = load_rerank_model(
                    args.query_generation_model,
                    bf16=args.bf16,
                    trust_remote_code=args.trust_remote_code,
                    attn_implementation=args.attn_implementation,
                )

    @property
    def backend(self) -> str:
        return self.args.query_generation_backend

    @property
    def model_name(self) -> str:
        return self.args.query_generation_model

    def count_text_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate_many(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        if self.args.query_generation_backend == "vllm":
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=self.args.query_temperature,
                top_p=self.args.query_top_p,
                max_tokens=self.args.query_max_new_tokens,
            )
            raw_outputs: list[str] = []
            for start in range(0, len(prompts), self.args.vllm_batch_size):
                batch = prompts[start : start + self.args.vllm_batch_size]
                outputs = self.llm.generate(batch, sampling_params)
                raw_outputs.extend(output.outputs[0].text.strip() if output.outputs else "" for output in outputs)
            return raw_outputs

        return [
            generate_text(
                self.tokenizer,
                self.llm,
                prompt,
                max_input_tokens=self.args.query_max_input_tokens,
                max_new_tokens=self.args.query_max_new_tokens,
                temperature=self.args.query_temperature,
                top_p=self.args.query_top_p,
            )
            for prompt in prompts
        ]

    def generate_one(self, prompt: str) -> str:
        return self.generate_many([prompt])[0]

    def close(self) -> None:
        if self.loaded_here:
            release_model_handles(self.llm, self.tokenizer)
            self.llm = None
            self.tokenizer = None


def build_prompt_under_query_budget(
    tokenizer: Any,
    message_builder: Any,
    context_max_chars: int,
    max_input_tokens: int,
) -> tuple[str, int, int]:
    context_options = [
        context_max_chars,
        min(context_max_chars, 8000),
        min(context_max_chars, 5000),
        min(context_max_chars, 2500),
    ]
    last_prompt = ""
    last_tokens = 0
    for ctx_chars in context_options:
        messages = message_builder(ctx_chars)
        prompt = messages_to_prompt(tokenizer, messages)
        token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        last_prompt = prompt
        last_tokens = token_count
        if token_count <= max_input_tokens:
            return prompt, token_count, ctx_chars
    return last_prompt, last_tokens, min(context_options)


def build_feedback_prompt_under_query_budget(
    tokenizer: Any,
    message_builder: Any,
    context_max_chars: int,
    doc_max_chars: int,
    max_input_tokens: int,
) -> tuple[str, int, int, int]:
    context_options = [
        context_max_chars,
        min(context_max_chars, 8000),
        min(context_max_chars, 5000),
        min(context_max_chars, 2500),
    ]
    doc_options = [doc_max_chars, min(doc_max_chars, 240), min(doc_max_chars, 120), 0]
    last_prompt = ""
    last_tokens = 0
    for ctx_chars in context_options:
        for doc_chars in doc_options:
            messages = message_builder(ctx_chars, doc_chars)
            prompt = messages_to_prompt(tokenizer, messages)
            token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            last_prompt = prompt
            last_tokens = token_count
            if token_count <= max_input_tokens:
                return prompt, token_count, ctx_chars, doc_chars
    return last_prompt, last_tokens, min(context_options), min(doc_options)


def llm_call_record(
    kind: str,
    raw_output: str,
    prompt_tokens: int,
    completion_tokens: int,
    parse_ok: bool,
    backend: str,
    model_name: str,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": kind,
        "parse_ok": parse_ok,
        "raw_output": raw_output,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "backend": backend,
        "model": model_name,
    }
    if extra_fields:
        record.update(extra_fields)
    return record


def extract_query_from_output(raw_output: str, fallback: str) -> tuple[str, bool]:
    return parse_query_description(raw_output, fallback)


def scalar_text(value: Any) -> str:
    if isinstance(value, str):
        return normalize_space(value)
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return normalize_space(", ".join(scalar_text(item) for item in value if scalar_text(item)))
    if isinstance(value, dict):
        return normalize_space(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return normalize_space(value)


def normalized_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [normalize_space(item) for item in value if normalize_space(item)]
    text = normalize_space(value)
    if not text:
        return []
    return [part.strip() for part in re.split(r"[,;]", text) if part.strip()]


def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return normalize_space(value).lower() in {"yes", "true", "1", "y"}


def format_feedback_candidates(
    candidates: list[dict[str, Any]],
    entity_type: str,
    limit: int,
    doc_max_chars: int,
) -> str:
    compatible = [candidate for candidate in candidates if not entity_type or candidate.get("type") == entity_type]
    if len(compatible) < limit:
        seen = {candidate.get("tag") for candidate in compatible}
        compatible.extend(candidate for candidate in candidates if candidate.get("tag") not in seen)
    selected = compatible[:limit]
    if not selected:
        return "No candidates were retrieved."
    return "\n\n".join(format_candidate_for_prompt(candidate, doc_max_chars) for candidate in selected)


def build_intrinsic_refinement_messages(
    example: Example,
    previous_grounding: str,
    context_max_chars: int,
) -> list[dict[str, str]]:
    evidence = serialize_evidence(example, context_max_chars)
    user = f"""You are given evidence from a financial document and your previous semantic interpretation of this evidence.

Critically evaluate your previous interpretation without seeing any retrieved taxonomy candidates. Consider whether you may have:
- Misidentified the entity type or accounting family
- Chosen the wrong temporal scope
- Confused a subtotal with a line item
- Missed a qualifier such as net vs gross or beginning vs ending
- Applied the wrong aggregation level

Return JSON only with this schema:
{{"critique": "brief critique", "query": "revised semantic retrieval description"}}

Evidence:
{evidence}

Previous interpretation:
{previous_grounding}

Rules:
- Do not include markdown.
- If the previous interpretation still looks correct, reuse it as the query."""
    return [
        {"role": "system", "content": "You refine ICD-10-CM retrieval queries by self-critique only."},
        {"role": "user", "content": user},
    ]


def build_retrieval_feedback_messages(
    example: Example,
    previous_grounding: str,
    candidates: list[dict[str, Any]],
    context_max_chars: int,
    doc_max_chars: int,
    feedback_candidate_count: int,
) -> list[dict[str, str]]:
    evidence = serialize_evidence(example, context_max_chars)
    candidate_text = format_feedback_candidates(candidates, example.entity_type, feedback_candidate_count, doc_max_chars)
    user = f"""You are given evidence from a financial document, your previous semantic interpretation, and the top taxonomy concepts retrieved using that interpretation.

Use the retrieved concepts to assess whether the interpretation is on the right track. Rewrite the semantic description to better capture what the evidence expresses. You may change entity type, temporal scope, qualifiers, aggregation level, or any other aspect.

Return JSON only with this schema:
{{"assessment": "brief assessment of the retrieved neighborhood", "query": "revised semantic retrieval description"}}

Evidence:
{evidence}

Previous interpretation:
{previous_grounding}

Retrieved concepts:
{candidate_text}

Rules:
- Do not name a gold concept.
- Do not include markdown."""
    return [
        {"role": "system", "content": "You refine ICD-10-CM retrieval queries using retrieved-neighborhood feedback."},
        {"role": "user", "content": user},
    ]


def build_parallel_sampling_messages(
    example: Example,
    sample_idx: int,
    total_samples: int,
    context_max_chars: int,
) -> list[dict[str, str]]:
    evidence = serialize_evidence(example, context_max_chars)
    user = f"""Generate a semantic interpretation of clinical evidence for retrieving the correct ICD-10-CM diagnosis code.

This is interpretation {sample_idx} of {total_samples}. Explore a different plausible reading of the evidence. Consider varying:
- The accounting family or entity type
- The temporal interpretation
- Whether this is a line item, subtotal, total, or reconciliation entry
- Measurement basis such as gross/net or beginning/ending
- The level of aggregation

Return JSON only with this schema:
{{"query": "distinct semantic retrieval description"}}

Evidence:
{evidence}

Rules:
- Make this interpretation meaningfully distinct.
- Do not include explanations or markdown."""
    return [
        {"role": "system", "content": "You generate diverse ICD-10-CM retrieval hypotheses."},
        {"role": "user", "content": user},
    ]


def build_decomposed_retrieval_messages(
    example: Example,
    total_queries: int,
    context_max_chars: int,
) -> list[dict[str, str]]:
    evidence = serialize_evidence(example, context_max_chars)
    user = f"""Decompose the financial evidence into {total_queries} separate retrieval queries, each focused on a different semantic dimension.

Use these dimensions when {total_queries} is 4:
1. Entity type and accounting family
2. Temporal scope
3. Measurement and qualifiers
4. Aggregation and structure

Return JSON only with this schema:
{{"queries": [{{"focus": "entity_family", "query": "..."}}, {{"focus": "temporal", "query": "..."}}, {{"focus": "measurement", "query": "..."}}, {{"focus": "aggregation", "query": "..."}}]}}

Evidence:
{evidence}

Rules:
- Each query must be self-contained.
- Each query should be useful for retrieval on its own.
- Do not include markdown."""
    return [
        {"role": "system", "content": "You decompose clinical evidence into dimension-focused ICD-10-CM retrieval queries."},
        {"role": "user", "content": user},
    ]


def parse_decomposed_queries(raw_output: str, fallback: str, total_queries: int) -> tuple[list[dict[str, str]], bool]:
    parsed, parse_ok = parse_json_value(raw_output)
    query_items: list[dict[str, str]] = []
    if isinstance(parsed, dict):
        values = parsed.get("queries") or parsed.get("sub_queries") or parsed.get("retrieval_queries")
        if isinstance(values, list):
            for idx, item in enumerate(values, start=1):
                if isinstance(item, dict):
                    query = scalar_text(item.get("query") or item.get("description"))
                    focus = scalar_text(item.get("focus") or item.get("dimension") or f"query_{idx}")
                else:
                    query = scalar_text(item)
                    focus = f"query_{idx}"
                if query:
                    query_items.append({"focus": focus, "query": query})
        else:
            for idx in range(1, total_queries + 1):
                query = scalar_text(parsed.get(f"query_{idx}") or parsed.get(f"query{idx}"))
                if query:
                    query_items.append({"focus": f"query_{idx}", "query": query})
    elif isinstance(parsed, list):
        for idx, item in enumerate(parsed, start=1):
            query = scalar_text(item.get("query") if isinstance(item, dict) else item)
            focus = scalar_text(item.get("focus") if isinstance(item, dict) else f"query_{idx}")
            if query:
                query_items.append({"focus": focus or f"query_{idx}", "query": query})

    if not query_items:
        cleaned = clean_model_text(raw_output)
        for line in cleaned.splitlines():
            line = normalize_space(re.sub(r"^(?:query|interpretation)\s*\d+\s*[:.)-]\s*", "", line, flags=re.I))
            if line:
                query_items.append({"focus": f"query_{len(query_items) + 1}", "query": line})
            if len(query_items) >= total_queries:
                break

    while len(query_items) < total_queries:
        query_items.append({"focus": f"fallback_{len(query_items) + 1}", "query": fallback})
    return query_items[:total_queries], bool(parse_ok and query_items)


def build_operator_initial_messages(example: Example, context_max_chars: int) -> list[dict[str, str]]:
    evidence = serialize_evidence(example, context_max_chars)
    user = f"""Produce a structured semantic hypothesis for grounding clinical evidence to an ICD-10-CM diagnosis code.

Fill each dimension only if the evidence directly supports it. Use "UNRESOLVED" when unsupported.

Dimensions:
- FAMILY: broad accounting domain
- ROLE: specific function
- EVENT: event or state
- QUALIFIER: modifiers such as gross/net, current/noncurrent, pre-tax/after-tax, weighted average
- SCOPE: dimensional context such as segment, geography, plan, security class, subsidiary
- TEMPORAL: time interpretation

Operator library:
{", ".join(OPERATOR_LIBRARY)}

Return JSON only with this schema:
{{"dimensions": {{"FAMILY": "...", "ROLE": "...", "EVENT": "...", "QUALIFIER": "...", "SCOPE": "...", "TEMPORAL": "..."}}, "operators": ["direct_label"], "retrieval_query": "compact retrieval query"}}

Evidence:
{evidence}"""
    return [
        {"role": "system", "content": "You create structured ICD-10-CM grounding hypotheses."},
        {"role": "user", "content": user},
    ]


def parse_hypothesis(raw_output: str, fallback: str) -> tuple[dict[str, Any], bool]:
    parsed, parse_ok = parse_json_object(raw_output)
    dimensions = {dimension: "UNRESOLVED" for dimension in DIMENSIONS}
    if isinstance(parsed.get("dimensions"), dict):
        for dimension in DIMENSIONS:
            value = scalar_text(parsed["dimensions"].get(dimension) or parsed["dimensions"].get(dimension.lower()))
            if value:
                dimensions[dimension] = value
    else:
        for dimension in DIMENSIONS:
            value = scalar_text(parsed.get(dimension) or parsed.get(dimension.lower()))
            if value:
                dimensions[dimension] = value

    operators = normalized_list(parsed.get("operators") or parsed.get("operator"))
    operators = [operator for operator in operators if operator in OPERATOR_LIBRARY] or ["direct_label"]
    retrieval_query = scalar_text(
        parsed.get("retrieval_query")
        or parsed.get("query")
        or parsed.get("semantic_description")
        or parsed.get("description")
    )
    if not retrieval_query:
        retrieval_query, parse_ok = extract_query_from_output(raw_output, fallback)
    return {
        "dimensions": dimensions,
        "operators": operators,
        "retrieval_query": retrieval_query or fallback,
    }, bool(parse_ok and retrieval_query)


def build_operator_feedback_messages(
    example: Example,
    hypothesis: dict[str, Any],
    candidates: list[dict[str, Any]],
    context_max_chars: int,
    doc_max_chars: int,
    feedback_candidate_count: int,
) -> list[dict[str, str]]:
    evidence = serialize_evidence(example, context_max_chars)
    candidate_text = format_feedback_candidates(candidates, example.entity_type, feedback_candidate_count, doc_max_chars)
    user = f"""Assess a structured grounding hypothesis using the retrieved taxonomy neighborhood.

For each semantic dimension, decide whether the retrieved concepts SUPPORT, CONTRADICT, or leave UNRESOLVED that dimension. Also identify alternative values and whether there is a structural-strategy mismatch.

Return JSON only with this schema:
{{"supported_dimensions": ["FAMILY"], "contradicted_dimensions": ["ROLE"], "unresolved_dimensions": ["TEMPORAL"], "alternative_values": {{"ROLE": ["..."]}}, "structural_mismatch": {{"is_mismatch": false, "reason": "..."}}}}

Evidence:
{evidence}

Current hypothesis:
{json.dumps(hypothesis, ensure_ascii=False, sort_keys=True)}

Retrieved concepts:
{candidate_text}

Rules:
- Do not try to identify the gold concept.
- Focus only on dimensional assessment.
- Do not include markdown."""
    return [
        {"role": "system", "content": "You provide structured feedback for ICD-10-CM grounding hypotheses."},
        {"role": "user", "content": user},
    ]


def parse_feedback(raw_output: str) -> tuple[dict[str, Any], bool]:
    parsed, parse_ok = parse_json_object(raw_output)
    mismatch = parsed.get("structural_mismatch")
    if not isinstance(mismatch, dict):
        mismatch = {
            "is_mismatch": bool_from_any(mismatch),
            "reason": scalar_text(mismatch),
        }
    alternatives = parsed.get("alternative_values") or parsed.get("alternatives") or {}
    if not isinstance(alternatives, dict):
        alternatives = {}
    return {
        "supported_dimensions": normalized_list(parsed.get("supported_dimensions") or parsed.get("D+") or parsed.get("supported")),
        "contradicted_dimensions": normalized_list(parsed.get("contradicted_dimensions") or parsed.get("D-") or parsed.get("contradicted")),
        "unresolved_dimensions": normalized_list(parsed.get("unresolved_dimensions") or parsed.get("D?") or parsed.get("unresolved")),
        "alternative_values": {
            normalize_space(key).upper(): normalized_list(value)
            for key, value in alternatives.items()
            if normalize_space(key)
        },
        "structural_mismatch": {
            "is_mismatch": bool_from_any(mismatch.get("is_mismatch")),
            "reason": scalar_text(mismatch.get("reason")),
        },
    }, parse_ok


def format_search_history(transitions: list[dict[str, Any]]) -> str:
    if not transitions:
        return "No previous interventions."
    parts = []
    for transition in transitions:
        parts.append(
            json.dumps(
                {
                    "from_round": transition.get("from_round"),
                    "to_round": transition.get("to_round"),
                    "feedback": transition.get("feedback"),
                    "directive": transition.get("directive"),
                    "hypothesis_after": transition.get("revised_hypothesis"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return "\n".join(parts)


def format_operator_memories(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "None."
    redacted = []
    for memory in memories:
        redacted.append(
            {
                "evidence_profile": memory.get("evidence_profile"),
                "feedback": memory.get("feedback"),
                "directive": memory.get("directive"),
                "semantic_difference": memory.get("semantic_difference"),
                "delta_reward": memory.get("delta_reward"),
            }
        )
    return json.dumps(redacted, ensure_ascii=False, sort_keys=True)


def build_operator_controller_messages(
    example: Example,
    hypothesis: dict[str, Any],
    feedback: dict[str, Any],
    transitions: list[dict[str, Any]],
    positive_memories: list[dict[str, Any]] | None,
    negative_memories: list[dict[str, Any]] | None,
    context_max_chars: int,
) -> list[dict[str, str]]:
    evidence = serialize_evidence(example, context_max_chars)
    memory_block = ""
    if positive_memories is not None or negative_memories is not None:
        memory_block = f"""

Positive memories - interventions that improved retrieval:
{format_operator_memories(positive_memories or [])}

Negative memories - interventions that did not improve retrieval:
{format_operator_memories(negative_memories or [])}
"""
    user = f"""Select one atomic semantic intervention for the next retrieval round.

Search modes:
- REFINE: fix a contradicted dimension
- BRANCH: try an alternative for an unresolved dimension
- CHANGE_STRATEGY: switch interpretation operator when the structure looks wrong

Rules:
- Modify exactly one dimension, or one tightly coupled pair.
- Preserve supported dimensions.
- Do not repeat a hypothesis already tested.

Return JSON only with this schema:
{{"mode": "REFINE", "operator": "direct_label", "target_dimension": "ROLE", "semantic_patch": "...", "preserve": ["FAMILY"], "rationale": "..."}}

Evidence:
{evidence}

Current hypothesis:
{json.dumps(hypothesis, ensure_ascii=False, sort_keys=True)}

Current feedback:
{json.dumps(feedback, ensure_ascii=False, sort_keys=True)}

Search history:
{format_search_history(transitions)}
{memory_block}"""
    return [
        {"role": "system", "content": "You control structured search over ICD-10-CM grounding hypotheses."},
        {"role": "user", "content": user},
    ]


def parse_directive(raw_output: str) -> tuple[dict[str, Any], bool]:
    parsed, parse_ok = parse_json_object(raw_output)
    mode = scalar_text(parsed.get("mode")).upper()
    if mode not in {"REFINE", "BRANCH", "CHANGE_STRATEGY"}:
        mode = "BRANCH"
    operator = scalar_text(parsed.get("operator"))
    if operator not in OPERATOR_LIBRARY:
        operator = "direct_label"
    return {
        "mode": mode,
        "operator": operator,
        "target_dimension": scalar_text(parsed.get("target_dimension") or parsed.get("dimension")).upper(),
        "semantic_patch": scalar_text(parsed.get("semantic_patch") or parsed.get("patch") or parsed.get("new_value")),
        "preserve": normalized_list(parsed.get("preserve") or parsed.get("preservation_set")),
        "rationale": scalar_text(parsed.get("rationale") or parsed.get("reason")),
    }, parse_ok


def build_operator_revision_messages(
    example: Example,
    hypothesis: dict[str, Any],
    directive: dict[str, Any],
    context_max_chars: int,
) -> list[dict[str, str]]:
    evidence = serialize_evidence(example, context_max_chars)
    user = f"""Apply the controller directive to revise a structured grounding hypothesis.

Rules:
- Apply only the specified change.
- Preserve all dimensions listed in the preservation set.
- Do not make additional changes.

Return JSON only with this schema:
{{"dimensions": {{"FAMILY": "...", "ROLE": "...", "EVENT": "...", "QUALIFIER": "...", "SCOPE": "...", "TEMPORAL": "..."}}, "operators": ["direct_label"], "semantic_difference": "what changed", "retrieval_query": "compact retrieval query"}}

Evidence:
{evidence}

Current hypothesis:
{json.dumps(hypothesis, ensure_ascii=False, sort_keys=True)}

Directive:
{json.dumps(directive, ensure_ascii=False, sort_keys=True)}"""
    return [
        {"role": "system", "content": "You revise structured ICD-10-CM grounding hypotheses."},
        {"role": "user", "content": user},
    ]


def generate_query_descriptions(
    args: argparse.Namespace,
    examples: list[Example],
    output_path: Path,
    tokenizer: Any | None = None,
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    if args.query_generation_backend == "vllm":
        return generate_query_descriptions_vllm(args, examples, output_path, tokenizer, llm)
    return generate_query_descriptions_transformers(args, examples, output_path)


def generate_query_descriptions_transformers(
    args: argparse.Namespace,
    examples: list[Example],
    output_path: Path,
) -> list[dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_query_descriptions(output_path) if args.resume else {}
    records = dict(existing)
    pending = [example for example in examples if example.example_idx not in records]
    if not pending:
        return [records[example.example_idx] for example in examples if example.example_idx in records]

    tokenizer, model = load_rerank_model(
        args.query_generation_model,
        bf16=args.bf16,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )

    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for offset, example in enumerate(pending, start=1):
            prompt, prompt_tokens, used_context_chars = build_query_description_prompt(
                tokenizer,
                example,
                context_max_chars=args.query_context_max_chars,
                max_input_tokens=args.query_max_input_tokens,
            )
            raw_output = generate_text(
                tokenizer,
                model,
                prompt,
                max_input_tokens=args.query_max_input_tokens,
                max_new_tokens=args.query_max_new_tokens,
                temperature=args.query_temperature,
                top_p=args.query_top_p,
            )
            record = query_description_record(
                example,
                raw_output=raw_output,
                prompt_tokens=prompt_tokens,
                used_context_chars=used_context_chars,
                backend="transformers",
                model_name=args.query_generation_model,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records[example.example_idx] = record

            if offset % args.log_every == 0:
                print(f"Generated {offset}/{len(pending)} query descriptions")

    release_model_handles(model, tokenizer)

    return [records[example.example_idx] for example in examples if example.example_idx in records]


def generate_query_descriptions_vllm(
    args: argparse.Namespace,
    examples: list[Example],
    output_path: Path,
    tokenizer: Any | None = None,
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_query_descriptions(output_path) if args.resume else {}
    records = dict(existing)
    pending = [example for example in examples if example.example_idx not in records]
    if not pending:
        return [records[example.example_idx] for example in examples if example.example_idx in records]

    loaded_here = tokenizer is None or llm is None
    if tokenizer is None or llm is None:
        tokenizer, llm = load_vllm_engine(args, args.query_generation_model)

    sampling_params = SamplingParams(
        temperature=args.query_temperature,
        top_p=args.query_top_p,
        max_tokens=args.query_max_new_tokens,
    )

    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for start in range(0, len(pending), args.vllm_batch_size):
            batch_examples = pending[start : start + args.vllm_batch_size]
            prompts: list[str] = []
            prompt_meta: list[tuple[int, int]] = []
            for example in batch_examples:
                prompt, prompt_tokens, used_context_chars = build_query_description_prompt(
                    tokenizer,
                    example,
                    context_max_chars=args.query_context_max_chars,
                    max_input_tokens=args.query_max_input_tokens,
                )
                prompts.append(prompt)
                prompt_meta.append((prompt_tokens, used_context_chars))

            outputs = llm.generate(prompts, sampling_params)
            for example, output, (prompt_tokens, used_context_chars) in zip(
                batch_examples,
                outputs,
                prompt_meta,
                strict=True,
            ):
                raw_output = output.outputs[0].text.strip() if output.outputs else ""
                record = query_description_record(
                    example,
                    raw_output=raw_output,
                    prompt_tokens=prompt_tokens,
                    used_context_chars=used_context_chars,
                    backend="vllm",
                    model_name=args.query_generation_model,
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records[example.example_idx] = record

            handle.flush()
            processed = min(start + len(batch_examples), len(pending))
            if processed % args.log_every == 0 or processed == len(pending):
                print(f"Generated {processed}/{len(pending)} pending query descriptions with vLLM")

    if loaded_here:
        release_model_handles(llm, tokenizer)

    return [records[example.example_idx] for example in examples if example.example_idx in records]


def load_existing_method_records(path: Path, query_mode: str) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(path):
        try:
            if candidate_query_mode(row) != query_mode:
                continue
            if "rounds" not in row or "candidates" not in row:
                continue
            records[int(row["example_idx"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return records


def make_round_record(
    retriever: TaxonomyRetriever,
    example: Example,
    round_idx: int,
    grounding: str,
    top_k: int,
    llm_calls: list[dict[str, Any]] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query = retrieval_query_from_grounding(example, grounding)
    record: dict[str, Any] = {
        "round": round_idx,
        "grounding": normalize_space(grounding),
        "query": query,
        "candidates": retrieve_candidates(retriever, query, example.entity_type, top_k),
    }
    if llm_calls:
        record["llm_calls"] = llm_calls
    if extra_fields:
        record.update(extra_fields)
    return record


def retrieve_dual_observation(
    retriever: TaxonomyRetriever,
    example: Example,
    surfaces: dict[str, str],
    round_idx: int,
    top_k: int,
    rrf_kappa: float,
    llm_calls: list[dict[str, Any]] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    q_lab = normalize_space(surfaces.get("q_lab"))
    q_def = normalize_space(surfaces.get("q_def"))
    grounding = normalize_space(surfaces.get("grounding") or q_def or q_lab)
    query_lab = retrieval_query_from_grounding(example, q_lab or grounding)
    query_def = retrieval_query_from_grounding(example, q_def or grounding)
    candidates_lab = retrieve_candidates(retriever, query_lab, example.entity_type, top_k)
    candidates_def = retrieve_candidates(retriever, query_def, example.entity_type, top_k)
    fused = fuse_round_candidates(
        [
            {"round": 1, "candidates": candidates_lab},
            {"round": 2, "candidates": candidates_def},
        ],
        top_k,
        rrf_kappa,
    )
    record: dict[str, Any] = {
        "round": round_idx,
        "grounding": grounding,
        "q_lab": q_lab,
        "q_def": q_def,
        "query": normalize_space(f"{query_lab} || {query_def}"),
        "query_lab": query_lab,
        "query_def": query_def,
        "candidates_q_lab": candidates_lab,
        "candidates_q_def": candidates_def,
        "candidates_fused": fused,
        "candidates": fused,
        "retrieval_calls": 2,
        "fusion": "per_hypothesis_dual_rrf",
    }
    if llm_calls:
        record["llm_calls"] = llm_calls
    if extra_fields:
        record.update(extra_fields)
    return record


def mean_top_score(candidates: list[dict[str, Any]], top_m: int) -> float:
    selected = candidates[:top_m]
    if not selected:
        return 0.0
    scores = [
        float(
            candidate.get("rrf_score")
            if candidate.get("rrf_score") is not None
            else candidate.get("bm25_score")
            if candidate.get("bm25_score") is not None
            else 0.0
        )
        for candidate in selected
    ]
    return sum(scores) / len(scores) if scores else 0.0


def sum_llm_usage(rounds: list[dict[str, Any]]) -> tuple[int, int, int]:
    calls = [
        call
        for round_record in rounds
        for call in round_record.get("llm_calls", [])
    ]
    prompt_tokens = sum(int(call.get("prompt_tokens", 0) or 0) for call in calls)
    completion_tokens = sum(int(call.get("completion_tokens", 0) or 0) for call in calls)
    return len(calls), prompt_tokens, completion_tokens


def parse_success_rate_from_records(records: list[dict[str, Any]]) -> float | None:
    calls = [
        call
        for record in records
        for round_record in record.get("rounds", [])
        for call in round_record.get("llm_calls", [])
        if "parse_ok" in call
    ]
    if not calls:
        return None
    return round(sum(bool(call.get("parse_ok")) for call in calls) / len(calls), 6)


def generate_initial_grounding(
    generator: QueryGenerator,
    args: argparse.Namespace,
    example: Example,
) -> tuple[str, dict[str, Any]]:
    fallback = build_direct_query(example)
    prompt, prompt_tokens, used_context_chars = build_query_description_prompt(
        generator.tokenizer,
        example,
        context_max_chars=args.query_context_max_chars,
        max_input_tokens=args.query_max_input_tokens,
    )
    raw_output = generator.generate_one(prompt)
    query, parse_ok = extract_query_from_output(raw_output, fallback)
    call = llm_call_record(
        "initial_grounding",
        raw_output=raw_output,
        prompt_tokens=prompt_tokens,
        completion_tokens=generator.count_text_tokens(raw_output),
        parse_ok=parse_ok,
        backend=generator.backend,
        model_name=generator.model_name,
        extra_fields={"used_context_max_chars": used_context_chars},
    )
    return query, call


def generate_query_from_messages(
    generator: QueryGenerator,
    kind: str,
    prompt: str,
    prompt_tokens: int,
    fallback: str,
    extra_fields: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    raw_output = generator.generate_one(prompt)
    query, parse_ok = extract_query_from_output(raw_output, fallback)
    call = llm_call_record(
        kind,
        raw_output=raw_output,
        prompt_tokens=prompt_tokens,
        completion_tokens=generator.count_text_tokens(raw_output),
        parse_ok=parse_ok,
        backend=generator.backend,
        model_name=generator.model_name,
        extra_fields=extra_fields,
    )
    return query, call


def build_freeform_method_record(
    args: argparse.Namespace,
    query_mode: str,
    generator: QueryGenerator,
    retriever: TaxonomyRetriever,
    example: Example,
) -> dict[str, Any]:
    start_time = time.monotonic()
    rounds: list[dict[str, Any]] = []
    total_rounds = args.retrieval_rounds
    fallback = build_direct_query(example)

    if query_mode == "intrinsic_self_refinement":
        grounding, call = generate_initial_grounding(generator, args, example)
        rounds.append(make_round_record(retriever, example, 1, grounding, args.top_k, [call]))
        for round_idx in range(2, total_rounds + 1):
            prompt, prompt_tokens, used_context_chars = build_prompt_under_query_budget(
                generator.tokenizer,
                lambda ctx_chars, previous=grounding: build_intrinsic_refinement_messages(
                    example,
                    previous,
                    ctx_chars,
                ),
                context_max_chars=args.query_context_max_chars,
                max_input_tokens=args.query_max_input_tokens,
            )
            grounding, call = generate_query_from_messages(
                generator,
                "intrinsic_refinement",
                prompt,
                prompt_tokens,
                fallback=grounding or fallback,
                extra_fields={"used_context_max_chars": used_context_chars},
            )
            rounds.append(make_round_record(retriever, example, round_idx, grounding, args.top_k, [call]))

    elif query_mode == "retrieval_feedback_refinement":
        grounding, call = generate_initial_grounding(generator, args, example)
        rounds.append(make_round_record(retriever, example, 1, grounding, args.top_k, [call]))
        for round_idx in range(2, total_rounds + 1):
            previous_candidates = rounds[-1]["candidates"]
            prompt, prompt_tokens, used_context_chars, used_doc_chars = build_feedback_prompt_under_query_budget(
                generator.tokenizer,
                lambda ctx_chars, doc_chars, previous=grounding, candidates=previous_candidates: build_retrieval_feedback_messages(
                    example,
                    previous,
                    candidates,
                    ctx_chars,
                    doc_chars,
                    args.feedback_candidate_count,
                ),
                context_max_chars=args.query_context_max_chars,
                doc_max_chars=args.candidate_doc_max_chars,
                max_input_tokens=args.query_max_input_tokens,
            )
            grounding, call = generate_query_from_messages(
                generator,
                "retrieval_feedback_refinement",
                prompt,
                prompt_tokens,
                fallback=grounding or fallback,
                extra_fields={
                    "used_context_max_chars": used_context_chars,
                    "used_candidate_doc_max_chars": used_doc_chars,
                },
            )
            rounds.append(make_round_record(retriever, example, round_idx, grounding, args.top_k, [call]))

    elif query_mode == "parallel_sampling":
        for round_idx in range(1, total_rounds + 1):
            prompt, prompt_tokens, used_context_chars = build_prompt_under_query_budget(
                generator.tokenizer,
                lambda ctx_chars, sample_idx=round_idx: build_parallel_sampling_messages(
                    example,
                    sample_idx,
                    total_rounds,
                    ctx_chars,
                ),
                context_max_chars=args.query_context_max_chars,
                max_input_tokens=args.query_max_input_tokens,
            )
            grounding, call = generate_query_from_messages(
                generator,
                "parallel_sampling",
                prompt,
                prompt_tokens,
                fallback=fallback,
                extra_fields={"used_context_max_chars": used_context_chars, "sample_idx": round_idx},
            )
            rounds.append(make_round_record(retriever, example, round_idx, grounding, args.top_k, [call]))

    elif query_mode == "decomposed_retrieval":
        prompt, prompt_tokens, used_context_chars = build_prompt_under_query_budget(
            generator.tokenizer,
            lambda ctx_chars: build_decomposed_retrieval_messages(example, total_rounds, ctx_chars),
            context_max_chars=args.query_context_max_chars,
            max_input_tokens=args.query_max_input_tokens,
        )
        raw_output = generator.generate_one(prompt)
        query_items, parse_ok = parse_decomposed_queries(raw_output, fallback, total_rounds)
        call = llm_call_record(
            "decomposed_retrieval",
            raw_output=raw_output,
            prompt_tokens=prompt_tokens,
            completion_tokens=generator.count_text_tokens(raw_output),
            parse_ok=parse_ok,
            backend=generator.backend,
            model_name=generator.model_name,
            extra_fields={"used_context_max_chars": used_context_chars},
        )
        for round_idx, query_item in enumerate(query_items, start=1):
            llm_calls = [call] if round_idx == 1 else []
            rounds.append(
                make_round_record(
                    retriever,
                    example,
                    round_idx,
                    query_item["query"],
                    args.top_k,
                    llm_calls,
                    extra_fields={"focus": query_item.get("focus", f"query_{round_idx}")},
                )
            )

    else:
        raise ValueError(f"Unsupported free-form comparison method: {query_mode}")

    total_llm_calls, total_prompt_tokens, total_completion_tokens = sum_llm_usage(rounds)
    return finalize_candidate_record(
        example,
        query_mode=query_mode,
        rounds=rounds,
        top_k=args.top_k,
        rrf_kappa=args.rrf_kappa,
        total_llm_calls=total_llm_calls,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        wall_time=time.monotonic() - start_time,
        extra_fields={"rrf_kappa": args.rrf_kappa},
    )


def candidate_reward(candidates: list[dict[str, Any]], gold_tags: list[str]) -> float:
    rank = first_gold_rank([candidate["tag"] for candidate in candidates], gold_tags)
    return 0.0 if rank is None else 1.0 / rank


def log_discount_reward(candidates: list[dict[str, Any]], gold_tags: list[str]) -> float:
    rank = first_gold_rank([candidate["tag"] for candidate in candidates], gold_tags)
    return 0.0 if rank is None else 1.0 / math.log2(rank + 1.0)


def pairwise_candidate_overlap(rounds: list[dict[str, Any]], top_k: int) -> float | None:
    if len(rounds) < 2:
        return None
    values = []
    for left_idx in range(len(rounds)):
        left = {candidate["tag"] for candidate in rounds[left_idx].get("candidates", [])[:top_k]}
        for right_idx in range(left_idx + 1, len(rounds)):
            right = {candidate["tag"] for candidate in rounds[right_idx].get("candidates", [])[:top_k]}
            union = left | right
            values.append(len(left & right) / len(union) if union else 0.0)
    return round(sum(values) / len(values), 6) if values else None


def coverage_after_rounds(rounds: list[dict[str, Any]], gold_tags: list[str]) -> dict[str, bool]:
    seen: set[str] = set()
    curve: dict[str, bool] = {}
    for idx, round_record in enumerate(rounds, start=1):
        seen.update(candidate["tag"] for candidate in round_record.get("candidates", []))
        curve[str(idx)] = any(tag in set(gold_tags) for tag in seen)
    return curve


def build_freeform_feedback_messages(
    example: Example,
    grounding: str,
    candidates: list[dict[str, Any]],
    context_max_chars: int,
    doc_max_chars: int,
    feedback_candidate_count: int,
) -> list[dict[str, str]]:
    evidence = serialize_evidence(example, context_max_chars)
    candidate_text = format_feedback_candidates(candidates, example.entity_type, feedback_candidate_count, doc_max_chars)
    user = f"""Critique a free-text grounding interpretation using the retrieved taxonomy neighborhood.

Return concise free text. Do not identify a gold concept. Explain what semantic direction looks supported, what looks mismatched, and what the next rewrite should attend to.

Evidence:
{evidence}

Current grounding:
{grounding}

Retrieved concepts:
{candidate_text}"""
    return [
        {"role": "system", "content": "You critique ICD-10-CM grounding interpretations using retrieved-neighborhood feedback."},
        {"role": "user", "content": user},
    ]


def build_freeform_revision_messages(
    example: Example,
    grounding: str,
    feedback_text: str,
    arm: str,
    previous_groundings: list[str],
    memories: list[dict[str, Any]],
    context_max_chars: int,
) -> list[dict[str, str]]:
    evidence = serialize_evidence(example, context_max_chars)
    previous = "\n".join(f"- {item}" for item in previous_groundings) or "None."
    memory_text = json.dumps(
        [
            {
                "arm": memory.get("arm"),
                "feedback": memory.get("feedback_text"),
                "semantic_difference": memory.get("semantic_difference"),
                "delta_reward": memory.get("delta_reward"),
            }
            for memory in memories
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    user = f"""Revise a free-text grounding interpretation for retrieving the correct ICD-10-CM diagnosis code.

Selected rewrite strategy: {arm}
Strategy instruction: {FREEFORM_REWRITE_INSTRUCTIONS.get(arm, FREEFORM_REWRITE_INSTRUCTIONS["freeform_revise"])}

Return JSON only with this schema:
{{"grounding": "revised free-text interpretation", "q_lab": "compact label-form query", "q_def": "definition-style retrieval sentence", "semantic_difference": "how this differs from the previous grounding"}}

Formatting:
- {Q_LAB_FORMATTING_INSTRUCTION}
- q_def should be one definition-style sentence expressing the same interpretation.

Evidence:
{evidence}

Current grounding:
{grounding}

Critique:
{feedback_text}

Previously tested groundings:
{previous}

Relevant contrastive memories:
{memory_text}

Rules:
- Apply the selected rewrite strategy.
- State how the new grounding differs from previously tested groundings.
- Do not name a specific ICD-10-CM code unless it is explicitly present in the source context.
- Do not include markdown."""
    return [
        {"role": "system", "content": "You revise ICD-10-CM grounding interpretations with generic rewrite strategies."},
        {"role": "user", "content": user},
    ]


def freeform_surface_features(example: Example, grounding: str, feedback_text: str, novelty: float, round_idx: int, prior_arms: list[str]) -> dict[str, float]:
    grounding_tokens = tokenize(grounding)
    feedback_tokens = set(tokenize(feedback_text))
    text = grounding.lower()
    aggregation_cues = {"total", "subtotal", "aggregate", "net", "gross", "average", "consolidated"}
    temporal_cues = {"current", "noncurrent", "year", "month", "quarter", "annual", "duration", "instant", "ended"}
    scope_cues = {"segment", "geography", "plan", "class", "subsidiary", "member"}
    return {
        "bias": 1.0,
        "is_table": 1.0 if example.input_type == "table" else 0.0,
        "is_text": 1.0 if example.input_type == "text" else 0.0,
        "is_monetary": 1.0 if "monetary" in example.entity_type.lower() else 0.0,
        "is_shares": 1.0 if "share" in example.entity_type.lower() else 0.0,
        "token_count_lt_12": 1.0 if len(grounding_tokens) < 12 else 0.0,
        "token_count_gt_30": 1.0 if len(grounding_tokens) > 30 else 0.0,
        "has_temporal_cue": 1.0 if any(cue in text for cue in temporal_cues) else 0.0,
        "has_aggregation_cue": 1.0 if any(cue in text for cue in aggregation_cues) else 0.0,
        "has_scope_cue": 1.0 if any(cue in text for cue in scope_cues) else 0.0,
        "critique_mismatch_flag": 1.0 if feedback_tokens & {"wrong", "mismatch", "mismatched", "different", "irrelevant"} else 0.0,
        "neighborhood_novelty": float(novelty),
        "round_idx": float(round_idx),
        "prior_arm_count": float(len(prior_arms)),
    }


class DiagonalLinTS:
    def __init__(self, arms: tuple[str, ...], feature_names: list[str], ridge: float, alpha: float, seed: int) -> None:
        self.arms = arms
        self.feature_names = feature_names
        self.ridge = ridge
        self.alpha = alpha
        self.rng = random.Random(seed)
        self.a = {arm: [ridge for _ in feature_names] for arm in arms}
        self.b = {arm: [0.0 for _ in feature_names] for arm in arms}
        self.n_updates = Counter()

    def vectorize(self, features: dict[str, float]) -> list[float]:
        return [float(features.get(name, 0.0)) for name in self.feature_names]

    def sample_score(self, arm: str, x: list[float]) -> float:
        score = 0.0
        for idx, value in enumerate(x):
            mean = self.b[arm][idx] / self.a[arm][idx]
            stdev = self.alpha / math.sqrt(self.a[arm][idx])
            score += self.rng.gauss(mean, stdev) * value
        return score

    def mean_score(self, arm: str, x: list[float]) -> float:
        return sum((self.b[arm][idx] / self.a[arm][idx]) * value for idx, value in enumerate(x))

    def update(self, arm: str, x: list[float], reward: float) -> None:
        for idx, value in enumerate(x):
            self.a[arm][idx] += value * value
            self.b[arm][idx] += reward * value
        self.n_updates[arm] += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            arm: {
                "mu": [round(self.b[arm][idx] / self.a[arm][idx], 8) for idx in range(len(self.feature_names))],
                "sigma_diag": [round(1.0 / self.a[arm][idx], 8) for idx in range(len(self.feature_names))],
                "n_updates": int(self.n_updates[arm]),
            }
            for arm in self.arms
        }


def memory_similarity(query_text: str, memory: dict[str, Any]) -> float:
    left = set(tokenize(query_text))
    right = set(tokenize(memory.get("search_text", "")))
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def retrieve_freeform_memories(
    memory_store: list[dict[str, Any]],
    example: Example,
    grounding: str,
    feedback_text: str,
    arm: str,
    top_k: int,
) -> list[dict[str, Any]]:
    current = normalize_space(f"{serialize_evidence(example, 1200)} {grounding} {feedback_text} {arm}")
    compatible = [
        memory
        for memory in memory_store
        if memory.get("input_type") == example.input_type
        and memory.get("type") == example.entity_type
        and memory.get("source_sample_idx") != example.source_sample_idx
        and memory.get("arm") == arm
    ]
    if not compatible:
        compatible = [
            memory
            for memory in memory_store
            if memory.get("input_type") == example.input_type
            and memory.get("source_sample_idx") != example.source_sample_idx
        ]
    compatible.sort(key=lambda memory: memory_similarity(current, memory), reverse=True)
    positives = [memory for memory in compatible if float(memory.get("delta_reward", 0.0)) > 0.0]
    negatives = [memory for memory in compatible if float(memory.get("delta_reward", 0.0)) <= 0.0]
    return (positives[:top_k] + negatives[:top_k])[: 2 * top_k]


def surface_overlap(left: dict[str, str], right: dict[str, str]) -> float:
    left_tokens = set(tokenize(f"{left.get('q_lab', '')} {left.get('q_def', '')}"))
    right_tokens = set(tokenize(f"{right.get('q_lab', '')} {right.get('q_def', '')}"))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def build_parallel_diversity_method_record(
    args: argparse.Namespace,
    generator: QueryGenerator,
    retriever: TaxonomyRetriever,
    example: Example,
) -> dict[str, Any]:
    start_time = time.monotonic()
    rounds: list[dict[str, Any]] = []
    prior_interpretations: list[str] = []
    fallback = build_direct_query(example)

    for round_idx in range(1, args.retrieval_rounds + 1):
        prompt, prompt_tokens, used_context_chars = build_prompt_under_query_budget(
            generator.tokenizer,
            lambda ctx_chars, prior=list(prior_interpretations): build_parallel_diversity_messages(
                example,
                prior,
                ctx_chars,
            ),
            context_max_chars=args.query_context_max_chars,
            max_input_tokens=args.query_max_input_tokens,
        )
        raw_output = generator.generate_one(prompt)
        surfaces, parse_ok = parse_grounding_surfaces(raw_output, fallback)
        call = llm_call_record(
            "parallel_sampling_diversity",
            raw_output=raw_output,
            prompt_tokens=prompt_tokens,
            completion_tokens=generator.count_text_tokens(raw_output),
            parse_ok=parse_ok,
            backend=generator.backend,
            model_name=generator.model_name,
            extra_fields={
                "used_context_max_chars": used_context_chars,
                "temperature": args.query_temperature,
                "prompt_id": "sequential_exclusion_v1",
                "prior_interpretations": list(prior_interpretations),
            },
        )
        round_record = retrieve_dual_observation(
            retriever,
            example,
            surfaces,
            round_idx,
            args.top_k,
            args.rrf_kappa,
            [call],
            extra_fields={
                "hypothesis_index": round_idx,
                "prior_interpretations": list(prior_interpretations),
                "temperature": args.query_temperature,
                "prompt_id": "sequential_exclusion_v1",
            },
        )
        rounds.append(round_record)
        prior_interpretations.append(surfaces["grounding"])

    total_llm_calls, total_prompt_tokens, total_completion_tokens = sum_llm_usage(rounds)
    return finalize_candidate_record(
        example,
        query_mode="parallel_sampling_diversity",
        rounds=rounds,
        top_k=args.top_k,
        rrf_kappa=args.rrf_kappa,
        total_llm_calls=total_llm_calls,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        wall_time=time.monotonic() - start_time,
        extra_fields={
            "rrf_kappa": args.rrf_kappa,
            "fusion_mode": "per_hypothesis_then_across",
            "generation_design": "sequential_exclusion_B_calls",
            "mean_pairwise_neighborhood_overlap_at_200": pairwise_candidate_overlap(rounds, args.top_k),
            "coverage_after_hypothesis": coverage_after_rounds(rounds, example.gold_tags),
            "temperature_source": "QUERY_TEMPERATURE; should match stochastic parallel sampling config",
        },
    )


def build_freeform_initial_surfaces(
    args: argparse.Namespace,
    generator: QueryGenerator,
    example: Example,
    initial_idx: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    fallback = build_direct_query(example)
    prompt, prompt_tokens, used_context_chars = build_prompt_under_query_budget(
        generator.tokenizer,
        lambda ctx_chars: build_dual_grounding_messages(
            example,
            ctx_chars,
            extra_instructions=f"\nThis is stochastic initial grounding {initial_idx}. Produce one plausible interpretation.",
        ),
        context_max_chars=args.query_context_max_chars,
        max_input_tokens=args.query_max_input_tokens,
    )
    raw_output = generator.generate_one(prompt)
    surfaces, parse_ok = parse_grounding_surfaces(raw_output, fallback)
    call = llm_call_record(
        "bandit_freeform_initial",
        raw_output=raw_output,
        prompt_tokens=prompt_tokens,
        completion_tokens=generator.count_text_tokens(raw_output),
        parse_ok=parse_ok,
        backend=generator.backend,
        model_name=generator.model_name,
        extra_fields={"used_context_max_chars": used_context_chars, "initial_idx": initial_idx},
    )
    return surfaces, call


def build_freeform_feedback(
    args: argparse.Namespace,
    generator: QueryGenerator,
    example: Example,
    grounding: str,
    candidates: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    prompt, prompt_tokens, used_context_chars, used_doc_chars = build_feedback_prompt_under_query_budget(
        generator.tokenizer,
        lambda ctx_chars, doc_chars: build_freeform_feedback_messages(
            example,
            grounding,
            candidates,
            ctx_chars,
            doc_chars,
            args.feedback_candidate_count,
        ),
        context_max_chars=args.query_context_max_chars,
        doc_max_chars=args.candidate_doc_max_chars,
        max_input_tokens=args.query_max_input_tokens,
    )
    raw_output = generator.generate_one(prompt)
    feedback_text = clean_model_text(raw_output)
    call = llm_call_record(
        "bandit_freeform_feedback",
        raw_output=raw_output,
        prompt_tokens=prompt_tokens,
        completion_tokens=generator.count_text_tokens(raw_output),
        parse_ok=bool(feedback_text),
        backend=generator.backend,
        model_name=generator.model_name,
        extra_fields={
            "used_context_max_chars": used_context_chars,
            "used_candidate_doc_max_chars": used_doc_chars,
        },
    )
    return feedback_text, call


def build_freeform_revision(
    args: argparse.Namespace,
    generator: QueryGenerator,
    example: Example,
    grounding: str,
    feedback_text: str,
    arm: str,
    previous_groundings: list[str],
    memories: list[dict[str, Any]],
) -> tuple[dict[str, str], str, dict[str, Any]]:
    fallback = grounding or build_direct_query(example)
    prompt, prompt_tokens, used_context_chars = build_prompt_under_query_budget(
        generator.tokenizer,
        lambda ctx_chars: build_freeform_revision_messages(
            example,
            grounding,
            feedback_text,
            arm,
            previous_groundings,
            memories,
            ctx_chars,
        ),
        context_max_chars=args.query_context_max_chars,
        max_input_tokens=args.query_max_input_tokens,
    )
    raw_output = generator.generate_one(prompt)
    surfaces, parse_ok = parse_grounding_surfaces(raw_output, fallback)
    parsed, _ = parse_json_object(raw_output)
    semantic_difference = scalar_text(parsed.get("semantic_difference")) if parsed else ""
    call = llm_call_record(
        "bandit_freeform_revision",
        raw_output=raw_output,
        prompt_tokens=prompt_tokens,
        completion_tokens=generator.count_text_tokens(raw_output),
        parse_ok=parse_ok,
        backend=generator.backend,
        model_name=generator.model_name,
        extra_fields={"used_context_max_chars": used_context_chars, "arm": arm},
    )
    return surfaces, semantic_difference, call


def build_bandit_freeform_method_record(
    args: argparse.Namespace,
    query_mode: str,
    generator: QueryGenerator,
    retriever: TaxonomyRetriever,
    example: Example,
    memory_store: list[dict[str, Any]],
    posterior: DiagonalLinTS,
) -> dict[str, Any]:
    start_time = time.monotonic()
    arms = FREEFORM_REWRITE_ARMS_10 if query_mode == "bandit_freeform_10arm" else FREEFORM_REWRITE_ARMS
    rounds: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    previous_groundings: list[str] = []
    executed_surfaces: list[dict[str, str]] = []

    for initial_idx in range(1, args.bandit_initial_groundings + 1):
        surfaces, call = build_freeform_initial_surfaces(args, generator, example, initial_idx)
        round_record = retrieve_dual_observation(
            retriever,
            example,
            surfaces,
            initial_idx,
            args.top_k,
            args.rrf_kappa,
            [call],
            extra_fields={"initial_grounding_idx": initial_idx, "phase": "initial"},
        )
        rounds.append(round_record)
        previous_groundings.append(surfaces["grounding"])
        executed_surfaces.append(surfaces)

    initial_scores = [mean_top_score(round_record["candidates"], args.feedback_candidate_count) for round_record in rounds]
    carried_idx = max(range(len(rounds)), key=lambda idx: (initial_scores[idx], -idx))
    current_surfaces = {
        "grounding": rounds[carried_idx]["grounding"],
        "q_lab": rounds[carried_idx]["q_lab"],
        "q_def": rounds[carried_idx]["q_def"],
    }
    prior_arms: list[str] = []

    for step_idx in range(1, args.retrieval_rounds):
        latest_union = fuse_round_candidates(
            [{"round": idx + 1, "candidates": round_record["candidates"]} for idx, round_record in enumerate(rounds)],
            args.top_k,
            args.rrf_kappa,
        )
        feedback_text, feedback_call = build_freeform_feedback(
            args,
            generator,
            example,
            current_surfaces["grounding"],
            latest_union,
        )
        novelty = neighborhood_novelty(latest_union, rounds)
        features = freeform_surface_features(
            example,
            current_surfaces["grounding"],
            feedback_text,
            novelty,
            step_idx,
            prior_arms,
        )
        x = posterior.vectorize(features)
        scores = {arm: posterior.sample_score(arm, x) for arm in arms}
        ranked_arms = sorted(arms, key=lambda arm: scores[arm], reverse=True)
        selected_arm = ranked_arms[0]
        runner_up_arm = ranked_arms[1] if len(ranked_arms) > 1 else None
        gate_rejections = 0
        selected_surfaces: dict[str, str] | None = None
        semantic_difference = ""
        revision_call: dict[str, Any] | None = None
        selected_round: dict[str, Any] | None = None

        for arm in ranked_arms[: max(args.bandit_max_gate_rejections + 1, 1)]:
            memories = retrieve_freeform_memories(
                memory_store,
                example,
                current_surfaces["grounding"],
                feedback_text,
                arm,
                args.memory_top_k,
            )
            candidate_surfaces, candidate_difference, candidate_call = build_freeform_revision(
                args,
                generator,
                example,
                current_surfaces["grounding"],
                feedback_text,
                arm,
                previous_groundings,
                memories,
            )
            overlap = max((surface_overlap(candidate_surfaces, seen) for seen in executed_surfaces), default=0.0)
            if overlap <= args.bandit_query_overlap_threshold:
                selected_arm = arm
                selected_surfaces = candidate_surfaces
                semantic_difference = candidate_difference
                revision_call = candidate_call
                break
            gate_rejections += 1

        if selected_surfaces is None or revision_call is None:
            break

        selected_round = retrieve_dual_observation(
            retriever,
            example,
            selected_surfaces,
            len(rounds) + 1,
            args.top_k,
            args.rrf_kappa,
            [feedback_call, revision_call],
            extra_fields={
                "phase": "bandit_revision",
                "step_idx": step_idx,
                "psi": features,
                "psi_feature_names": posterior.feature_names,
                "slate": list(arms),
                "arm_selected": selected_arm,
                "arm_runner_up": runner_up_arm,
                "selection_scores": {arm: round(scores[arm], 8) for arm in arms},
                "gate_rejections": gate_rejections,
                "novelty_n": novelty,
                "critique_text": feedback_text,
                "semantic_difference": semantic_difference,
                "reward_temporal": None,
                "reward_replay": None,
                "reward_final": None,
            },
        )
        before_reward = log_discount_reward(latest_union, example.gold_tags)
        after_union = fuse_round_candidates(
            [{"round": idx + 1, "candidates": round_record["candidates"]} for idx, round_record in enumerate(rounds + [selected_round])],
            args.top_k,
            args.rrf_kappa,
        )
        after_reward = log_discount_reward(after_union, example.gold_tags)
        delta_reward = after_reward - before_reward
        replay_delta = 0.0
        replay_round: dict[str, Any] | None = None
        if runner_up_arm and args.bandit_replay:
            memories = retrieve_freeform_memories(
                memory_store,
                example,
                current_surfaces["grounding"],
                feedback_text,
                runner_up_arm,
                args.memory_top_k,
            )
            replay_surfaces, replay_difference, replay_call = build_freeform_revision(
                args,
                generator,
                example,
                current_surfaces["grounding"],
                feedback_text,
                runner_up_arm,
                previous_groundings,
                memories,
            )
            replay_round = retrieve_dual_observation(
                retriever,
                example,
                replay_surfaces,
                len(rounds) + 1,
                args.top_k,
                args.rrf_kappa,
                [replay_call],
                extra_fields={
                    "phase": "counterfactual_replay",
                    "arm_runner_up": runner_up_arm,
                    "semantic_difference": replay_difference,
                },
            )
            replay_union = fuse_round_candidates(
                [{"round": idx + 1, "candidates": round_record["candidates"]} for idx, round_record in enumerate(rounds + [replay_round])],
                args.top_k,
                args.rrf_kappa,
            )
            replay_delta = log_discount_reward(replay_union, example.gold_tags) - before_reward

        reward_final = args.bandit_reward_alpha * delta_reward + (1.0 - args.bandit_reward_alpha) * (delta_reward - replay_delta)
        selected_round["reward_temporal"] = round(delta_reward, 8)
        selected_round["reward_replay"] = round(delta_reward - replay_delta, 8)
        selected_round["reward_final"] = round(reward_final, 8)
        if replay_round is not None:
            selected_round["counterfactual_replay_round"] = replay_round
        posterior.update(selected_arm, x, reward_final)
        transitions.append(
            {
                "round": selected_round["round"],
                "arm": selected_arm,
                "runner_up_arm": runner_up_arm,
                "psi": features,
                "feedback_text": feedback_text,
                "grounding_before": current_surfaces["grounding"],
                "grounding_after": selected_surfaces["grounding"],
                "semantic_difference": semantic_difference,
                "reward_before": round(before_reward, 8),
                "reward_after": round(after_reward, 8),
                "delta_reward": round(delta_reward, 8),
                "reward_final": round(reward_final, 8),
            }
        )
        memory_store.append(
            {
                "input_type": example.input_type,
                "type": example.entity_type,
                "source_sample_idx": example.source_sample_idx,
                "arm": selected_arm,
                "feedback_text": feedback_text,
                "semantic_difference": semantic_difference,
                "delta_reward": round(delta_reward, 8),
                "search_text": normalize_space(
                    f"{evidence_profile(example)} {current_surfaces['grounding']} {feedback_text} {selected_arm} {semantic_difference}"
                ),
            }
        )
        rounds.append(selected_round)
        previous_groundings.append(selected_surfaces["grounding"])
        executed_surfaces.append(selected_surfaces)
        current_surfaces = selected_surfaces
        prior_arms.append(selected_arm)

    total_llm_calls, total_prompt_tokens, total_completion_tokens = sum_llm_usage(rounds)
    return finalize_candidate_record(
        example,
        query_mode=query_mode,
        rounds=rounds,
        top_k=args.top_k,
        rrf_kappa=args.rrf_kappa,
        total_llm_calls=total_llm_calls,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        wall_time=time.monotonic() - start_time,
        extra_fields={
            "rrf_kappa": args.rrf_kappa,
            "bandit_freeform_manifest": {
                "arms": list(arms),
                "n_arms": len(arms),
                "initial_groundings": args.bandit_initial_groundings,
                "round_budget": args.retrieval_rounds,
                "renderer": "model_emitted_dual",
                "beta": 0.0,
                "use_memory": True,
                "use_replay": bool(args.bandit_replay),
                "use_admissibility": False,
                "use_factorized_state": False,
                "carry_forward_rule": "argmax mean top-10 fused score",
                "psi_feature_names": posterior.feature_names,
                "psi_dimensionality": len(posterior.feature_names),
            },
            "operator_transitions": transitions,
            "posterior_snapshot": posterior.snapshot(),
            "mean_pairwise_neighborhood_overlap_at_200": pairwise_candidate_overlap(rounds, args.top_k),
            "coverage_after_round": coverage_after_rounds(rounds, example.gold_tags),
        },
    )


def neighborhood_novelty(
    candidates: list[dict[str, Any]],
    previous_rounds: list[dict[str, Any]],
) -> float:
    current = {candidate["tag"] for candidate in candidates}
    if not current or not previous_rounds:
        return 1.0
    max_overlap = 0.0
    for round_record in previous_rounds:
        previous = {candidate["tag"] for candidate in round_record.get("candidates", [])}
        union = current | previous
        if union:
            max_overlap = max(max_overlap, len(current & previous) / len(union))
    return round(1.0 - max_overlap, 6)


def evidence_profile(example: Example) -> dict[str, Any]:
    return {
        "input_type": example.input_type,
        "datatype": example.entity_type,
        "has_row_context": bool(example.row_context),
        "has_column_context": bool(example.column_context),
        "row_context": truncate_text(example.row_context, 240),
        "column_context": truncate_text(example.column_context, 240),
    }


def transition_memory_text(transition: dict[str, Any]) -> str:
    return normalize_space(
        " ".join(
            [
                scalar_text(transition.get("evidence_profile")),
                scalar_text(transition.get("feedback")),
                scalar_text(transition.get("directive")),
                scalar_text(transition.get("hypothesis_before")),
            ]
        )
    )


def retrieve_operator_memories(
    memory_store: list[dict[str, Any]],
    example: Example,
    hypothesis: dict[str, Any],
    feedback: dict[str, Any],
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    current_text = normalize_space(
        " ".join(
            [
                serialize_evidence(example, 1200),
                scalar_text(hypothesis),
                scalar_text(feedback),
            ]
        )
    )
    current_tokens = set(tokenize(current_text))
    current_operators = set(hypothesis.get("operators", []))

    def compatible(memory: dict[str, Any]) -> bool:
        if memory.get("input_type") != example.input_type:
            return False
        if memory.get("type") and memory.get("type") != example.entity_type:
            return False
        operator = memory.get("operator")
        return not current_operators or not operator or operator in current_operators

    def similarity(memory: dict[str, Any]) -> float:
        memory_tokens = set(tokenize(memory.get("search_text", "")))
        if not current_tokens or not memory_tokens:
            return 0.0
        return len(current_tokens & memory_tokens) / len(current_tokens | memory_tokens)

    compatible_memories = [memory for memory in memory_store if compatible(memory)]
    if not compatible_memories:
        compatible_memories = [memory for memory in memory_store if memory.get("input_type") == example.input_type]
    if not compatible_memories:
        compatible_memories = list(memory_store)

    positives = [memory for memory in compatible_memories if float(memory.get("delta_reward", 0.0)) > 0.0]
    negatives = [memory for memory in compatible_memories if float(memory.get("delta_reward", 0.0)) <= 0.0]
    positives.sort(key=similarity, reverse=True)
    negatives.sort(key=similarity, reverse=True)
    return positives[:top_k], negatives[:top_k]


def build_operator_transition(
    example: Example,
    from_round: dict[str, Any],
    to_round: dict[str, Any],
    feedback: dict[str, Any],
    directive: dict[str, Any],
) -> dict[str, Any]:
    reward_before = candidate_reward(from_round.get("candidates", []), example.gold_tags)
    reward_after = candidate_reward(to_round.get("candidates", []), example.gold_tags)
    transition = {
        "from_round": from_round.get("round"),
        "to_round": to_round.get("round"),
        "evidence_profile": evidence_profile(example),
        "input_type": example.input_type,
        "type": example.entity_type,
        "hypothesis_before": from_round.get("hypothesis"),
        "feedback": feedback,
        "directive": directive,
        "operator": directive.get("operator"),
        "target_dimension": directive.get("target_dimension"),
        "revised_hypothesis": to_round.get("hypothesis"),
        "semantic_difference": to_round.get("semantic_difference"),
        "reward_before": round(reward_before, 8),
        "reward_after": round(reward_after, 8),
        "delta_reward": round(reward_after - reward_before, 8),
    }
    transition["search_text"] = transition_memory_text(transition)
    return transition


def append_memories_from_record(record: dict[str, Any], memory_store: list[dict[str, Any]]) -> None:
    for transition in record.get("operator_transitions", []):
        memory_store.append(dict(transition))


def build_structured_method_record(
    args: argparse.Namespace,
    query_mode: str,
    generator: QueryGenerator,
    retriever: TaxonomyRetriever,
    example: Example,
    memory_store: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start_time = time.monotonic()
    rounds: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    fallback = build_direct_query(example)

    prompt, prompt_tokens, used_context_chars = build_prompt_under_query_budget(
        generator.tokenizer,
        lambda ctx_chars: build_operator_initial_messages(example, ctx_chars),
        context_max_chars=args.query_context_max_chars,
        max_input_tokens=args.query_max_input_tokens,
    )
    raw_output = generator.generate_one(prompt)
    hypothesis, parse_ok = parse_hypothesis(raw_output, fallback)
    call = llm_call_record(
        "operator_initial_hypothesis",
        raw_output=raw_output,
        prompt_tokens=prompt_tokens,
        completion_tokens=generator.count_text_tokens(raw_output),
        parse_ok=parse_ok,
        backend=generator.backend,
        model_name=generator.model_name,
        extra_fields={"used_context_max_chars": used_context_chars},
    )
    rounds.append(
        make_round_record(
            retriever,
            example,
            1,
            hypothesis["retrieval_query"],
            args.top_k,
            [call],
            extra_fields={"hypothesis": hypothesis},
        )
    )

    for round_idx in range(2, args.retrieval_rounds + 1):
        current_round = rounds[-1]
        current_hypothesis = current_round.get("hypothesis", hypothesis)
        feedback_prompt, feedback_prompt_tokens, feedback_context_chars, feedback_doc_chars = build_feedback_prompt_under_query_budget(
            generator.tokenizer,
            lambda ctx_chars, doc_chars, current=current_hypothesis, candidates=current_round["candidates"]: build_operator_feedback_messages(
                example,
                current,
                candidates,
                ctx_chars,
                doc_chars,
                args.feedback_candidate_count,
            ),
            context_max_chars=args.query_context_max_chars,
            doc_max_chars=args.candidate_doc_max_chars,
            max_input_tokens=args.query_max_input_tokens,
        )
        feedback_raw = generator.generate_one(feedback_prompt)
        feedback, feedback_parse_ok = parse_feedback(feedback_raw)
        feedback_call = llm_call_record(
            "operator_feedback",
            raw_output=feedback_raw,
            prompt_tokens=feedback_prompt_tokens,
            completion_tokens=generator.count_text_tokens(feedback_raw),
            parse_ok=feedback_parse_ok,
            backend=generator.backend,
            model_name=generator.model_name,
            extra_fields={
                "used_context_max_chars": feedback_context_chars,
                "used_candidate_doc_max_chars": feedback_doc_chars,
            },
        )

        positive_memories = negative_memories = None
        if query_mode == "memory_guided_refinement":
            positive_memories, negative_memories = retrieve_operator_memories(
                memory_store or [],
                example,
                current_hypothesis,
                feedback,
                args.memory_top_k,
            )

        controller_prompt, controller_prompt_tokens, controller_context_chars = build_prompt_under_query_budget(
            generator.tokenizer,
            lambda ctx_chars: build_operator_controller_messages(
                example,
                current_hypothesis,
                feedback,
                transitions,
                positive_memories,
                negative_memories,
                ctx_chars,
            ),
            context_max_chars=args.query_context_max_chars,
            max_input_tokens=args.query_max_input_tokens,
        )
        controller_raw = generator.generate_one(controller_prompt)
        directive, directive_parse_ok = parse_directive(controller_raw)
        controller_call = llm_call_record(
            "operator_controller",
            raw_output=controller_raw,
            prompt_tokens=controller_prompt_tokens,
            completion_tokens=generator.count_text_tokens(controller_raw),
            parse_ok=directive_parse_ok,
            backend=generator.backend,
            model_name=generator.model_name,
            extra_fields={
                "used_context_max_chars": controller_context_chars,
                "positive_memory_count": len(positive_memories or []),
                "negative_memory_count": len(negative_memories or []),
            },
        )

        revision_prompt, revision_prompt_tokens, revision_context_chars = build_prompt_under_query_budget(
            generator.tokenizer,
            lambda ctx_chars: build_operator_revision_messages(example, current_hypothesis, directive, ctx_chars),
            context_max_chars=args.query_context_max_chars,
            max_input_tokens=args.query_max_input_tokens,
        )
        revision_raw = generator.generate_one(revision_prompt)
        revised_hypothesis, revision_parse_ok = parse_hypothesis(revision_raw, current_hypothesis.get("retrieval_query", fallback))
        revision_parsed, _ = parse_json_object(revision_raw)
        semantic_difference = scalar_text(revision_parsed.get("semantic_difference"))
        revision_call = llm_call_record(
            "operator_revision",
            raw_output=revision_raw,
            prompt_tokens=revision_prompt_tokens,
            completion_tokens=generator.count_text_tokens(revision_raw),
            parse_ok=revision_parse_ok,
            backend=generator.backend,
            model_name=generator.model_name,
            extra_fields={"used_context_max_chars": revision_context_chars},
        )

        new_round = make_round_record(
            retriever,
            example,
            round_idx,
            revised_hypothesis["retrieval_query"],
            args.top_k,
            [feedback_call, controller_call, revision_call],
            extra_fields={
                "hypothesis": revised_hypothesis,
                "feedback_from_previous": feedback,
                "directive_from_previous": directive,
                "semantic_difference": semantic_difference,
            },
        )
        new_round["neighborhood_novelty"] = neighborhood_novelty(new_round["candidates"], rounds)
        transition = build_operator_transition(example, current_round, new_round, feedback, directive)
        transitions.append(transition)
        rounds.append(new_round)

    total_llm_calls, total_prompt_tokens, total_completion_tokens = sum_llm_usage(rounds)
    return finalize_candidate_record(
        example,
        query_mode=query_mode,
        rounds=rounds,
        top_k=args.top_k,
        rrf_kappa=args.rrf_kappa,
        total_llm_calls=total_llm_calls,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        wall_time=time.monotonic() - start_time,
        extra_fields={
            "rrf_kappa": args.rrf_kappa,
            "operator_transitions": transitions,
        },
    )


def build_comparison_candidate_records(
    args: argparse.Namespace,
    examples: list[Example],
    taxonomy: list[Concept],
    trace_path: Path,
    tokenizer: Any | None = None,
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    query_mode = canonical_query_mode(args.query_mode)
    retriever = TaxonomyRetriever(taxonomy, type_filter=args.type_filter)

    # frozen_ags scores retrieval with label coverage active (w_cov=1.0) and gates on the
    # spec's startup assertions before any example is processed.
    frozen_ags_config = None
    frozen_ags_normalization_map = None
    ags_seq_config = None
    ags_seq_bank = None
    build_ags_seq_method_record = None
    if query_mode in AGS_SEQ_QUERY_MODES or query_mode in SEQ_VERIFIER_QUERY_MODES:
        # The sequential arms reuse frozen_ags's retrieval scaling and startup gate, then
        # add the controller loop on top; the posterior bank spans the whole stream.
        from ags_frozen_grounding import FrozenAgsConfig, frozen_ags_startup_assertions
        from ags_sequential_arms import (
            OPERATORS,
            AgsSeqConfig,
            SequentialPosteriorBank,
            build_ags_seq_method_record,
        )
        from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map

        frozen_ags_config = FrozenAgsConfig()
        retriever.label_coverage_weight = frozen_ags_config.label_coverage_weight
        retriever.label_coverage_pool_multiplier = args.label_coverage_pool_multiplier
        frozen_ags_normalization_map = load_normalization_map(
            args.normalization_map or DEFAULT_NORMALIZATION_MAP
        )
        assertion_report = frozen_ags_startup_assertions(
            retriever, taxonomy, frozen_ags_normalization_map, frozen_ags_config
        )
        print(f"{query_mode} startup assertions passed: {json.dumps(assertion_report)}", flush=True)
        ags_seq_config = AgsSeqConfig(
            frozen=frozen_ags_config,
            max_rounds=args.ags_seq_max_rounds,
            slate_limit=args.ags_seq_slate_limit,
            feedback_top_m=args.ags_seq_feedback_top_m,
            novelty_gate=args.ags_seq_novelty_gate,
            novelty_threshold=args.ags_seq_novelty_threshold,
            reward_alpha=args.ags_seq_reward_alpha,
            posterior_ridge=args.ags_seq_posterior_ridge,
            posterior_sigma=args.ags_seq_posterior_sigma,
            posterior_nu=args.ags_seq_posterior_nu,
            posterior_forgetting=args.ags_seq_posterior_forgetting,
            seed=args.ags_seq_seed,
        )
        ags_seq_bank = SequentialPosteriorBank(OPERATORS, ags_seq_config)

    if query_mode in FROZEN_AGS_FAMILY_QUERY_MODES:
        from ags_frozen_grounding import (
            FrozenAgsConfig,
            build_frozen_ags_method_record,
            fhs_j1_config,
            fhs_j3_wcov0_config,
            fhs_j3_wcov1_config,
            fhs_j4_wcov0_config,
            fhs_j4_wcov1_config,
            fhs_no_verifier_config,
            frozen_ags_startup_assertions,
            one_pass_structured_config,
            with_label_coverage_weight,
        )
        from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map

        # Same startup assertions for both: the one-pass baseline must prove the retriever,
        # taxonomy and vocabulary are in exactly the state AGS requires, or the "identical
        # except J=1" claim is unverified.
        if query_mode in ONE_PASS_STRUCTURED_QUERY_MODES:
            frozen_ags_config = one_pass_structured_config()
        elif query_mode in FHS_J1_QUERY_MODES:
            frozen_ags_config = fhs_j1_config()
        elif query_mode in FHS_NO_VERIFIER_QUERY_MODES:
            frozen_ags_config = fhs_no_verifier_config()
        elif query_mode == "fhs_j3_wcov0":
            frozen_ags_config = fhs_j3_wcov0_config()
        elif query_mode == "fhs_j3_wcov1":
            frozen_ags_config = fhs_j3_wcov1_config()
        elif query_mode == "fhs_j4_wcov0":
            frozen_ags_config = fhs_j4_wcov0_config()
        elif query_mode == "fhs_j4_wcov1":
            frozen_ags_config = fhs_j4_wcov1_config()
        else:
            frozen_ags_config = FrozenAgsConfig()
        if args.label_coverage_weight is not None:
            frozen_ags_config = with_label_coverage_weight(
                frozen_ags_config,
                args.label_coverage_weight,
            )
        retriever.label_coverage_weight = frozen_ags_config.label_coverage_weight
        retriever.label_coverage_pool_multiplier = args.label_coverage_pool_multiplier
        frozen_ags_normalization_map = load_normalization_map(
            args.normalization_map or DEFAULT_NORMALIZATION_MAP
        )
        assertion_report = frozen_ags_startup_assertions(
            retriever, taxonomy, frozen_ags_normalization_map, frozen_ags_config
        )
        print(
            f"{frozen_ags_config.variant} startup assertions passed: {json.dumps(assertion_report)}",
            flush=True,
        )

    # The coverage term is an index property the paper reports as applied to every method
    # (Appendix "Label-coverage diagnostics" measures its gain for raw-context and free-text
    # queries too), but the assignments above fire only for the frozen_ags family, so the
    # non-structured baselines silently ran at the constructor default w_cov=0.0. This override
    # applies to any query mode. Default None leaves every existing run's behaviour unchanged.
    if args.label_coverage_weight is not None:
        # Guard: the frozen family already ran frozen_ags_startup_assertions against the w_cov
        # its config pins. Overriding after that gate would run a configuration nothing asserted,
        # and ags_frozen_grounding requires w_cov > 0. Refuse rather than drift silently.
        if frozen_ags_config is not None and (
            args.label_coverage_weight != frozen_ags_config.label_coverage_weight
        ):
            raise SystemExit(
                f"--label-coverage-weight {args.label_coverage_weight} conflicts with "
            f"{frozen_ags_config.variant}'s pinned w_cov="
                f"{frozen_ags_config.label_coverage_weight}; the startup assertions already "
                "gated on the pinned value. Use this flag only for non-frozen query modes."
            )
        retriever.label_coverage_weight = args.label_coverage_weight
        retriever.label_coverage_pool_multiplier = args.label_coverage_pool_multiplier
        print(
            f"label coverage overridden: w_cov={retriever.label_coverage_weight} "
            f"pool_multiplier={retriever.label_coverage_pool_multiplier}",
            flush=True,
        )

    existing = load_existing_method_records(trace_path, query_mode) if args.resume else {}
    records: dict[int, dict[str, Any]] = {}
    missing_examples = [example for example in examples if example.example_idx not in existing]

    generator: QueryGenerator | None = None
    handle = None
    memory_store: list[dict[str, Any]] = []
    bandit_memory_store: list[dict[str, Any]] = []
    bandit_arms = FREEFORM_REWRITE_ARMS_10 if query_mode == "bandit_freeform_10arm" else FREEFORM_REWRITE_ARMS
    bandit_posterior = DiagonalLinTS(
        bandit_arms,
        FREEFORM_FEATURE_NAMES,
        ridge=args.bandit_posterior_ridge,
        alpha=args.bandit_posterior_alpha,
        seed=args.bandit_seed,
    )
    try:
        if missing_examples:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if args.resume and trace_path.exists() else "w"
            handle = trace_path.open(mode, encoding="utf-8")
            generator = QueryGenerator(args, tokenizer=tokenizer, llm=llm)

        for offset, example in enumerate(examples, start=1):
            if example.example_idx in existing:
                record = existing[example.example_idx]
                records[example.example_idx] = record
                if query_mode == "memory_guided_refinement":
                    append_memories_from_record(record, memory_store)
                if ags_seq_bank is not None:
                    # Replay the stored credits so a resumed run rebuilds the same bank.
                    ags_seq_bank.ingest_record(record)
                if query_mode in {"bandit_freeform", "bandit_freeform_10arm"}:
                    for transition in record.get("operator_transitions", []):
                        arm = transition.get("arm")
                        if arm in bandit_arms:
                            features = transition.get("psi", {})
                            bandit_posterior.update(
                                arm,
                                bandit_posterior.vectorize(features if isinstance(features, dict) else {}),
                                float(transition.get("reward_final", transition.get("delta_reward", 0.0)) or 0.0),
                            )
                            bandit_memory_store.append(
                                {
                                    "input_type": record.get("input_type"),
                                    "type": record.get("type"),
                                    "source_sample_idx": record.get("source_sample_idx"),
                                    "arm": arm,
                                    "feedback_text": transition.get("feedback_text", ""),
                                    "semantic_difference": transition.get("semantic_difference", ""),
                                    "delta_reward": transition.get("delta_reward", 0.0),
                                    "search_text": normalize_space(
                                        f"{record.get('input_type')} {record.get('type')} {transition.get('grounding_before', '')} {transition.get('feedback_text', '')} {arm}"
                                    ),
                                }
                            )
                continue

            if generator is None or handle is None:
                raise RuntimeError("Query generator was not initialized for missing comparison records.")

            if query_mode in {"intrinsic_self_refinement", "retrieval_feedback_refinement", "parallel_sampling", "decomposed_retrieval"}:
                record = build_freeform_method_record(args, query_mode, generator, retriever, example)
            elif query_mode == "parallel_sampling_diversity":
                record = build_parallel_diversity_method_record(args, generator, retriever, example)
            elif query_mode in {"bandit_freeform", "bandit_freeform_10arm"}:
                record = build_bandit_freeform_method_record(
                    args,
                    query_mode,
                    generator,
                    retriever,
                    example,
                    bandit_memory_store,
                    bandit_posterior,
                )
            elif query_mode in STRUCTURED_QUERY_MODES:
                record = build_structured_method_record(
                    args,
                    query_mode,
                    generator,
                    retriever,
                    example,
                    memory_store=memory_store,
                )
            elif query_mode in SEQ_VERIFIER_QUERY_MODES:
                from ags_seq_verifier_arm import build_seq_verifier_record

                record = build_seq_verifier_record(
                    args,
                    query_mode,
                    generator,
                    retriever,
                    example,
                    frozen_ags_normalization_map,
                    None,
                    cfg=ags_seq_config,
                )
            elif query_mode in AGS_SEQ_QUERY_MODES:
                record = build_ags_seq_method_record(
                    args,
                    query_mode,
                    generator,
                    retriever,
                    example,
                    frozen_ags_normalization_map,
                    ags_seq_bank,
                    cfg=ags_seq_config,
                )
            elif query_mode in FROZEN_AGS_FAMILY_QUERY_MODES:
                # one_pass_structured takes this exact branch with J=1/beta=0 -- reusing the
                # AGS record builder is what makes "AGS with J=1" true by construction.
                record = build_frozen_ags_method_record(
                    args,
                    generator,
                    retriever,
                    example,
                    frozen_ags_normalization_map,
                    cfg=frozen_ags_config,
                )
            else:
                raise ValueError(f"Unsupported comparison query_mode={query_mode}")

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records[example.example_idx] = record
            if query_mode == "memory_guided_refinement":
                append_memories_from_record(record, memory_store)
            if offset % args.log_every == 0 or offset == len(examples):
                print(f"Built {query_mode} candidates for {offset}/{len(examples)} examples")
    finally:
        if handle is not None:
            handle.close()
        if generator is not None:
            generator.close()

    if ags_seq_bank is not None:
        snapshot_path = trace_path.parent / f"{query_mode}_posteriors.json"
        snapshot_path.write_text(
            json.dumps(ags_seq_bank.snapshot(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Wrote {query_mode} posterior snapshot: {snapshot_path}", flush=True)

    return [records[example.example_idx] for example in examples if example.example_idx in records]


def prediction_from_raw_output(
    record: dict[str, Any],
    raw_output: str,
    prompt_tokens: int,
    used_context_chars: int,
    used_doc_chars: int,
    backend: str,
) -> dict[str, Any]:
    parsed, parse_ok = parse_json_object(raw_output)
    model_ranked, selected_tag = parse_ranked_candidate_tags(parsed, record["candidates"])
    final_ranking = composite_ranking(model_ranked, record["candidates"])

    return {
        "example_idx": int(record["example_idx"]),
        "context_id": record.get("context_id"),
        "source_sample_idx": record.get("source_sample_idx"),
        "input_type": record.get("input_type"),
        "entity": record.get("entity"),
        "type": record.get("type"),
        "gold_tags": record.get("gold_tags", []),
        "selected_tag": final_ranking[0] if final_ranking else selected_tag,
        "model_ranked_tags": model_ranked,
        "final_ranking": final_ranking,
        "parse_ok": parse_ok,
        "raw_output": raw_output,
        "prompt_tokens": prompt_tokens,
        "used_context_max_chars": used_context_chars,
        "used_candidate_doc_max_chars": used_doc_chars,
        "backend": backend,
        "retrieval_metrics": record.get("retrieval_metrics", {}),
        "search_coverage": record.get("search_coverage", False),
    }


def parse_ranked_candidate_tags(
    parsed: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[list[str], str | None]:
    by_index = {int(candidate["rank"]): candidate["tag"] for candidate in candidates}
    candidate_tags = {candidate["tag"] for candidate in candidates}
    ranked: list[str] = []

    for key in ("ranked_indices", "ranking", "ranked_candidate_indices"):
        value = parsed.get(key)
        if isinstance(value, list):
            for item in value:
                try:
                    tag = by_index.get(int(item))
                except (TypeError, ValueError):
                    tag = None
                if tag and tag not in ranked:
                    ranked.append(tag)

    for key in ("ranked_tags", "tags"):
        value = parsed.get(key)
        if isinstance(value, list):
            for item in value:
                tag = normalize_tag(item)
                if tag in candidate_tags and tag not in ranked:
                    ranked.append(tag)

    selected_tag: str | None = None
    try:
        selected_tag = by_index.get(int(parsed.get("selected_index")))
    except (TypeError, ValueError):
        selected_tag = None

    if selected_tag is None and parsed.get("selected_tag") is not None:
        candidate = normalize_tag(parsed.get("selected_tag"))
        if candidate in candidate_tags:
            selected_tag = candidate

    if selected_tag and selected_tag not in ranked:
        ranked.insert(0, selected_tag)
    elif selected_tag:
        ranked = [selected_tag] + [tag for tag in ranked if tag != selected_tag]

    return ranked, selected_tag


def composite_ranking(model_ranked: list[str], candidates: list[dict[str, Any]]) -> list[str]:
    candidate_order = [candidate["tag"] for candidate in candidates]
    candidate_set = set(candidate_order)
    ranking = []
    for tag in model_ranked:
        tag = normalize_tag(tag)
        if tag in candidate_set and tag not in ranking:
            ranking.append(tag)
    ranking.extend(tag for tag in candidate_order if tag not in ranking)
    return ranking


def rerank_records(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    output_path: Path,
    tokenizer: Any | None = None,
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    if args.rerank_backend == "vllm":
        return rerank_records_vllm(args, records, output_path, tokenizer, llm)
    return rerank_records_transformers(args, records, output_path)


def rerank_records_transformers(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    output_path: Path,
) -> list[dict[str, Any]]:
    tokenizer, model = load_rerank_model(
        args.rerank_model,
        bf16=args.bf16,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_predictions(output_path) if args.resume else {}
    predictions = dict(existing)

    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for offset, record in enumerate(records, start=1):
            example_idx = int(record["example_idx"])
            if example_idx in predictions:
                continue

            prompt, prompt_tokens, used_context_chars, used_doc_chars = build_prompt_under_token_budget(
                tokenizer,
                record,
                context_max_chars=args.context_max_chars,
                doc_max_chars=args.candidate_doc_max_chars,
                rerank_list_size=args.rerank_list_size,
                max_input_tokens=args.max_input_tokens,
            )
            raw_output = generate_text(
                tokenizer,
                model,
                prompt,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            prediction = prediction_from_raw_output(
                record,
                raw_output=raw_output,
                prompt_tokens=prompt_tokens,
                used_context_chars=used_context_chars,
                used_doc_chars=used_doc_chars,
                backend="transformers",
            )
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()
            predictions[example_idx] = prediction

            if offset % args.log_every == 0:
                print(f"Reranked {offset}/{len(records)} records")

    return [predictions[int(record["example_idx"])] for record in records if int(record["example_idx"]) in predictions]


def rerank_records_vllm(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    output_path: Path,
    tokenizer: Any | None = None,
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    loaded_here = tokenizer is None or llm is None
    if tokenizer is None or llm is None:
        tokenizer, llm = load_vllm_engine(args, args.rerank_model)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_predictions(output_path) if args.resume else {}
    predictions = dict(existing)

    pending = [record for record in records if int(record["example_idx"]) not in predictions]
    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for start in range(0, len(pending), args.vllm_batch_size):
            batch_records = pending[start : start + args.vllm_batch_size]
            prompts: list[str] = []
            prompt_meta: list[tuple[int, int, int]] = []
            for record in batch_records:
                prompt, prompt_tokens, used_context_chars, used_doc_chars = build_prompt_under_token_budget(
                    tokenizer,
                    record,
                    context_max_chars=args.context_max_chars,
                    doc_max_chars=args.candidate_doc_max_chars,
                    rerank_list_size=args.rerank_list_size,
                    max_input_tokens=args.max_input_tokens,
                )
                prompts.append(prompt)
                prompt_meta.append((prompt_tokens, used_context_chars, used_doc_chars))

            outputs = llm.generate(prompts, sampling_params)
            for record, output, (prompt_tokens, used_context_chars, used_doc_chars) in zip(
                batch_records,
                outputs,
                prompt_meta,
                strict=True,
            ):
                raw_output = output.outputs[0].text.strip() if output.outputs else ""
                prediction = prediction_from_raw_output(
                    record,
                    raw_output=raw_output,
                    prompt_tokens=prompt_tokens,
                    used_context_chars=used_context_chars,
                    used_doc_chars=used_doc_chars,
                    backend="vllm",
                )
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                predictions[int(record["example_idx"])] = prediction

            handle.flush()
            processed = min(start + len(batch_records), len(pending))
            if processed % args.log_every == 0 or processed == len(pending):
                print(f"Reranked {processed}/{len(pending)} pending records with vLLM")

    if loaded_here:
        release_model_handles(llm, tokenizer)

    return [predictions[int(record["example_idx"])] for record in records if int(record["example_idx"]) in predictions]


def evaluate_records(
    candidate_records: list[dict[str, Any]],
    rerank_predictions: list[dict[str, Any]] | None,
    top_k: int,
    taxonomy_tags: set[str],
) -> dict[str, Any]:
    metrics = evaluate_record_scope(candidate_records, rerank_predictions, top_k, taxonomy_tags)
    metrics["by_input_type"] = {}
    input_types = sorted({normalize_space(record.get("input_type")) for record in candidate_records})
    for input_type in input_types:
        if not input_type:
            continue
        subset = [
            record
            for record in candidate_records
            if normalize_space(record.get("input_type")) == input_type
        ]
        metrics["by_input_type"][input_type] = evaluate_record_scope(
            subset,
            rerank_predictions,
            top_k,
            taxonomy_tags,
        )
    return metrics


def evaluate_record_scope(
    candidate_records: list[dict[str, Any]],
    rerank_predictions: list[dict[str, Any]] | None,
    top_k: int,
    taxonomy_tags: set[str],
) -> dict[str, Any]:
    top_ks = (10, 50, top_k)
    retrieval_rows = [
        metric_row([candidate["tag"] for candidate in record["candidates"]], record["gold_tags"], top_ks)
        for record in candidate_records
    ]
    coverage_rows = [bool(record.get("search_coverage")) for record in candidate_records]
    metrics: dict[str, Any] = {
        "n_examples": len(candidate_records),
        "top_k": top_k,
        "gold_taxonomy_coverage": round(
            sum(
                all(tag in taxonomy_tags for tag in record.get("gold_tags", []))
                for record in candidate_records
            )
            / len(candidate_records),
            6,
        )
        if candidate_records
        else 0.0,
        "bm25_retrieval": aggregate_metric_rows(retrieval_rows, top_ks),
        "search_coverage": round(sum(coverage_rows) / len(coverage_rows), 6) if coverage_rows else 0.0,
    }
    metrics["bm25_retrieval"]["search_coverage"] = metrics["search_coverage"]

    if rerank_predictions is not None:
        allowed_examples = {int(record["example_idx"]) for record in candidate_records}
        scoped_predictions = [
            prediction
            for prediction in rerank_predictions
            if int(prediction.get("example_idx", -1)) in allowed_examples
        ]
        rerank_rows = [
            metric_row(prediction.get("final_ranking", []), prediction.get("gold_tags", []), top_ks)
            for prediction in scoped_predictions
        ]
        parse_rows = [bool(prediction.get("parse_ok")) for prediction in scoped_predictions]
        metrics["qwen_reranked"] = aggregate_metric_rows(rerank_rows, top_ks)
        metrics["qwen_reranked"]["parse_success_rate"] = (
            round(sum(parse_rows) / len(parse_rows), 6) if parse_rows else 0.0
        )
        metrics["qwen_reranked"]["search_coverage"] = metrics["search_coverage"]
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    parser.add_argument("--taxonomy-jsonl", type=Path, default=DEFAULT_TAXONOMY_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument(
        "--normalization-map",
        type=Path,
        default=None,
        help="Symbolic dimension vocabulary for frozen_ags agreement scoring "
        "(defaults to ags_symbolic_agreement.DEFAULT_NORMALIZATION_MAP).",
    )
    parser.add_argument(
        "--label-coverage-pool-multiplier",
        type=int,
        default=0,
        help="frozen_ags coverage rescoring pool; <=0 scores the full type-filtered pool.",
    )
    parser.add_argument(
        "--label-coverage-weight",
        type=float,
        default=None,
        help=(
            "Force w_cov for ANY query mode. Without it only the frozen_ags family enables "
            "the coverage term and every other baseline runs at w_cov=0.0."
        ),
    )
    parser.add_argument(
        "--frozen-ags-top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling top-p for frozen_ags hypothesis generation.",
    )
    parser.add_argument(
        "--retrieval-rounds",
        type=int,
        default=4,
        help="Maximum retrieval calls for multi-round comparison methods.",
    )
    parser.add_argument(
        "--feedback-candidate-count",
        type=int,
        default=10,
        help="Number of retrieved candidates shown in feedback prompts.",
    )
    parser.add_argument(
        "--rrf-kappa",
        type=float,
        default=60.0,
        help="RRF smoothing constant for multi-round candidate fusion.",
    )
    parser.add_argument(
        "--memory-top-k",
        type=int,
        default=3,
        help="Positive and negative operator memories shown for memory-guided refinement.",
    )
    parser.add_argument("--bandit-initial-groundings", type=int, default=3)
    parser.add_argument("--bandit-posterior-ridge", type=float, default=1.0)
    parser.add_argument("--bandit-posterior-alpha", type=float, default=0.75)
    parser.add_argument("--bandit-seed", type=int, default=20260728)
    parser.add_argument("--bandit-query-overlap-threshold", type=float, default=0.85)
    parser.add_argument("--bandit-max-gate-rejections", type=int, default=2)
    parser.add_argument("--bandit-replay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bandit-reward-alpha", type=float, default=0.5)
    # Sequential control arms (ags_seq / ags_seq_random); see ags_seq_arms_spec.md.
    parser.add_argument("--ags-seq-max-rounds", type=int, default=4, help="B, counting AGS's round one.")
    parser.add_argument("--ags-seq-slate-limit", type=int, default=6, help="L, directives per round.")
    parser.add_argument("--ags-seq-feedback-top-m", type=int, default=10)
    parser.add_argument(
        "--ags-seq-novelty-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pre-retrieval token-Jaccard gate. Off by default (spec section 3): with it on "
        "the arm terminates early and never spends its own budget.",
    )
    parser.add_argument("--ags-seq-novelty-threshold", type=float, default=0.02)
    parser.add_argument("--ags-seq-reward-alpha", type=float, default=0.5)
    parser.add_argument("--ags-seq-posterior-ridge", type=float, default=1.0)
    parser.add_argument("--ags-seq-posterior-sigma", type=float, default=1.0)
    parser.add_argument("--ags-seq-posterior-nu", type=float, default=0.75)
    parser.add_argument("--ags-seq-posterior-forgetting", type=float, default=0.995)
    parser.add_argument("--ags-seq-seed", type=int, default=20260724)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument(
        "--query-mode",
        choices=sorted(QUERY_MODE_ALIASES),
        default="direct_retrieval",
        help=(
            "Retrieval method. Public names include direct_retrieval, one_pass_grounding, "
            "intrinsic_self_refinement, retrieval_feedback_refinement, parallel_sampling, "
            "parallel_sampling_diversity, decomposed_retrieval, operator_refinement, "
            "memory_guided_refinement, bandit_freeform, bandit_freeform_10arm, frozen_ags, "
            "fhs_j1, fhs_no_verifier, fhs_j3_wcov0, fhs_j3_wcov1, fhs_j4_wcov0, "
            "fhs_j4_wcov1, one_pass_structured (= AGS with J=1; alias ags_j1), "
            "ags_seq, ags_seq_random, and gold_label_definition_retrieval."
        ),
    )
    parser.add_argument(
        "--reuse-candidates",
        action="store_true",
        help="Reuse output-dir/bm25_candidates.jsonl if it already exists.",
    )
    parser.add_argument("--type-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-rerank", action="store_true", help="Run Qwen reranking after retrieval.")
    parser.add_argument("--rerank-model", default="Qwen/Qwen3-32B")
    parser.add_argument(
        "--rerank-backend",
        choices=["vllm", "transformers"],
        default="vllm",
        help="Generation backend for reranking. vLLM batches prompts and is the default.",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--query-generation-model", default="Qwen/Qwen3-32B")
    parser.add_argument(
        "--query-generation-backend",
        choices=["vllm", "transformers"],
        default="vllm",
    )
    parser.add_argument("--query-description-path", type=Path, default=None)
    parser.add_argument("--query-context-max-chars", type=int, default=12000)
    parser.add_argument("--query-max-input-tokens", type=int, default=16000)
    parser.add_argument("--query-max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--fhs-verifier-max-new-tokens",
        type=int,
        default=3072,
        help="Generation budget for frozen_ags six-dimensional FHS candidate verifier.",
    )
    parser.add_argument("--query-temperature", type=float, default=0.0)
    parser.add_argument("--query-top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--vllm-batch-size", type=int, default=32)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--context-max-chars", type=int, default=12000)
    parser.add_argument("--candidate-doc-max-chars", type=int, default=320)
    parser.add_argument("--rerank-list-size", type=int, default=20)
    parser.add_argument("--max-input-tokens", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.query_mode = canonical_query_mode(args.query_mode)
    run_final_listwise_rerank = args.run_rerank
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = args.output_dir / "bm25_candidates.jsonl"
    query_description_path = args.query_description_path or (args.output_dir / "query_descriptions.jsonl")
    grounding_trace_path = args.output_dir / "grounding_traces.jsonl"
    taxonomy = load_taxonomy(args.taxonomy_jsonl)
    query_description_records: list[dict[str, Any]] | None = None
    shared_tokenizer = None
    shared_llm = None

    if args.reuse_candidates and candidates_path.exists():
        candidate_records = load_jsonl(candidates_path)
        if args.limit is not None:
            candidate_records = candidate_records[: args.limit]
        validate_candidate_records(candidate_records, args.query_mode, args.top_k)
        print(f"Reused BM25 candidates from: {candidates_path}")
    else:
        examples = load_examples(args.test_jsonl, limit=args.limit)
        query_descriptions: dict[int, str] | None = None
        if args.query_mode == "one_pass_grounding":
            can_share_vllm = (
                run_final_listwise_rerank
                and args.query_generation_backend == "vllm"
                and args.rerank_backend == "vllm"
                and args.query_generation_model == args.rerank_model
            )
            existing_query_records = (
                load_existing_query_descriptions(query_description_path)
                if args.resume and query_description_path.exists()
                else {}
            )
            missing_query_count = sum(
                1 for example in examples if example.example_idx not in existing_query_records
            )
            if can_share_vllm and missing_query_count:
                shared_tokenizer, shared_llm = load_vllm_engine(args, args.query_generation_model)
            query_description_records = generate_query_descriptions(
                args,
                examples,
                query_description_path,
                tokenizer=shared_tokenizer,
                llm=shared_llm,
            )
            query_descriptions = {
                int(record["example_idx"]): normalize_space(record.get("query_description"))
                for record in query_description_records
            }

        if args.query_mode in MULTI_ROUND_QUERY_MODES:
            can_share_vllm = (
                run_final_listwise_rerank
                and args.query_generation_backend == "vllm"
                and args.rerank_backend == "vllm"
                and args.query_generation_model == args.rerank_model
            )
            existing_method_records = (
                load_existing_method_records(grounding_trace_path, args.query_mode)
                if args.resume and grounding_trace_path.exists()
                else {}
            )
            missing_method_count = sum(
                1 for example in examples if example.example_idx not in existing_method_records
            )
            if can_share_vllm and missing_method_count and shared_llm is None:
                shared_tokenizer, shared_llm = load_vllm_engine(args, args.query_generation_model)
            candidate_records = build_comparison_candidate_records(
                args,
                examples,
                taxonomy,
                grounding_trace_path,
                tokenizer=shared_tokenizer,
                llm=shared_llm,
            )
        else:
            candidate_records = build_candidate_records(
                examples,
                taxonomy,
                top_k=args.top_k,
                type_filter=args.type_filter,
                query_mode=args.query_mode,
                rrf_kappa=args.rrf_kappa,
                query_descriptions=query_descriptions,
                **(
                    {
                        "label_coverage_weight": args.label_coverage_weight,
                        "label_coverage_pool_multiplier": args.label_coverage_pool_multiplier,
                    }
                    if args.label_coverage_weight is not None
                    else {}
                ),
            )
        write_jsonl(candidates_path, candidate_records)

    rerank_predictions = None
    rerank_path = args.output_dir / "qwen_rerank_predictions.jsonl"
    if run_final_listwise_rerank:
        rerank_predictions = rerank_records(
            args,
            candidate_records,
            rerank_path,
            tokenizer=shared_tokenizer,
            llm=shared_llm,
        )

    if shared_llm is not None or shared_tokenizer is not None:
        release_model_handles(shared_llm, shared_tokenizer)
        shared_llm = None
        shared_tokenizer = None

    if query_description_records is None and args.query_mode == "one_pass_grounding" and query_description_path.exists():
        query_description_records = list(load_existing_query_descriptions(query_description_path).values())
        if args.limit is not None:
            allowed = {int(record["example_idx"]) for record in candidate_records}
            query_description_records = [
                record for record in query_description_records if int(record["example_idx"]) in allowed
            ]

    metrics = evaluate_records(
        candidate_records,
        rerank_predictions,
        top_k=args.top_k,
        taxonomy_tags={concept.tag for concept in taxonomy},
    )
    metrics.update(
        {
            "test_jsonl": str(args.test_jsonl),
            "taxonomy_jsonl": str(args.taxonomy_jsonl),
            "candidate_path": str(candidates_path),
            "query_mode": args.query_mode,
            "retrieval_rounds": args.retrieval_rounds if args.query_mode in MULTI_ROUND_QUERY_MODES else 1,
            "feedback_candidate_count": args.feedback_candidate_count
            if args.query_mode in {"retrieval_feedback_refinement", "operator_refinement", "memory_guided_refinement"}
            else None,
            "rrf_kappa": args.rrf_kappa,
            "grounding_trace_path": str(grounding_trace_path) if args.query_mode in MULTI_ROUND_QUERY_MODES else None,
            "query_description_path": str(query_description_path) if args.query_mode == "one_pass_grounding" else None,
            "query_generation_model": args.query_generation_model if args.query_mode in LLM_QUERY_MODES else None,
            "query_generation_backend": args.query_generation_backend if args.query_mode in LLM_QUERY_MODES else None,
            "query_generation_parse_success_rate": (
                round(
                    sum(bool(record.get("parse_ok")) for record in query_description_records)
                    / len(query_description_records),
                    6,
                )
                if query_description_records
                else parse_success_rate_from_records(candidate_records)
            ),
            "rerank_prediction_path": str(rerank_path) if run_final_listwise_rerank else None,
            "final_listwise_rerank_skipped": bool(args.run_rerank and not run_final_listwise_rerank),
            "type_filter": args.type_filter,
            "rerank_model": args.rerank_model if run_final_listwise_rerank else None,
            "rerank_backend": args.rerank_backend if run_final_listwise_rerank else None,
        }
    )
    metrics_path = args.output_dir / ("metrics.json" if run_final_listwise_rerank else "bm25_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
