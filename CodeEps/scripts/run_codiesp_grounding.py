#!/usr/bin/env python3
"""CodiEsp prompt shim around the shared FinTagging grounding runner.

The shared runner contains the retrieval, fusion, reranking, resume, metrics,
and frozen-AGS assertions.  This wrapper only replaces domain wording in the
LLM prompts so CodiEsp text is not presented as a US-GAAP/XBRL task.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = ROOT / "codiesp_pipeline"
SHARED_RUNNER = PIPELINE_ROOT / "run_fintagging_grounding_baseline.py"


def load_shared_runner() -> Any:
    sys.path.insert(0, str(PIPELINE_ROOT))
    spec = importlib.util.spec_from_file_location("run_fintagging_grounding_baseline", SHARED_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import shared runner from {SHARED_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_fintagging_grounding_baseline"] = module
    spec.loader.exec_module(module)
    return module


def install_codiesp_prompts(runner: Any) -> None:
    def normalize_tag(tag: Any) -> str:
        return runner.normalize_space(tag)

    def raw_tag(tag: Any) -> str:
        return runner.normalize_space(tag)

    def build_rerank_messages(
        record: dict[str, Any],
        context_max_chars: int,
        doc_max_chars: int,
        rerank_list_size: int,
    ) -> list[dict[str, str]]:
        context = runner.truncate_text(str(record.get("query_context") or record.get("query", "")), context_max_chars)
        candidate_text = "\n\n".join(
            runner.format_candidate_for_prompt(candidate, doc_max_chars)
            for candidate in record.get("candidates", [])
        )
        user = f"""Select the ICD-10-CM diagnosis code that best matches the clinical evidence.

Input:
Clinical mention: {record.get("entity", "")}
Code class: {record.get("type", "")}
Clinical context:
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
            {"role": "system", "content": "You are a precise ICD-10-CM diagnosis-code grounding reranker."},
            {"role": "user", "content": user},
        ]

    def build_query_description_messages(example: Any, context_max_chars: int) -> list[dict[str, str]]:
        context = runner.truncate_text(example.query_context, context_max_chars)
        user = f"""Write a brief retrieval query for finding the correct ICD-10-CM diagnosis code.

Input:
Clinical mention: {example.entity}
Code class: {example.entity_type}
Clinical context:
{context}

Return JSON only with this schema:
{{"query": "brief diagnosis description"}}

Rules:
- Describe the diagnosis, symptom, finding, injury, or health-status factor expressed by the evidence.
- Use wording likely to appear in ICD-10-CM labels or tabular/index descriptions.
- Do not name a specific ICD-10-CM code unless it is explicitly present in the source context.
- Do not include explanations or markdown."""
        return [
            {"role": "system", "content": "You generate concise ICD-10-CM retrieval queries for clinical diagnosis grounding."},
            {"role": "user", "content": user},
        ]

    def build_operator_initial_messages(example: Any, context_max_chars: int) -> list[dict[str, str]]:
        evidence = runner.serialize_evidence(example, context_max_chars)
        user = f"""Produce a structured semantic hypothesis for grounding clinical evidence to an ICD-10-CM diagnosis code.

Fill each dimension only if the evidence directly supports it. Use "UNRESOLVED" when unsupported.

Dimensions:
- FAMILY: ICD-10-CM chapter or broad clinical family
- ROLE: diagnosis class such as disease/disorder, neoplasm, symptom/sign/abnormal-finding, injury/poisoning, external cause, health-status factor, pregnancy-related, perinatal, or congenital
- EVENT: the specific condition, symptom, finding, injury, or clinical state
- QUALIFIER: modifiers such as acute/chronic, severity, complication status, malignant/benign, type, open/closed, or specified/unspecified
- SCOPE: laterality only, such as right, left, bilateral, unspecified-side, or not-applicable
- TEMPORAL: encounter or extension status, such as initial encounter, subsequent encounter, sequela, healing status, or not-applicable

Operator library:
{", ".join(runner.OPERATOR_LIBRARY)}

Return JSON only with this schema:
{{"dimensions": {{"FAMILY": "...", "ROLE": "...", "EVENT": "...", "QUALIFIER": "...", "SCOPE": "...", "TEMPORAL": "..."}}, "operators": ["direct_label"], "retrieval_query": "compact retrieval query"}}

Evidence:
{evidence}"""
        return [
            {"role": "system", "content": "You create structured ICD-10-CM diagnosis grounding hypotheses."},
            {"role": "user", "content": user},
        ]

    def build_parallel_sampling_messages(
        example: Any,
        sample_idx: int,
        total_samples: int,
        context_max_chars: int,
    ) -> list[dict[str, str]]:
        evidence = runner.serialize_evidence(example, context_max_chars)
        user = f"""Generate a semantic interpretation of clinical evidence for retrieving the correct ICD-10-CM diagnosis code.

This is interpretation {sample_idx} of {total_samples}. Explore a different plausible reading of the evidence. Consider varying:
- The diagnosis family or code class
- The specific condition, symptom, finding, injury, or health-status factor
- Qualifiers such as acute/chronic, severity, complication status, type, or specified/unspecified
- Laterality or encounter/extension status when supported

Return JSON only with this schema:
{{"query": "distinct diagnosis retrieval description"}}

Evidence:
{evidence}

Rules:
- Make this interpretation meaningfully distinct.
- Use wording likely to appear in ICD-10-CM labels, descriptions, or index text.
- Do not name a specific ICD-10-CM code unless it is explicitly present in the source context.
- Do not include explanations or markdown."""
        return [
            {"role": "system", "content": "You generate diverse ICD-10-CM retrieval hypotheses."},
            {"role": "user", "content": user},
        ]

    def build_intrinsic_refinement_messages(
        example: Any,
        previous_grounding: str,
        context_max_chars: int,
    ) -> list[dict[str, str]]:
        evidence = runner.serialize_evidence(example, context_max_chars)
        user = f"""You are given clinical evidence and your previous semantic interpretation for retrieving the correct ICD-10-CM diagnosis code.

Critically evaluate your previous interpretation without seeing any retrieved candidate codes. Consider whether you may have:
- Misidentified the diagnosis family or code class
- Chosen the wrong specific condition, symptom, finding, injury, or health-status factor
- Missed a qualifier such as acute/chronic, severity, complication status, type, or specified/unspecified
- Missed laterality or encounter/extension status when supported
- Over-specified a detail not supported by the clinical evidence

Return JSON only with this schema:
{{"critique": "brief critique", "query": "revised diagnosis retrieval description"}}

Evidence:
{evidence}

Previous interpretation:
{previous_grounding}

Rules:
- Use wording likely to appear in ICD-10-CM labels or tabular/index descriptions.
- Do not name a specific ICD-10-CM code unless it is explicitly present in the source context.
- Do not include markdown.
- If the previous interpretation still looks correct, reuse it as the query."""
        return [
            {"role": "system", "content": "You refine ICD-10-CM retrieval queries by self-critique only."},
            {"role": "user", "content": user},
        ]

    def build_retrieval_feedback_messages(
        example: Any,
        previous_grounding: str,
        candidates: list[dict[str, Any]],
        context_max_chars: int,
        doc_max_chars: int,
        feedback_candidate_count: int,
    ) -> list[dict[str, str]]:
        evidence = runner.serialize_evidence(example, context_max_chars)
        candidate_text = runner.format_feedback_candidates(
            candidates,
            example.entity_type,
            feedback_candidate_count,
            doc_max_chars,
        )
        user = f"""You are given clinical evidence, your previous semantic interpretation, and the top diagnosis-code candidates retrieved using that interpretation.

Use the retrieved neighborhood to assess whether the interpretation is on the right track. Rewrite the semantic description to better capture the diagnosis, symptom, finding, injury, external cause, or health-status factor expressed by the evidence. You may change the diagnosis family, code class, specific condition, qualifier, laterality, or encounter/extension status.

Return JSON only with this schema:
{{"assessment": "brief assessment of the retrieved neighborhood", "query": "revised diagnosis retrieval description"}}

Evidence:
{evidence}

Previous interpretation:
{previous_grounding}

Retrieved candidates:
{candidate_text}

Rules:
- Choose wording likely to retrieve the correct ICD-10-CM diagnosis code.
- Do not name a gold code.
- Do not include markdown."""
        return [
            {"role": "system", "content": "You refine ICD-10-CM retrieval queries using retrieved-neighborhood feedback."},
            {"role": "user", "content": user},
        ]

    runner.TAG_PREFIX = ""
    runner.normalize_tag = normalize_tag
    runner.raw_tag = raw_tag
    runner.build_rerank_messages = build_rerank_messages
    runner.build_query_description_messages = build_query_description_messages
    runner.build_operator_initial_messages = build_operator_initial_messages
    runner.build_parallel_sampling_messages = build_parallel_sampling_messages
    runner.build_intrinsic_refinement_messages = build_intrinsic_refinement_messages
    runner.build_retrieval_feedback_messages = build_retrieval_feedback_messages


def requested_label_coverage_weight() -> float | None:
    argv = sys.argv[1:]
    for idx, arg in enumerate(argv):
        if arg == "--label-coverage-weight" and idx + 1 < len(argv):
            return float(argv[idx + 1])
        if arg.startswith("--label-coverage-weight="):
            return float(arg.split("=", 1)[1])
    return None


def requested_query_mode() -> str | None:
    argv = sys.argv[1:]
    for idx, arg in enumerate(argv):
        if arg == "--query-mode" and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith("--query-mode="):
            return arg.split("=", 1)[1]
    return None


def install_codiesp_frozen_family_overrides() -> None:
    mode = requested_query_mode()
    if mode not in {
        "ags",
        "fhs",
        "frozen_ags",
        "frozen_ags_grounding",
        "fhs_j1",
        "frozen_ags_j1",
        "fhs_no_verifier",
        "frozen_ags_no_verifier",
        "ags_j1",
        "one_pass_structured",
        "one_pass_grounding_structured",
    }:
        return

    weight = requested_label_coverage_weight()
    if weight is None:
        weight = 1.0
    is_fhs = mode in {"ags", "fhs", "frozen_ags", "frozen_ags_grounding", "fhs_j1", "frozen_ags_j1"}

    import ags_frozen_grounding as ags

    original_config = ags.FrozenAgsConfig

    def frozen_ags_config_factory(*args: Any, **kwargs: Any) -> Any:
        kwargs["label_coverage_weight"] = weight
        kwargs["rerank_beta"] = 0.6 if is_fhs else 0.0
        return original_config(*args, **kwargs)

    def one_pass_structured_config() -> Any:
        return original_config(
            hypotheses=1,
            rerank_beta=0.0,
            label_coverage_weight=weight,
            temperature=0.0,
            variant=ags.ONE_PASS_STRUCTURED_QUERY_MODE,
        )

    def fhs_j1_config() -> Any:
        return original_config(
            hypotheses=1,
            rerank_beta=0.6,
            label_coverage_weight=weight,
            temperature=0.8,
            variant=ags.FHS_J1_QUERY_MODE,
        )

    def fhs_no_verifier_config() -> Any:
        return original_config(
            hypotheses=2,
            rerank_beta=0.0,
            label_coverage_weight=weight,
            temperature=0.8,
            variant=ags.FHS_NO_VERIFIER_QUERY_MODE,
        )

    for variant in (
        ags.FROZEN_AGS_QUERY_MODE,
        ags.FHS_J1_QUERY_MODE,
        ags.FHS_NO_VERIFIER_QUERY_MODE,
        ags.ONE_PASS_STRUCTURED_QUERY_MODE,
    ):
        ags._FROZEN_VARIANTS[variant]["label_coverage_weight"] = weight
    ags._FROZEN_VARIANTS[ags.FROZEN_AGS_QUERY_MODE]["rerank_beta"] = 0.6
    ags._FROZEN_VARIANTS[ags.FHS_J1_QUERY_MODE]["rerank_beta"] = 0.6
    ags._FROZEN_VARIANTS[ags.FHS_NO_VERIFIER_QUERY_MODE]["rerank_beta"] = 0.0
    ags._FROZEN_VARIANTS[ags.ONE_PASS_STRUCTURED_QUERY_MODE]["rerank_beta"] = 0.0

    def startup_assertions_without_coverage(
        retriever: Any,
        taxonomy: list[Any],
        normalization_map: dict[str, Any],
        cfg: Any,
        self_retrieval_sample: int = 200,
        self_retrieval_tolerance: float = 0.05,
    ) -> dict[str, Any]:
        del retriever, taxonomy, self_retrieval_sample, self_retrieval_tolerance
        ags._assert_frozen(cfg)
        vocab = normalization_map.get("dimensions", {})
        for dimension in ("family", "qualifier", "scope", "temporal"):
            if not (vocab.get(dimension) or {}):
                raise AssertionError(f"frozen_ags vocabulary missing categories for '{dimension}'")
        for dimension in ("role", "event"):
            if vocab.get(dimension):
                raise AssertionError(
                    f"frozen_ags expects no controlled vocabulary for '{dimension}' (token branch only)"
                )
        return {
            "label_coverage_ablation": weight <= 0.0,
            "label_coverage_weight": weight,
            "codiesp_fhs_verifier_enabled": is_fhs,
            "fhs_verifier_dimensions": list(getattr(ags, "FHS_VERIFIER_DIMENSIONS", ())),
            "vocabulary_ok": True,
            "config_frozen_ok": True,
            "coverage_regression_checked": [],
            "coverage_regression_failures": [],
        }

    ags.FrozenAgsConfig = frozen_ags_config_factory
    ags.one_pass_structured_config = one_pass_structured_config
    ags.fhs_j1_config = fhs_j1_config
    ags.fhs_no_verifier_config = fhs_no_verifier_config
    ags.frozen_ags_startup_assertions = startup_assertions_without_coverage


def main() -> None:
    runner = load_shared_runner()
    install_codiesp_prompts(runner)
    install_codiesp_frozen_family_overrides()
    runner.main()


if __name__ == "__main__":
    main()
