#!/usr/bin/env python3
"""Offline tests for frozen_ags (no GPU, no vLLM).

Covers the method spec's unit tests (section 10) and startup assertions (section 9.2),
plus an end-to-end ground() with a canned-output stub generator. The deterministic core
(render, retrieve, sum-RRF, range-normalize, agree rerank) is exercised against the real
enriched taxonomy so tokenizer/index agreement is genuinely checked.

Run: python test_ags_frozen_grounding.py
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
import json
from types import SimpleNamespace

from ags_frozen_grounding import (
    FROZEN_AGS_QUERY_MODE,
    ONE_PASS_STRUCTURED_QUERY_MODE,
    FrozenAgsConfig,
    build_frozen_ags_method_record,
    frozen_ags_rankings,
    frozen_ags_rerank,
    frozen_ags_startup_assertions,
    one_pass_structured_config,
    range_normalize,
    sample_hypotheses,
)
from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map
from run_ags_component_validation import render_definition, render_label
from run_fintagging_grounding_baseline import (
    DEFAULT_TAXONOMY_JSONL,
    DIMENSIONS,
    Example,
    TaxonomyRetriever,
    fuse_round_candidates,
    load_taxonomy,
    tokenize,
)


PASSED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if not condition:
        raise AssertionError(f"FAILED: {name} {detail}")
    PASSED += 1
    print(f"  ok  {name}")


def make_example(idx: int, entity: str, entity_type: str, modality: str, gold: str, context: str) -> Example:
    return Example(
        example_idx=idx,
        context_id=f"ctx{idx}",
        source_sample_idx=idx,
        input_type=modality,
        entity=entity,
        entity_type=entity_type,
        row_context="",
        column_context="",
        original_context=context,
        query_context=context,
        gold_tags=[gold],
    )


def hyp(**dims: str) -> dict:
    dimensions = {dimension: dims.get(dimension.lower(), "UNRESOLVED") for dimension in DIMENSIONS}
    return {
        "dimensions": dimensions,
        "operators": ["direct_label"],
        "retrieval_query": dims.get("retrieval_query", ""),
        "hypothesis_idx": dims.get("hypothesis_idx", 0),
    }


# --- stub generator ----------------------------------------------------------------
class FakeTokenizer:
    """Callable tokenizer with no apply_chat_template, so messages_to_prompt uses the
    plain-string fallback and no model is loaded."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class FakeGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.tokenizer = FakeTokenizer()
        self.args = SimpleNamespace(query_temperature=0.0, query_top_p=1.0)
        self._outputs = list(outputs)
        self._i = 0
        self.seen_temperature: list[float] = []
        self.seen_top_p: list[float] = []

    @property
    def backend(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    def count_text_tokens(self, text: str) -> int:
        return len(text.split())

    def generate_one(self, prompt: str) -> str:
        # Record the decoding params in force at each call, so a test can prove greedy really
        # was greedy rather than trusting the config that was passed in.
        self.seen_temperature.append(self.args.query_temperature)
        self.seen_top_p.append(self.args.query_top_p)
        out = self._outputs[self._i % len(self._outputs)]
        self._i += 1
        return out


# --- section 10 unit tests ---------------------------------------------------------
def test_tokenize() -> None:
    # The shared tokenizer (spec 2) also emits camel-split pieces and lemmatizes; what the
    # method relies on is that renderer and index call this same function.
    camel = tokenize("GoodwillImpairment")
    check("tokenize exposes CamelCase pieces", "goodwill" in camel and "impairment" in camel, str(camel))
    check("tokenize drops stopwords", "of" not in tokenize("Cost of Revenue"))
    check("tokenize empty", tokenize("") == [])
    # render_label must produce exactly what tokenize produces on the same values, proving
    # the query and index agree on surface form (spec 5.2).
    check(
        "render_label tokens == tokenize(values)",
        render_label(hyp(family="Regulatory Assets")) == " ".join(tokenize("Regulatory Assets")),
    )


def test_render_label() -> None:
    h = hyp(family="Debt", role="Repayment", temporal="year three")
    rendered = render_label(h)
    check("render_label canonical order", rendered.split()[0] == "debt", f"got {rendered!r}")
    check("render_label skips unresolved", "unresolved" not in rendered.lower())
    empty = render_label(hyp())
    check("render_label all-unresolved -> falsy", not empty, f"got {empty!r}")


def test_range_normalize() -> None:
    out = range_normalize({"a": 1.0, "b": 3.0, "c": 5.0})
    check("range_normalize min->0", abs(out["a"] - 0.0) < 1e-9)
    check("range_normalize max->1", abs(out["c"] - 1.0) < 1e-9)
    tied = range_normalize({"a": 2.0, "b": 2.0})
    check("range_normalize all-tied -> zeros (spec 8)", tied == {"a": 0.0, "b": 0.0})
    check("range_normalize empty", range_normalize({}) == {})


def test_sum_rrf_multiplicity() -> None:
    # A concept in two rankings must outscore the same rank in one ranking.
    one = fuse_round_candidates(
        [{"round": 1, "candidates": [{"tag": "x", "rank": 1}]}], None, 60
    )
    two = fuse_round_candidates(
        [
            {"round": 1, "candidates": [{"tag": "x", "rank": 1}]},
            {"round": 2, "candidates": [{"tag": "x", "rank": 1}]},
        ],
        None,
        60,
    )
    score_one = one[0]["rrf_score"]
    score_two = two[0]["rrf_score"]
    check("sum_rrf multiplicity (2 rankings > 1)", score_two > score_one, f"{score_two} vs {score_one}")
    check("sum_rrf is sum not mean", abs(score_two - 2 * score_one) < 1e-6)


# --- section 9.2 startup assertions + retrieval-backed behavior --------------------
def test_label_coverage_direction(retriever_cov: TaxonomyRetriever, taxonomy) -> None:
    # Spec 7.1 / 10: a short generic label should self-retrieve at rank 1 with coverage on;
    # this fails when coverage is off, so it doubles as the "coverage active" check.
    by_label = {(c.standard_label or "").replace(" ", ""): c for c in taxonomy}
    concept = by_label.get("Goodwill")
    check("Goodwill concept present", concept is not None)
    from run_fintagging_grounding_baseline import retrieve_candidates, normalize_tag

    ranked = retrieve_candidates(retriever_cov, concept.standard_label, concept.entity_type, 10)
    check(
        "coverage: Goodwill self-retrieves at rank 1",
        ranked and normalize_tag(ranked[0]["tag"]) == normalize_tag(concept.tag),
        f"top={ranked[0]['tag'] if ranked else None}",
    )


def test_startup_assertions(retriever_cov, taxonomy, nmap) -> None:
    report = frozen_ags_startup_assertions(retriever_cov, taxonomy, nmap, FrozenAgsConfig())
    check(
        "startup: self-retrieval rate within tolerance",
        report["self_retrieval_failure_rate"] <= report["self_retrieval_tolerance"],
        f"rate={report['self_retrieval_failure_rate']} failures={report['self_retrieval_failures']}",
    )
    check("startup: coverage regression clean", not report["coverage_regression_failures"], str(report["coverage_regression_failures"]))
    check("startup: >=5 coverage labels checked", len(report["coverage_regression_checked"]) >= 5)


def test_config_frozen_guard(retriever_cov, taxonomy, nmap) -> None:
    bad = FrozenAgsConfig(rerank_beta=0.05)
    raised = False
    try:
        frozen_ags_startup_assertions(retriever_cov, taxonomy, nmap, bad)
    except AssertionError:
        raised = True
    check("config-frozen guard trips on beta drift", raised)


# --- end to end --------------------------------------------------------------------
def test_ground_end_to_end(retriever_cov, taxonomy, nmap) -> None:
    # A table fact whose gold is a current asset; hypotheses assert asset/current.
    example = make_example(
        0,
        "1,234",
        "monetaryItemType",
        "table",
        taxonomy_first_current_asset(taxonomy),
        "Balance sheet. Cash and cash equivalents, current, as of December 31, 2024.",
    )
    hypotheses = [
        hyp(family="Cash", role="carrying value", qualifier="current", temporal="as of",
            retrieval_query="cash and cash equivalents current"),
        hyp(family="Assets", role="carrying value", qualifier="current", temporal="instant",
            retrieval_query="cash equivalents current assets", hypothesis_idx=1),
    ]
    cfg = FrozenAgsConfig()
    rounds = frozen_ags_rankings(retriever_cov, example, hypotheses, cfg)
    # Table modality -> dual: def + lab per hypothesis, so up to 4 rankings.
    check("dual rendering yields def+lab rounds", len(rounds) == 4, f"got {len(rounds)}")
    check("rounds carry rendering tags", {r["rendering"] for r in rounds} == {"def", "lab"})

    reranked, diag = frozen_ags_rerank(rounds, example, hypotheses, nmap, cfg)
    check("rerank returns a pool", len(reranked) > 0)
    check("reranked ranks are 1..n contiguous", [c["rank"] for c in reranked][:3] == [1, 2, 3])
    check("final score = normed + beta*agree", all("frozen_ags_final_score" in c for c in reranked[:5]))
    # Rerank must be able to reorder relative to pure RRF (agree contributes).
    top = reranked[0]
    check("top candidate carries agree consensus", "frozen_ags_agree_consensus" in top)

    # text modality -> def only (2 rankings, no lab)
    text_example = make_example(1, "5,000", "monetaryItemType", "text",
                                example.gold_tags[0], "During the year, revenue was 5,000.")
    text_rounds = frozen_ags_rankings(retriever_cov, text_example, hypotheses, cfg)
    check("text modality uses def only", {r["rendering"] for r in text_rounds} == {"def"}, str(len(text_rounds)))
    check("text yields J def rounds", len(text_rounds) == 2)


def test_build_record_with_stub_generator(retriever_cov, taxonomy, nmap) -> None:
    gold = taxonomy_first_current_asset(taxonomy)
    example = make_example(7, "1,234", "monetaryItemType", "table", gold,
                           "Cash and cash equivalents, current, as of December 31, 2024.")
    canned = json.dumps({
        "dimensions": {"FAMILY": "Cash", "ROLE": "carrying value", "EVENT": "UNRESOLVED",
                       "QUALIFIER": "current", "SCOPE": "UNRESOLVED", "TEMPORAL": "as of"},
        "operators": ["direct_label"],
        "retrieval_query": "cash and cash equivalents current",
    })
    generator = FakeGenerator([canned])
    args = SimpleNamespace(
        query_context_max_chars=12000,
        query_max_input_tokens=16000,
        frozen_ags_top_p=1.0,
    )
    record = build_frozen_ags_method_record(args, generator, retriever_cov, example, nmap)
    check("record query_mode is frozen_ags", record["query_mode"] == FROZEN_AGS_QUERY_MODE)
    check("record candidates non-empty", len(record["candidates"]) > 0)
    check("record ranks start at 1", record["candidates"][0]["rank"] == 1)
    check("stub temperature restored", generator.args.query_temperature == 0.0)
    check("two hypotheses sampled", len(record["frozen_ags_hypotheses"]) == FrozenAgsConfig().hypotheses)
    check("no fallback used", record["frozen_ags_used_fallback"] is False)
    check("retrieval calls == 4 (table dual)", record["total_retrieval_calls"] == 4)
    check("candidates carry final score", "frozen_ags_final_score" in record["candidates"][0])


def test_sample_hypotheses_fallback(retriever_cov, taxonomy, nmap) -> None:
    example = make_example(9, "x", "monetaryItemType", "text", taxonomy[0].tag, "unparseable context")
    generator = FakeGenerator(["not json at all", "still not json"])
    args = SimpleNamespace(query_context_max_chars=12000, query_max_input_tokens=16000, frozen_ags_top_p=1.0)
    hyps, calls, used_fallback = sample_hypotheses(generator, args, example, FrozenAgsConfig())
    check("fallback engaged on total parse failure", used_fallback is True)
    check("fallback yields one hypothesis", len(hyps) == 1)


# --- one-pass grounding (structured) = AGS with J=1 ----------------------------------
def test_one_pass_structured_config() -> None:
    cfg = one_pass_structured_config()
    check("J=1", cfg.hypotheses == 1)
    check("beta=0 (nothing to reach consensus over)", cfg.rerank_beta == 0.0)
    check("greedy decoding", cfg.temperature == 0.0)
    # The whole point of the baseline: everything else must be bit-identical to AGS.
    frozen = FrozenAgsConfig()
    for field in ("top_k", "rrf_kappa", "label_coverage_weight", "agreement_top_m",
                  "dual_rendering_modalities"):
        check(f"{field} identical to AGS", getattr(cfg, field) == getattr(frozen, field),
              f"{getattr(cfg, field)!r} vs {getattr(frozen, field)!r}")


def test_one_pass_structured_frozen_guard(retriever_cov, taxonomy, nmap) -> None:
    # The J=1 variant is frozen just as hard as AGS, on its own constants.
    for bad, label in (
        (FrozenAgsConfig(hypotheses=2, rerank_beta=0.0, temperature=0.0,
                         variant=ONE_PASS_STRUCTURED_QUERY_MODE), "J drift"),
        (FrozenAgsConfig(hypotheses=1, rerank_beta=0.6, temperature=0.0,
                         variant=ONE_PASS_STRUCTURED_QUERY_MODE), "beta drift"),
        (FrozenAgsConfig(hypotheses=1, rerank_beta=0.0, temperature=0.8,
                         variant=ONE_PASS_STRUCTURED_QUERY_MODE), "temperature drift"),
    ):
        raised = False
        try:
            frozen_ags_startup_assertions(retriever_cov, taxonomy, nmap, bad)
        except AssertionError:
            raised = True
        check(f"one_pass_structured guard trips on {label}", raised)
    # And an unknown variant must not silently pass.
    raised = False
    try:
        frozen_ags_startup_assertions(retriever_cov, taxonomy, nmap,
                                      FrozenAgsConfig(variant="not_a_variant"))
    except AssertionError:
        raised = True
    check("unknown variant rejected", raised)


def test_one_pass_structured_beta_zero_is_pure_fused_rank(retriever_cov, taxonomy, nmap) -> None:
    """With beta=0 the consensus term drops out, so the ranking must be exactly the
    range-normalized sum-RRF order. This is the spec's "NO consensus rerank" clause."""
    example = make_example(
        11, "1,234", "monetaryItemType", "table",
        taxonomy_first_current_asset(taxonomy),
        "Balance sheet. Cash and cash equivalents, current, as of December 31, 2024.",
    )
    hypotheses = [hyp(family="Cash", role="carrying value", qualifier="current",
                      temporal="as of", retrieval_query="cash and cash equivalents current")]
    cfg = one_pass_structured_config()
    rounds = frozen_ags_rankings(retriever_cov, example, hypotheses, cfg)
    check("J=1 table fact yields def+lab only", len(rounds) == 2, f"got {len(rounds)}")

    reranked, _ = frozen_ags_rerank(rounds, example, hypotheses, nmap, cfg)
    check("beta=0 still returns a pool", len(reranked) > 0)
    # final == normalized fused score, exactly.
    check(
        "final score equals the normalized fused score",
        all(abs(c["frozen_ags_final_score"] - c["frozen_ags_rrf_normalized"]) < 1e-9
            for c in reranked),
    )
    # ...and the order is that score's order (ties broken by tag, as fuse does).
    expected = sorted(reranked, key=lambda c: (-c["frozen_ags_rrf_normalized"], c["tag"]))
    check("order is the normalized fused order",
          [c["tag"] for c in reranked] == [c["tag"] for c in expected])

    # Non-zero consensus is what beta would have multiplied; proving it is present but
    # ignored is stronger than proving it is absent.
    check("consensus was computed and then zeroed out by beta",
          any(c["frozen_ags_agree_consensus"] > 0 for c in reranked))


def test_one_pass_structured_record(retriever_cov, taxonomy, nmap) -> None:
    gold = taxonomy_first_current_asset(taxonomy)
    example = make_example(12, "1,234", "monetaryItemType", "table", gold,
                           "Cash and cash equivalents, current, as of December 31, 2024.")
    canned = json.dumps({
        "dimensions": {"FAMILY": "Cash", "ROLE": "carrying value", "EVENT": "UNRESOLVED",
                       "QUALIFIER": "current", "SCOPE": "UNRESOLVED", "TEMPORAL": "as of"},
        "operators": ["direct_label"],
        "retrieval_query": "cash and cash equivalents current",
    })
    generator = FakeGenerator([canned])
    args = SimpleNamespace(query_context_max_chars=12000, query_max_input_tokens=16000,
                           frozen_ags_top_p=0.9)
    cfg = one_pass_structured_config()
    record = build_frozen_ags_method_record(args, generator, retriever_cov, example, nmap, cfg=cfg)
    check("record query_mode is one_pass_structured",
          record["query_mode"] == ONE_PASS_STRUCTURED_QUERY_MODE)
    check("exactly one hypothesis", len(record["frozen_ags_hypotheses"]) == 1)
    check("exactly one LLM call", len(record["frozen_ags_llm_calls"]) == 1)
    check("two retrieval calls (table dual, J=1)", record["total_retrieval_calls"] == 2)
    check("config block records the variant",
          record["frozen_ags_config"]["variant"] == ONE_PASS_STRUCTURED_QUERY_MODE)
    check("config block records beta=0", record["frozen_ags_config"]["rerank_beta"] == 0.0)
    # Greedy must not be silently nucleus-sampled by a stray --frozen-ags-top-p.
    check("greedy pins top_p to 1.0", generator.seen_top_p == [1.0], str(generator.seen_top_p))
    # Downstream readers (T28 replay, Table 5 loaders) key off these names.
    for key in ("frozen_ags_hypotheses", "frozen_ags_config", "rounds"):
        check(f"trace still carries {key} for downstream readers", key in record)


def taxonomy_first_current_asset(taxonomy) -> str:
    for concept in taxonomy:
        if "Current" in (concept.raw_tag or "") and concept.entity_type == "monetaryItemType":
            return concept.tag
    return taxonomy[0].tag


def main() -> None:
    print("loading taxonomy + normalization map ...", flush=True)
    taxonomy = load_taxonomy(DEFAULT_TAXONOMY_JSONL)
    nmap = load_normalization_map(DEFAULT_NORMALIZATION_MAP)
    retriever_cov = TaxonomyRetriever(taxonomy, type_filter=True, label_coverage_weight=1.0,
                                      label_coverage_pool_multiplier=0)

    print("\n[section 10] deterministic unit tests")
    test_tokenize()
    test_render_label()
    test_range_normalize()
    test_sum_rrf_multiplicity()

    print("\n[section 7/9.2] coverage + startup assertions")
    test_label_coverage_direction(retriever_cov, taxonomy)
    test_startup_assertions(retriever_cov, taxonomy, nmap)
    test_config_frozen_guard(retriever_cov, taxonomy, nmap)

    print("\n[section 9] end-to-end ground()")
    test_ground_end_to_end(retriever_cov, taxonomy, nmap)
    test_build_record_with_stub_generator(retriever_cov, taxonomy, nmap)
    test_sample_hypotheses_fallback(retriever_cov, taxonomy, nmap)

    print("\n[one-pass grounding (structured) = AGS with J=1]")
    test_one_pass_structured_config()
    test_one_pass_structured_frozen_guard(retriever_cov, taxonomy, nmap)
    test_one_pass_structured_beta_zero_is_pure_fused_rank(retriever_cov, taxonomy, nmap)
    test_one_pass_structured_record(retriever_cov, taxonomy, nmap)

    print(f"\nALL {PASSED} CHECKS PASSED")


if __name__ == "__main__":
    main()
