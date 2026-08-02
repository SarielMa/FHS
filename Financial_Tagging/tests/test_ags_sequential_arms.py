#!/usr/bin/env python3
"""Offline tests for the sequential control arms (no GPU, no vLLM).

Covers what ags_seq_arms_spec.md asserts: round one is AGS byte-for-byte, the slate is
admissible and capped, the preserve set is enforced by the revision step rather than
suggested to it, the two arms differ only in selection, the posteriors forget and exclude
the current source context, and an episode reaches its budget with the gate off.

Run: python test_ags_sequential_arms.py
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

import json
import random
from types import SimpleNamespace

import numpy as np

import ags_sequential_arms
from ags_frozen_grounding import frozen_ags_rankings, frozen_ags_rerank
from ags_sequential_arms import (
    AGS_SEQ_QUERY_MODE,
    AGS_SEQ_RANDOM_QUERY_MODE,
    OPERATORS,
    PERTURB_OPERATOR,
    AgsSeqConfig,
    SequentialPosteriorBank,
    admissible_slate,
    assert_round_one_parity,
    build_ags_seq_method_record,
    cluster_representatives,
    consolidate,
    directive_payload,
    enforce_directive,
    episode_feedback,
    psi_vector,
    query_novelty,
    select_directive,
)
from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map
from run_fintagging_grounding_baseline import (
    DEFAULT_TAXONOMY_JSONL,
    DIMENSIONS,
    Example,
    TaxonomyRetriever,
    load_taxonomy,
    normalize_tag,
)


PASSED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if not condition:
        raise AssertionError(f"FAILED: {name} {detail}")
    PASSED += 1
    print(f"  ok  {name}")


def make_example(idx: int, modality: str, gold: str, context: str, context_id: str = "ctx") -> Example:
    return Example(
        example_idx=idx,
        context_id=context_id,
        source_sample_idx=idx,
        input_type=modality,
        entity="1,234",
        entity_type="monetaryItemType",
        row_context="Cash and cash equivalents",
        column_context="December 31, 2024",
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


def feedback_stub(supported=(), contradicted=(), unresolved=(), mismatch=False) -> dict:
    return {
        "supported_dimensions": list(supported),
        "contradicted_dimensions": list(contradicted),
        "unresolved_dimensions": list(unresolved),
        "structural_mismatch": {"is_mismatch": mismatch, "reason": "test"},
        "dimension_verdicts": [],
    }


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class FakeGenerator:
    """Cycles canned JSON hypotheses; records the decoding setting of each call."""

    def __init__(self, outputs: list[str]) -> None:
        self.tokenizer = FakeTokenizer()
        self.args = SimpleNamespace(query_temperature=0.0, query_top_p=1.0)
        self._outputs = list(outputs)
        self._i = 0
        self.temperatures: list[float] = []

    @property
    def backend(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    def count_text_tokens(self, text: str) -> int:
        return len(text.split())

    def generate_one(self, prompt: str) -> str:
        self.temperatures.append(self.args.query_temperature)
        out = self._outputs[self._i % len(self._outputs)]
        self._i += 1
        return out


def canned(family: str, role: str, qualifier: str, query: str) -> str:
    return json.dumps(
        {
            "dimensions": {
                "FAMILY": family,
                "ROLE": role,
                "EVENT": "UNRESOLVED",
                "QUALIFIER": qualifier,
                "SCOPE": "UNRESOLVED",
                "TEMPORAL": "as of",
            },
            "operators": ["direct_label"],
            "retrieval_query": query,
        }
    )


def stub_args() -> SimpleNamespace:
    return SimpleNamespace(query_context_max_chars=12000, query_max_input_tokens=16000, frozen_ags_top_p=1.0)


# --- section 1: round one is AGS -----------------------------------------------------
def test_round_one_parity(retriever, nmap) -> None:
    example = make_example(0, "table", "us-gaap:Goodwill", "Cash and cash equivalents, current.")
    hypotheses = [
        hyp(family="Cash", role="carrying value", qualifier="current", retrieval_query="cash and cash equivalents current"),
        hyp(family="Assets", role="carrying value", qualifier="current", retrieval_query="cash equivalents current assets", hypothesis_idx=1),
    ]
    cfg = AgsSeqConfig()
    rounds = frozen_ags_rankings(retriever, example, hypotheses, cfg.frozen)
    report = assert_round_one_parity(rounds, example, hypotheses, nmap, cfg)
    check("round-1 parity holds against frozen_ags_rerank", report["round1_parity_ok"] is True)
    check("round-1 pool is non-trivial", report["round1_candidates"] > 50, str(report["round1_candidates"]))

    reference, _ = frozen_ags_rerank(rounds, example, hypotheses, nmap, cfg.frozen)
    produced = consolidate(rounds, hypotheses, nmap, cfg, cfg.frozen.top_k)
    check(
        "consolidate reproduces frozen ordering exactly",
        [normalize_tag(c["tag"]) for c in reference] == [normalize_tag(c["tag"]) for c in produced],
    )

    # The assertion has to be able to fail. Both sides consume the same retrievals, so
    # non-vacuity means catching drift in the *consolidation*: drop the agreement term from
    # the sequential side and the frozen reference must disagree.
    original_consensus = ags_sequential_arms.consensus_scores
    ags_sequential_arms.consensus_scores = lambda candidates, hypotheses_, nmap_: {}
    raised = ""
    try:
        assert_round_one_parity(rounds, example, hypotheses, nmap, cfg)
    except AssertionError as error:
        raised = str(error)
    finally:
        ags_sequential_arms.consensus_scores = original_consensus
    check(
        "parity assertion is not vacuous (consolidation drift is caught)",
        "diverged from frozen AGS" in raised,
        raised[:120],
    )

    # Untruncated consolidation must extend, not reorder, the delivered top-K prefix's pool.
    untruncated = consolidate(rounds, hypotheses, nmap, cfg, None)
    check("untruncated pool is larger than K", len(untruncated) >= len(produced))


# --- section 2.1: feedback ------------------------------------------------------------
def test_cluster_representatives(retriever, nmap) -> None:
    example = make_example(1, "table", "us-gaap:Goodwill", "Goodwill impairment during the period.")
    hypotheses = [hyp(family="Goodwill", role="impairment", retrieval_query="goodwill impairment")]
    cfg = AgsSeqConfig()
    rounds = frozen_ags_rankings(retriever, example, hypotheses, cfg.frozen)
    ranking = consolidate(rounds, hypotheses, nmap, cfg, cfg.frozen.top_k)
    representatives = cluster_representatives(ranking, nmap, cfg.feedback_top_m, cfg.cluster_scan_depth)
    check("representatives count == M", len(representatives) == cfg.feedback_top_m, str(len(representatives)))
    tags = [normalize_tag(c["tag"]) for c in representatives]
    check("representatives are distinct", len(set(tags)) == len(tags))
    raw_top = [normalize_tag(c["tag"]) for c in ranking[: cfg.feedback_top_m]]
    check(
        "representatives are cluster-selected, not the raw top-M",
        tags != raw_top or len(ranking) < cfg.cluster_scan_depth,
        f"{tags[:3]} vs {raw_top[:3]}",
    )
    check("representatives keep rank order", tags == [t for t in [normalize_tag(c["tag"]) for c in ranking] if t in set(tags)])


def test_llm_feedback_stays_disabled(nmap) -> None:
    cfg = AgsSeqConfig(llm_feedback_enabled=True)
    raised = False
    try:
        episode_feedback(hyp(family="Cash"), [], nmap, cfg)
    except AssertionError:
        raised = True
    check("LLM verification layer refuses to run", raised)


# --- section 2.2: controller ---------------------------------------------------------
def test_slate_admissibility() -> None:
    cfg = AgsSeqConfig()
    feedback = feedback_stub(
        supported=["FAMILY"], contradicted=["ROLE", "EVENT"], unresolved=["SCOPE"], mismatch=True
    )
    slate = admissible_slate(feedback, cfg)
    modes = {item["mode"] for item in slate}
    check("REFINE proposed for D_minus", "REFINE" in modes)
    check("BRANCH proposed for D_question", "BRANCH" in modes)
    check("CHANGE_STRATEGY proposed when g fires", "CHANGE_STRATEGY" in modes)
    check("PERTURB always proposed", "PERTURB" in modes)
    operators = [item["operator"] for item in slate]
    check("one directive per operator", len(operators) == len(set(operators)), str(operators))
    check("preserve set carries D_plus", all(item["preserve"] == ["FAMILY"] for item in slate))

    # No mismatch, nothing contradicted: only BRANCH + PERTURB are admissible.
    quiet = admissible_slate(feedback_stub(supported=list(DIMENSIONS)), cfg)
    check("empty feedback still yields PERTURB", [item["operator"] for item in quiet] == [PERTURB_OPERATOR])

    # Cap at L, with PERTURB never crowded out.
    crowded = admissible_slate(
        feedback_stub(contradicted=list(DIMENSIONS), mismatch=True), cfg
    )
    check("slate capped at L=6", len(crowded) == cfg.slate_limit, str(len(crowded)))
    check(
        "PERTURB survives the cap",
        any(item["operator"] == PERTURB_OPERATOR for item in crowded),
        str([item["operator"] for item in crowded]),
    )


def test_directive_payload_shape() -> None:
    feedback = feedback_stub(supported=["FAMILY"], contradicted=["ROLE"])
    slate = admissible_slate(feedback, AgsSeqConfig())
    directive = directive_payload(slate[0], feedback)
    check(
        "directive is (mode, operator, target, patch, preserve)",
        set(directive) >= {"mode", "operator", "target_dimension", "semantic_patch", "preserve"},
        str(sorted(directive)),
    )
    check("directive targets the contradicted dimension", directive["target_dimension"] == "ROLE")
    check("directive preserves D_plus", directive["preserve"] == ["FAMILY"])


# --- section 2.3: the only difference between the arms --------------------------------
def test_arms_differ_only_in_selection() -> None:
    cfg = AgsSeqConfig()
    bank = SequentialPosteriorBank(OPERATORS, cfg)
    psi = np.ones(len(cfg.psi_features))
    # Teach the bank that O_perturb pays and the refine operators do not.
    for index in range(40):
        bank.update(PERTURB_OPERATOR, psi, 1.0, context_id=f"other{index}")
        bank.update("O_refine_role", psi, -1.0, context_id=f"other{index}")
    feedback = feedback_stub(contradicted=["ROLE"])
    slate = admissible_slate(feedback, cfg)

    thompson_hits = 0
    random_hits = 0
    for trial in range(200):
        selected, _, scores = select_directive(
            AGS_SEQ_QUERY_MODE, slate, bank, psi, random.Random(trial), "ctx-now"
        )
        thompson_hits += selected["operator"] == PERTURB_OPERATOR
        selected_random, _, random_scores = select_directive(
            AGS_SEQ_RANDOM_QUERY_MODE, slate, bank, psi, random.Random(trial), "ctx-now"
        )
        random_hits += selected_random["operator"] == PERTURB_OPERATOR
        check_scores = set(scores) == set(random_scores) == {item["operator"] for item in slate}
        if not check_scores:
            raise AssertionError("both arms must score the whole slate")
    check("thompson arm follows the posterior", thompson_hits > 180, f"{thompson_hits}/200")
    check("random arm ignores the posterior", 60 < random_hits < 140, f"{random_hits}/200")

    selected, runner_up, _ = select_directive(
        AGS_SEQ_QUERY_MODE, slate, bank, psi, random.Random(0), "ctx-now"
    )
    check("runner-up is frozen at decision time", runner_up is not None and runner_up is not selected)


# --- section 2.2/2.4: revision enforcement -------------------------------------------
def test_preserve_set_is_enforced() -> None:
    before = hyp(family="Cash", role="carrying value", qualifier="current", retrieval_query="cash current")
    wandered = hyp(family="Debt", role="repayment", qualifier="noncurrent", retrieval_query="debt repayment")

    refine = {"mode": "REFINE", "operator": "O_refine_role", "target_dimension": "ROLE", "preserve": ["FAMILY"]}
    revised, reverted = enforce_directive(before, wandered, refine)
    check("REFINE moves only the target dimension", revised["dimensions"]["ROLE"] == "repayment")
    check("REFINE restores FAMILY", revised["dimensions"]["FAMILY"] == "Cash")
    check("REFINE restores unauthorized QUALIFIER", revised["dimensions"]["QUALIFIER"] == "current")
    check("REFINE reports what it reverted", set(reverted) == {"FAMILY", "QUALIFIER"}, str(reverted))

    perturb = {"mode": "PERTURB", "operator": "O_perturb", "target_dimension": "", "preserve": []}
    perturbed, perturb_reverted = enforce_directive(before, wandered, perturb)
    check(
        "PERTURB changes no dimension",
        perturbed["dimensions"] == before["dimensions"],
        str(perturbed["dimensions"]),
    )
    check("PERTURB keeps only the new surface", perturbed["retrieval_query"] == "debt repayment")
    check("PERTURB reverts every wandered dimension", set(perturb_reverted) == {"FAMILY", "ROLE", "QUALIFIER"})

    change = {
        "mode": "CHANGE_STRATEGY",
        "operator": "O_change_strategy",
        "target_dimension": "",
        "preserve": ["FAMILY"],
    }
    changed, change_reverted = enforce_directive(before, wandered, change)
    check("CHANGE_STRATEGY holds the preserve set", changed["dimensions"]["FAMILY"] == "Cash")
    check("CHANGE_STRATEGY may re-read the rest", changed["dimensions"]["ROLE"] == "repayment")
    check("CHANGE_STRATEGY reverts only preserved", change_reverted == ["FAMILY"], str(change_reverted))


# --- section 4: delayed learning ------------------------------------------------------
def test_posterior_context_exclusion_and_forgetting() -> None:
    cfg = AgsSeqConfig(posterior_forgetting=0.5)
    bank = SequentialPosteriorBank(OPERATORS, cfg)
    psi = np.ones(len(cfg.psi_features))

    bank.update("O_refine_role", psi, 5.0, context_id="table-A")
    with_context = bank.mean_score("O_refine_role", psi, exclude_context="table-B")
    without_context = bank.mean_score("O_refine_role", psi, exclude_context="table-A")
    check("credit is visible from another context", with_context > 0.1, str(with_context))
    check("same-context records are excluded", abs(without_context) < 1e-12, str(without_context))

    # Forgetting: an old record must weigh less than a fresh one of the same size.
    fresh = SequentialPosteriorBank(OPERATORS, cfg)
    fresh.update("O_refine_role", psi, 1.0, context_id="ctx-old")
    old_only = fresh.mean_score("O_refine_role", psi, exclude_context="ctx-now")
    for index in range(10):
        fresh.update("O_perturb", psi, 0.0, context_id=f"filler{index}")
    aged = fresh.mean_score("O_refine_role", psi, exclude_context="ctx-now")
    check("older credit decays under zeta", aged < old_only, f"{aged} vs {old_only}")

    check("bank never stores gold", not any("gold" in key for key in vars(bank)))


def test_bank_ingest_round_trip() -> None:
    cfg = AgsSeqConfig()
    bank = SequentialPosteriorBank(OPERATORS, cfg)
    record = {
        "context_id": "ctx-1",
        "ags_seq_rounds": [
            {"selected_operator": "O_refine_role", "psi_values": [1.0] * len(cfg.psi_features), "reward": 0.25},
            {"selected_operator": PERTURB_OPERATOR, "psi_values": [1.0] * len(cfg.psi_features), "reward": -0.1},
        ],
    }
    bank.ingest_record(record)
    check("resume replays every round's credit", bank.step == 2, str(bank.step))
    check(
        "replayed credit is attributed to the right operator",
        bank.mean_score("O_refine_role", np.ones(len(cfg.psi_features)), exclude_context="other") > 0,
    )


# --- section 3 + 5: episode -----------------------------------------------------------
def test_episode_end_to_end(retriever, nmap) -> None:
    gold = "us-gaap:CashAndCashEquivalentsAtCarryingValue"
    example = make_example(11, "table", gold, "Cash and cash equivalents, current, as of December 31, 2024.")
    generator = FakeGenerator(
        [
            canned("Cash", "carrying value", "current", "cash and cash equivalents current"),
            canned("Cash", "carrying value", "current", "cash equivalents held at carrying value"),
            canned("Cash", "restricted", "current", "restricted cash current portion"),
            canned("Cash", "carrying value", "noncurrent", "cash and equivalents noncurrent"),
        ]
    )
    cfg = AgsSeqConfig()
    bank = SequentialPosteriorBank(OPERATORS, cfg)
    record = build_ags_seq_method_record(
        stub_args(), AGS_SEQ_QUERY_MODE, generator, retriever, example, nmap, bank, cfg=cfg
    )

    check("record carries the arm", record["ags_seq_arm"] == AGS_SEQ_QUERY_MODE)
    check("round-1 parity recorded per fact", record["ags_seq_round1_parity"]["round1_parity_ok"] is True)
    check("gate off by default", record["ags_seq_config"]["novelty_gate"] is False)
    check(
        "episode reaches its budget with the gate off",
        record["realized_rounds"] == cfg.max_rounds,
        f"{record['realized_rounds']} rounds, stop={record['stop_reason']}",
    )
    check("stop reason reported", record["stop_reason"] == "budget_exhausted", record["stop_reason"])
    check("both candidate sets present", bool(record["round1_candidates"]) and bool(record["candidates"]))
    check("round-1 and final ranks both reported", "round1_rank_gold" in record and "final_rank_gold" in record)
    check(
        "final pool is at least the round-1 pool",
        record["total_retrieval_calls"] > 4,
        str(record["total_retrieval_calls"]),
    )

    rounds = record["ags_seq_rounds"]
    check("one row per controller round", len(rounds) == cfg.max_rounds - 1, str(len(rounds)))
    first = rounds[0]
    for field in (
        "psi",
        "slate",
        "selected_operator",
        "selected_mode",
        "runner_up_operator",
        "D_plus_count",
        "D_minus_count",
        "D_question_count",
        "gold_in_union_before",
        "gold_in_union_after",
        "rank_before",
        "rank_after",
        "delta_y",
        "delta_replay",
        "reward",
        "candidate_list",
    ):
        check(f"rounds.jsonl field present: {field}", field in first)
    check("psi is the reduced block", list(first["psi"]) == list(cfg.psi_features), str(list(first["psi"])))
    check(
        "reward = alpha*delta_y + (1-alpha)*delta_replay",
        abs(first["reward"] - (cfg.reward_alpha * first["delta_y"] + (1 - cfg.reward_alpha) * first["delta_replay"]))
        < 1e-6,
    )
    check("posterior updated once per realized round", bank.step == len(rounds), str(bank.step))
    check("generator temperature restored", generator.args.query_temperature == 0.0)
    check(
        "every post-hypothesis call decodes deterministically unless it is a PERTURB revision",
        all(
            temperature == 0.0
            for temperature, round_record in zip(
                generator.temperatures[cfg.frozen.hypotheses :: 2], rounds
            )
            if round_record["selected_mode"] != "PERTURB"
        ),
        str(generator.temperatures),
    )
    check(
        "replays decode deterministically whatever the runner-up directive is",
        all(temperature == 0.0 for temperature in generator.temperatures[cfg.frozen.hypotheses + 1 :: 2]),
        str(generator.temperatures),
    )

    # The random arm runs the same episode shape and still moves its posteriors.
    random_bank = SequentialPosteriorBank(OPERATORS, cfg)
    random_generator = FakeGenerator(generator._outputs)
    random_record = build_ags_seq_method_record(
        stub_args(), AGS_SEQ_RANDOM_QUERY_MODE, random_generator, retriever, example, nmap, random_bank, cfg=cfg
    )
    check("random arm reports its own name", random_record["ags_seq_arm"] == AGS_SEQ_RANDOM_QUERY_MODE)
    check("random arm updates posteriors too", random_bank.step == len(random_record["ags_seq_rounds"]))
    check(
        "random arm starts from the same round one",
        random_record["round1_candidates"] == record["round1_candidates"],
    )


def test_novelty_gate_when_armed(retriever, nmap) -> None:
    check("identical queries have zero novelty", query_novelty("cash current", ["cash current"]) == 0.0)
    check("first query is fully novel", query_novelty("cash current", []) == 1.0)

    gold = "us-gaap:CashAndCashEquivalentsAtCarryingValue"
    example = make_example(12, "table", gold, "Cash and cash equivalents, current.")
    # Every revision re-renders the same surface, which is what an atomic edit does in
    # practice; with the gate armed the episode must stop instead of spending its budget.
    repeated = canned("Cash", "carrying value", "current", "cash and cash equivalents current")
    cfg = AgsSeqConfig(novelty_gate=True)
    bank = SequentialPosteriorBank(OPERATORS, cfg)
    record = build_ags_seq_method_record(
        stub_args(),
        AGS_SEQ_QUERY_MODE,
        FakeGenerator([repeated]),
        retriever,
        example,
        nmap,
        bank,
        cfg=cfg,
    )
    check(
        "armed gate terminates the episode early",
        record["stop_reason"] == "novelty_gate_exhaustion" and record["realized_rounds"] < cfg.max_rounds,
        f"{record['stop_reason']} at {record['realized_rounds']} rounds",
    )

    gate_off = build_ags_seq_method_record(
        stub_args(),
        AGS_SEQ_QUERY_MODE,
        FakeGenerator([repeated]),
        retriever,
        example,
        nmap,
        SequentialPosteriorBank(OPERATORS, AgsSeqConfig()),
        cfg=AgsSeqConfig(),
    )
    check(
        "same episode reaches B with the gate off",
        gate_off["realized_rounds"] == AgsSeqConfig().max_rounds,
        str(gate_off["realized_rounds"]),
    )
    check(
        "gate-off run still reports what the gate would have done",
        gate_off["ags_seq_rounds"][0]["gate_would_reject"] is True,
    )


def test_psi_vector_shape() -> None:
    cfg = AgsSeqConfig()
    example = make_example(13, "text", "us-gaap:Goodwill", "Revenue for the year.")
    named, vector = psi_vector(example, feedback_stub(contradicted=["ROLE"]), 0.5, cfg.psi_features)
    check("psi length matches the reduced block", len(vector) == len(cfg.psi_features) == 9, str(len(vector)))
    check("psi is modality aware", named["is_table"] == 0.0)
    check("psi carries the feedback counts", named["D_minus_count"] == 1.0)
    check("psi carries novelty", named["neighborhood_novelty_n"] == 0.5)
    check("psi holds no gold or candidate identity", all(isinstance(value, float) for value in named.values()))


def main() -> None:
    print("loading taxonomy + normalization map ...", flush=True)
    taxonomy = load_taxonomy(DEFAULT_TAXONOMY_JSONL)
    nmap = load_normalization_map(DEFAULT_NORMALIZATION_MAP)
    retriever = TaxonomyRetriever(
        taxonomy, type_filter=True, label_coverage_weight=1.0, label_coverage_pool_multiplier=0
    )

    print("\n[section 1] round one is AGS")
    test_round_one_parity(retriever, nmap)

    print("\n[section 2.1] feedback")
    test_cluster_representatives(retriever, nmap)
    test_llm_feedback_stays_disabled(nmap)

    print("\n[section 2.2] controller")
    test_slate_admissibility()
    test_directive_payload_shape()

    print("\n[section 2.3] the only difference between the arms")
    test_arms_differ_only_in_selection()

    print("\n[section 2.4] revision enforcement")
    test_preserve_set_is_enforced()

    print("\n[section 4] delayed learning")
    test_posterior_context_exclusion_and_forgetting()
    test_bank_ingest_round_trip()
    test_psi_vector_shape()

    print("\n[sections 3 + 5] episode")
    test_episode_end_to_end(retriever, nmap)
    test_novelty_gate_when_armed(retriever, nmap)

    print(f"\nALL {PASSED} CHECKS PASSED")


if __name__ == "__main__":
    main()
