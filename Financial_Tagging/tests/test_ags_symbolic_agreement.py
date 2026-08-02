#!/usr/bin/env python3
"""Unit tests for shared AGS symbolic agreement."""

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

import unittest

from ags_symbolic_agreement import (
    VERDICT_CONTRADICT,
    VERDICT_SUPPORT,
    agree,
    merge_feedback_layers,
    normalize_hypothesis_dimensions,
    symbolic_feedback_from_candidates,
)


class SymbolicAgreementTests(unittest.TestCase):
    def test_unresolved_dimensions_are_excluded(self) -> None:
        candidate = {
            "tag": "us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree",
            "standard_label": "Long Term Debt Maturities Repayments Of Principal In Year Three",
            "documentation": "Amount of long-term debt principal repayment due in year three.",
            "type": "monetaryItemType",
        }
        result = agree(
            candidate,
            {
                "FAMILY": "debt",
                "ROLE": "repayment principal",
                "TEMPORAL": "UNRESOLVED",
            },
        )
        self.assertGreaterEqual(result.score, 0.5)
        self.assertEqual(result.evaluated, 2)

    def test_controlled_temporal_and_qualifier_match(self) -> None:
        candidate = {
            "tag": "us-gaap:WeightedAverageDilutedSharesOutstanding",
            "standard_label": "Weighted Average Diluted Shares Outstanding",
            "documentation": "Weighted average number of diluted shares outstanding during the period.",
            "type": "sharesItemType",
        }
        result = agree(
            candidate,
            {
                "FAMILY": "equity shares",
                "QUALIFIER": "weighted average diluted",
                "TEMPORAL": "duration",
            },
        )
        self.assertEqual(result.evaluated, 3)
        self.assertEqual(result.matched, 3)

    def test_normalization_exposes_controlled_categories(self) -> None:
        normalized = normalize_hypothesis_dimensions(
            {"QUALIFIER": "net of tax", "TEMPORAL": "third year"}
        )
        self.assertIn("aftertax", normalized["qualifier"]["categories"])
        self.assertIn("year_3", normalized["temporal"]["categories"])

    def test_symbolic_feedback_aggregates_top_candidates(self) -> None:
        candidates = [
            {
                "tag": "us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree",
                "standard_label": "Long Term Debt Maturities Repayments Of Principal In Year Three",
                "documentation": "Long-term debt principal repayments due in year three.",
                "type": "monetaryItemType",
            },
            {
                "tag": "us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour",
                "standard_label": "Long Term Debt Maturities Repayments Of Principal In Year Four",
                "documentation": "Long-term debt principal repayments due in year four.",
                "type": "monetaryItemType",
            },
        ]
        feedback = symbolic_feedback_from_candidates(
            {"FAMILY": "debt", "ROLE": "principal repayment", "TEMPORAL": "year three"},
            candidates,
            top_m=2,
        )
        self.assertIn("FAMILY", feedback["supported_dimensions"])
        self.assertIn("ROLE", feedback["supported_dimensions"])
        self.assertIn("TEMPORAL", feedback["unresolved_dimensions"])

    def test_merge_feedback_keeps_symbolic_conflict_auditable(self) -> None:
        symbolic = {
            "supported_dimensions": ["FAMILY"],
            "contradicted_dimensions": [],
            "unresolved_dimensions": ["ROLE"],
            "dimension_verdicts": [
                {
                    "dimension": "FAMILY",
                    "verdict": VERDICT_SUPPORT,
                    "source_layer": "symbolic",
                    "confidence": 0.9,
                }
            ],
            "structural_mismatch": {"is_mismatch": False, "reason": ""},
        }
        llm = {
            "supported_dimensions": [],
            "contradicted_dimensions": ["FAMILY"],
            "unresolved_dimensions": ["ROLE"],
            "structural_mismatch": {"is_mismatch": True, "reason": "LLM saw mismatch"},
        }
        merged = merge_feedback_layers(symbolic, llm)
        family = [v for v in merged["dimension_verdicts"] if v["dimension"] == "FAMILY"][0]
        self.assertEqual(family["verdict"], VERDICT_SUPPORT)
        self.assertTrue(family["llm_disagrees"])
        self.assertEqual(family["llm_verdict"], VERDICT_CONTRADICT)
        self.assertTrue(merged["structural_mismatch"]["is_mismatch"])


if __name__ == "__main__":
    unittest.main()
