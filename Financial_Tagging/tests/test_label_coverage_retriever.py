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

from run_fintagging_grounding_baseline import (
    DEFAULT_TAXONOMY_JSONL,
    TaxonomyRetriever,
    load_taxonomy,
    normalize_tag,
    retrieve_candidates,
)


class LabelCoverageRetrieverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.taxonomy = load_taxonomy(DEFAULT_TAXONOMY_JSONL)
        cls.by_tag = {concept.tag: concept for concept in cls.taxonomy}

    def test_known_generic_label_queries_retrieve_self_at_rank_1(self) -> None:
        retriever = TaxonomyRetriever(
            self.taxonomy,
            type_filter=True,
            label_coverage_weight=1.0,
            label_coverage_pool_multiplier=0,
        )
        tags = (
            "us-gaap:Assets",
            "us-gaap:Liabilities",
            "us-gaap:Revenues",
            "us-gaap:Goodwill",
            "us-gaap:Depreciation",
            "us-gaap:RegulatoryAssetsCurrent",
        )
        failures = []
        for tag in tags:
            concept = self.by_tag[normalize_tag(tag)]
            candidates = retrieve_candidates(
                retriever,
                concept.standard_label,
                concept.entity_type,
                10,
            )
            top_tag = normalize_tag(candidates[0]["tag"]) if candidates else None
            if top_tag != concept.tag:
                failures.append(
                    {
                        "query": concept.standard_label,
                        "expected": concept.tag,
                        "observed": top_tag,
                        "top_5": [candidate["tag"] for candidate in candidates[:5]],
                    }
                )
        self.assertEqual([], failures)


if __name__ == "__main__":
    unittest.main()
