"""End-to-end test of the verifier fixes, with a fake generator (no GPU, no vLLM).

Covers the three failure modes the truncation bug exposed:
  1. a truncated response still yields verdicts (salvage)
  2. an all-failing run aborts early instead of burning hours
  3. --resume regenerates unusable rows instead of skipping them
"""
import json, sys, types, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import verifier.run_llm_verifier as V

TAGS = [f"us-gaap:Concept{i}" for i in range(10)]


def make_trace(path, n_facts=30):
    with open(path, "w") as fh:
        for i in range(n_facts):
            cands = [
                {
                    "rank": j + 1, "tag": TAGS[j], "type": "monetaryItemType",
                    "standard_label": f"Concept {j}", "documentation": f"Amount of concept {j} revenue.",
                    "retrieval_text": f"Concept{j}. Concept {j}. Amount of concept {j} revenue.",
                }
                for j in range(10)
            ]
            fh.write(json.dumps({
                "example_idx": i, "context_id": i // 3, "input_type": "table",
                "entity": "100", "type": "monetaryItemType",
                "row_context": "Revenues", "column_context": "2024", "query_context": "ctx",
                "gold_tags": ["us-gaap:Concept1"],
                "candidates": cands, "final_candidates": cands,
                "frozen_ags_hypotheses": [
                    {"hypothesis_idx": 0, "dimensions": {"FAMILY": "REVENUE", "ROLE": "TOTAL", "EVENT": "STATE"}},
                    {"hypothesis_idx": 1, "dimensions": {"FAMILY": "REVENUE", "ROLE": "NET", "EVENT": "STATE"}},
                ],
            }) + "\n")


GOOD = json.dumps({"verdicts": [
    {"tag": t, "FAMILY": "support", "ROLE": "contradict", "EVENT": "unresolved", "confidence": 0.8}
    for t in TAGS
]})
# byte-for-byte the shape the real run produced: valid JSON cut off mid-array
TRUNCATED = GOOD[: int(len(GOOD) * 0.62)]


def fake_generator(output):
    class FakeGen:
        backend = "fake"
        model_name = "fake-model"
        def __init__(self, args): self.tokenizer = None
        def generate_one(self, prompt): return output
        # The runner batches through generate_many (it called generate_one per prompt until the
        # batching fix, which is why this fake silently stopped matching it and the test failed
        # with AttributeError rather than a wrong assertion).
        def generate_many(self, prompts): return [output for _ in prompts]
        def count_text_tokens(self, text): return len(text) // 4
        def close(self): pass
    return FakeGen


def run(tmp, output, **overrides):
    trace = Path(tmp) / "trace.jsonl"
    make_trace(trace)
    outdir = Path(tmp) / "out"
    argv = ["run_llm_verifier.py", "--test-trace", str(trace), "--output-dir", str(outdir)]
    for k, v in overrides.items():
        argv += [f"--{k.replace('_','-')}", str(v)]
    old_argv, old_gen, old_prompt = sys.argv, V.QueryGenerator, V.build_prompt_under_query_budget
    sys.argv = argv
    V.QueryGenerator = fake_generator(output)
    V.build_prompt_under_query_budget = lambda tok, builder, **kw: ("prompt", 100, 12000)
    try:
        V.main()
        return outdir, None
    except SystemExit as exc:
        return outdir, exc
    finally:
        sys.argv, V.QueryGenerator, V.build_prompt_under_query_budget = old_argv, old_gen, old_prompt


failures = []
def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {detail}")
    if not cond: failures.append(name)


# --- 1. truncated output is salvaged, run completes -----------------------------------
with tempfile.TemporaryDirectory() as tmp:
    outdir, exc = run(tmp, TRUNCATED)
    check("truncated run does not abort (salvage keeps it above threshold)", exc is None, str(exc)[:60])
    summary = json.loads((outdir / "llm_verifier_summary.json").read_text())
    check("salvage recorded in parse_modes", summary["parse_modes"].get("salvaged", 0) > 0, str(summary["parse_modes"]))
    check("parse_rate is 1.0 after salvage", summary["parse_rate"] == 1.0, str(summary["parse_rate"]))
    verdicts = json.loads((outdir / "llm_verifier_verdicts.json").read_text())
    fired = sum(1 for v in verdicts if any(x is not None for x in v["verdicts"].values()))
    check("verdicts recovered from truncated output", fired > 0, f"{fired} non-null verdict rows")

# --- 2. genuinely unparseable output aborts early --------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    outdir, exc = run(tmp, "I cannot answer that.")
    check("unparseable run ABORTS", isinstance(exc, SystemExit) and exc.code != 0)
    msg = str(exc)
    check("abort message names truncation + the flag", "query-max-new-tokens" in msg, "")
    calls = [json.loads(l) for l in (outdir / "llm_verifier_calls.jsonl").open()]
    # The guard used to fire on equality with --abort-check-after (~25 calls). Generation is now
    # batched, so it fires on the first CHUNK that reaches the threshold: the bound is
    # abort_check_after + generation_chunk, not 26. What matters is that it aborts on a small
    # multiple of the chunk rather than running the whole trace.
    check("aborted within one chunk of the threshold, not the whole run",
          len(calls) <= 25 + 64 and len(calls) < 200,
          f"{len(calls)} calls before abort")

# --- 3. resume regenerates unusable rows ------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    trace = Path(tmp) / "trace.jsonl"; make_trace(trace)
    outdir = Path(tmp) / "out"; outdir.mkdir(parents=True)
    stale = outdir / "llm_verifier_calls.jsonl"
    with stale.open("w") as fh:
        for i in range(30):
            for h in (0, 1):
                fh.write(json.dumps({"fact_id": i, "hypothesis_idx": h, "candidate_tags": TAGS,
                                     "verdicts_by_tag": {}, "parse_ok": False, "call": {}}) + "\n")
    reusable, skipped = V.load_existing(stale)
    check("stale failed rows are not reusable", len(reusable) == 0 and skipped == 60, f"reusable={len(reusable)} skipped={skipped}")

    # one good row among them must survive resume
    with stale.open("a") as fh:
        fh.write(json.dumps({"fact_id": 0, "hypothesis_idx": 0, "candidate_tags": TAGS,
                             "verdicts_by_tag": {TAGS[0]: {"FAMILY": True, "ROLE": None, "EVENT": None, "confidence": 0.5}},
                             "parse_ok": True, "call": {}}) + "\n")
    reusable, skipped = V.load_existing(stale)
    check("a genuinely good row IS reused", len(reusable) == 1 and (0, 0) in reusable, f"reusable={len(reusable)}")

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}"); sys.exit(1)
print("all verifier-fix checks passed")
