# FHS — Factorized Hypothesis Search for Evidence-to-Taxonomy Retrieval

Code for the two main tables of the paper: **Table 1 (main results)** and **Table 2 (component
ablations)**. Every file here is the code that produced those numbers, not a reimplementation —
see [Provenance](#provenance).

## Task

Given a fact observed in a filing — a source context, a locus identifier with its datatype, and a
value — retrieve the US-GAAP concept a human tagger would assign. There is no natural-language
query in the input; the system has to form one. FHS samples $J$ *factorized* hypotheses about the
fact, renders each into queries, fuses the retrievals by reciprocal-rank fusion with a
cross-hypothesis consensus term, and reranks the fused head with a candidate-level LLM verifier.

## Layout

```
src/
  run_fintagging_grounding_baseline.py   every query mode, the retriever, the shared listwise
                                         selector — one file, one code path for all methods
  ags_frozen_grounding.py                FHS itself; the deployed constants are pinned here
                                         (J=2, beta=0.6, kappa=60, w_cov=1.0, K=200)
  ags_configuration_scoring.py           per-fact configuration state used by consensus scoring
  ags_symbolic_agreement.py (+ .yaml)    the program-driven dimension check — the *rejected*
                                         scoring design reported in Table 2
  ags_seq_verifier_arm.py                FHS-Seq, the matched sequential control (Table 1)
  ags_sequential_arms.py                 shared loop machinery for the sequential arms
  generation_budget.py                   token/call budgeting shared by the arms
  verifier/                              candidate-level verifier
    run_llm_verifier.py                  verdict generation over the fused top-K_v window
    core.py                              fusion, consensus, and the three scoring modes
    dump_reranked_ranking.py             offline re-scoring of a ranking from stored verdicts
    data_prep.py                         trace loading
analysis/
  stage_arm_windows.py                   each ablation arm's OWN fused window, so an arm is
                                         judged on what it ranks first rather than on FHS's head
  check_ablation_window_coverage.py      how much of an arm's head FHS's verdicts would cover
  paired_final_bootstrap.py              paired context-clustered bootstrap on final accuracy
  compute_ags_seq_arm_metrics.py         FHS-Seq metrics
  verify_single_code_path.py             asserts the arms share one code path
  run_ags_component_validation.py        \
  run_ags_coverage_pilot.py               > imported as libraries by the engine and the verifier
  run_ags_verifier_ablation.py           /  (rendering, fact loading, arm definitions)
scripts/
  run_fintagging_grounding_baseline.sh   builds the python argv for any arm
  slurm/                                 one submission wrapper per arm
tools/                                   taxonomy index construction and the split builder
tests/                                   unit tests for the scoring, rendering and fusion paths
```

## Reproducing Table 1 (main results)

Each row is one arm of the same pipeline; only the query mode changes. Retrieval columns are
measured before the shared listwise selector, `Acc.` after it — that convention holds for both
tables.

| row | command |
|---|---|
| Direct retrieval | `sbatch scripts/slurm/apply_server_fintagging_direct_retrieval.sh` |
| One-pass, free-text | `sbatch scripts/slurm/apply_server_fintagging_one_pass_grounding.sh` |
| One-pass, structured | `sbatch scripts/slurm/apply_server_fintagging_one_pass_structured.sh` |
| Parallel, stochastic ($J{=}2$) | `sbatch scripts/slurm/apply_server_fintagging_parallel_sampling.sh` (`QUERY_TEMPERATURE=0.8`, or greedy decoding makes the two samples identical) |
| Decomposed | `sbatch scripts/slurm/apply_server_fintagging_decomposed_retrieval.sh` |
| Intrinsic refinement | `sbatch scripts/slurm/apply_server_fintagging_intrinsic_self_refinement.sh` |
| Feedback refinement | `sbatch scripts/slurm/apply_server_fintagging_retrieval_feedback_refinement.sh` |
| FHS-Seq | `sbatch scripts/slurm/apply_server_fintagging_ags_seq.sh` |
| **FHS (full)** | three stages, below |

All baselines run at `LABEL_COVERAGE_WEIGHT=1.0`: the coverage term is a property of the shared
index, enabled identically for every method.

**FHS (full)** is three stages, because its ranking is produced by an LLM and not by retrieval
alone:

1. `sbatch scripts/slurm/apply_server_fintagging_frozen_ags.sh` — generation and fusion; writes
   the trace every later stage reads.
2. `sbatch --export=ALL,JUDGE_DIMENSIONS=all,TOP_M=10,WINDOW_SOURCE=fused
   scripts/slurm/apply_server_ags_table5_llm_verifier.sh` — verdicts over the fused top-10
   window. `WINDOW_SOURCE=fused` matters: without it the window is cut from the deterministically
   reranked list, which lets the program-driven check decide what the LLM is allowed to see.
3. `sbatch --export=ALL,ARM=no_determ,VERDICTS=<verdicts.json>
   scripts/slurm/apply_server_verifier_ablation_rerank.sh` — re-score with the verifier term and
   run the shared listwise selector.

## Reproducing Table 2 (component ablations)

Every row is stage 3 above with a different `ARM`, over verdicts generated for **that arm's own
fused window** (`analysis/stage_arm_windows.py`). Reusing FHS's verdicts for an arm whose head
differs would silently reproduce the row it is meant to replace; measured head coverage is
0.76–0.84, and `analysis/check_ablation_window_coverage.py` reports it.

| row | how |
|---|---|
| FHS (full) | `ARM=no_determ` over the full window's verdicts |
| $-$ verifier | `ARM=no_verifier` (`beta=0`) |
| Program-driven score | `ARM=no_llm` — the symbolic dimension check supplies the rerank term instead of the LLM |
| $-$ ensemble ($J{=}1$) | `ARM=llmonly_ensemble_idx0` and `idx1`, averaged |
| $-$ factorization | the free-text ensemble, i.e.\ the parallel-sampling arm of Table 1 |
| $-$ label-form / $-$ definition-form | `ARM=llmonly_definition_form` / `llmonly_label_form` (each renders only one form; note the row name states the form *removed*) |
| mean RRF | `ARM=llmonly_mean_fusion` |
| raw fused scores | `ARM=llmonly_raw_scaling` |
| $-$ label coverage | `scripts/slurm/apply_server_wcov0_rerank.sh` — index property, so the whole pipeline reruns at `w_cov=0` |
| Oracle best single | `ARM=llmonly_oracle_single` — best of the $J$ hypotheses chosen with gold, an upper bound, not a method |

`apply_server_verifier_ablation_rerank.sh` carries the exact flag set for every arm and refuses
an arm that was handed the wrong verdict file.

## Provenance

The paper's numbers were produced on a SLURM cluster by this code. Every file here was taken from
the run tree that produced them and checked file by file: a file was accepted from the
release-layout copy only when its only differences were the import preamble and path constants;
otherwise the original was used verbatim. Two consequences worth stating:

- `src/ags_frozen_grounding.py` is the **serial** hypothesis-generation path that produced the
  reported runs. A later variant in our working tree batches the $J$ samples into one call and
  adds $J{=}3,4$ arms; it is distributionally equivalent but not token-identical, and it is
  deliberately **not** included here.
- The SLURM wrappers were edited in exactly two ways: mail-notification directives were removed,
  and absolute cluster paths were replaced by `${REPO_ROOT}` (each script resolves it from its own
  location) and `${TMPDIR}`. No logic was changed; `bash -n` passes on all of them.

## Not included

Run outputs, traces, verdict files and logs (gigabytes, and not needed to re-run); the dataset
splits and the taxonomy index, which are released separately; and the code for the appendix
studies — retriever robustness, efficiency, the verifier-quality and bridge analyses, the beta
sweep, and the query-form probe — none of which Table 1 or Table 2 depends on.

## Requirements

Python 3.11+, `vllm`, `torch`, `transformers`, `rank_bm25`, `numpy`, `pyyaml`. Generation and
reranking use Qwen3-32B through vLLM on one 80GB-class GPU; the offline re-scoring and all
analysis scripts are CPU-only.
