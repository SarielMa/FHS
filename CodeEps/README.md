# CodiEsp Diagnosis Grounding Code

This repository contains the code used for the CodiEsp diagnosis-code grounding
experiments reported in the paper. It is a code-only release: prepared data,
retrieval inventories, model outputs, logs, caches, and raw archives are not
included.

## What Is Included

- `scripts/prepare_codiesp_domain.py`: builds the prepared CodiEsp test facts,
  ICD-10-CM retrieval inventory, and schema files from raw inputs.
- `scripts/relocate_codiesp_evidence.py`: relocates CodiEsp evidence into the
  English context used by the grounding experiments.
- `scripts/run_codiesp_grounding.py`: CodiEsp prompt shim.
- `codiesp_pipeline/`: vendored grounding runner and FHS/AGS components used by
  the CodiEsp experiments.
- `run_codiesp_grounding_baseline.sh`: local entry point for one method/config.
- `slurm/`: example Slurm wrappers for the reported experiment arms.
- `codiesp_instantiation.tex`: LaTeX task-instantiation text.

## Data Layout

The prepared data, retrieval index, and schema files are available at
https://huggingface.co/datasets/lm2445/CodeEps_FHS/tree/main.

Place the data artifact contents at the repository root before running the
experiments:

```text
data/codiesp/facts_test_full_exact.jsonl
data/codiesp/evidence_relocations_full_exact.jsonl
data/codiesp/stats_full_exact.json
data/codiesp/test_docs_full_exact.txt
data/codiesp/spotcheck_50_full_exact.tsv

index/icd10cm_fy2018/icd10cm_fy2018_retrieval.jsonl
index/icd10cm_fy2018/code_metadata.jsonl
index/icd10cm_fy2018/inventory_manifest.json
index/icd10cm_fy2018/self_retrieval_probe.json

schema/icd10cm/normalization_map.json
schema/icd10cm/vocab_*.json
```

Those files are intentionally not tracked in this GitHub package.

## Reported Full-Test Configuration

The paper uses the prepared official CodiEsp test slice
`data/codiesp/facts_test_full_exact.jsonl` with ICD-10-CM FY2018 retrieval
inventory `index/icd10cm_fy2018/icd10cm_fy2018_retrieval.jsonl`.

For methods that were run with both label-coverage settings, the release
configuration of interest is `LABEL_COVERAGE_WEIGHT=1.0` (`wcov1`). The gold
label-definition oracle is run once.

Example:

```bash
MODE=full \
QUERY_MODE=frozen_ags \
LABEL_COVERAGE_WEIGHT=1.0 \
OUTPUT_DIR=runs_codiesp_grounding_baseline/qwen3_32b_frozen_ags_full_wcov1 \
bash run_codiesp_grounding_baseline.sh
```

Common `QUERY_MODE` values:

```text
direct_retrieval
one_pass_grounding
one_pass_structured
parallel_sampling
frozen_ags
fhs_j1
fhs_no_verifier
intrinsic_self_refinement
retrieval_feedback_refinement
gold_label_definition_retrieval
```

## Slurm

The scripts under `slurm/` are examples matching the local experiment framework.
They use repository-relative paths and do not contain user-specific accounts,
email addresses, or absolute filesystem paths. Adjust partition, GPU, module,
and conda environment names for a different cluster.

## Outputs

By default, runs write to `runs_codiesp_grounding_baseline/`. That directory is
ignored by Git because candidate lists, traces, rerank predictions, and metrics
can be large.
