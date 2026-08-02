#!/bin/bash
#SBATCH --job-name=fintag_frozen_ags
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=logs/%j_fintag_frozen_ags_qwen3_b200.txt

set -euo pipefail

# Frozen AGS single-pass grounding comparison (gold-input grounding experiment):
#   J=2 structured hypotheses -> def/lab render -> coverage retrieval -> sum-RRF ->
#   range-normalize -> agree rerank. Runs on the gold FinTagging grounding test JSONL
#   (no extractor). The extractor-driven variant fed the end-to-end table, which the paper
#   does not report, so it is not part of this release.
#   The listwise rerank runs on the AGS ranking so the method is scored at the same stage as
#   every other query mode (bm25_retrieval + qwen_reranked in metrics.json).

FHS_ROOT="${FHS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REPO_ROOT="${FHS_ROOT}"

export SCRATCH_ROOT="${SCRATCH_ROOT:-${TMPDIR:-/tmp}/fhs-scratch}"
export RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs/runs_fintagging_grounding_baseline}"
export QUERY_MODE="${QUERY_MODE:-frozen_ags}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/qwen3_32b_frozen_ags}"
export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
# Full eval: BM25/AGS retrieval stage + listwise rerank, same as the other methods.
# REUSE_CANDIDATES=1 reuses the existing bm25_candidates.jsonl, so a rerun is rerank-only.
export RUN_RERANK="${RUN_RERANK:-1}"
export REUSE_CANDIDATES="${REUSE_CANDIDATES:-1}"
export RESUME="${RESUME:-1}"

bash "${REPO_ROOT}/scripts/slurm/apply_server_fintagging_direct_retrieval.sh"
