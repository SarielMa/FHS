#!/bin/bash
#SBATCH --job-name=fintag_parallel
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=logs/%j_fintag_parallel_sampling_qwen3_b200.txt

set -euo pipefail

# Parallel sampling:
#   Qwen generates B independent retrieval hypotheses and RRF fuses their candidates.

FHS_ROOT="${FHS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REPO_ROOT="${FHS_ROOT}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-${TMPDIR:-/tmp}/fhs-scratch}"
export RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs/runs_fintagging_grounding_baseline}"
export QUERY_MODE="${QUERY_MODE:-parallel_sampling}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/qwen3_32b_parallel_sampling}"
export RETRIEVAL_ROUNDS="${RETRIEVAL_ROUNDS:-4}"
export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
export RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-32B}"
export RERANK_BACKEND="${RERANK_BACKEND:-vllm}"
export REUSE_CANDIDATES="${REUSE_CANDIDATES:-1}"
export RESUME="${RESUME:-1}"

bash "${REPO_ROOT}/scripts/slurm/apply_server_fintagging_direct_retrieval.sh"
