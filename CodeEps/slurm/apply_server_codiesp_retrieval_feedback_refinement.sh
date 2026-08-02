#!/bin/bash
#SBATCH --job-name=codiesp_feedback
#SBATCH --mail-type=ALL
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=logs/%j_codiesp_retrieval_feedback_refinement_qwen3_b200.txt

set -euo pipefail

# Retrieval-feedback refinement:
#   Qwen revises the diagnosis retrieval query for B rounds after seeing the top
#   retrieved candidates from the previous round.

DOMAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QUERY_MODE="${QUERY_MODE:-retrieval_feedback_refinement}"
export OUTPUT_DIR="${OUTPUT_DIR:-${DOMAIN_ROOT}/runs_codiesp_grounding_baseline/qwen3_32b_retrieval_feedback_refinement}"
export RETRIEVAL_ROUNDS="${RETRIEVAL_ROUNDS:-4}"
export FEEDBACK_CANDIDATE_COUNT="${FEEDBACK_CANDIDATE_COUNT:-10}"
export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
export RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-32B}"
export RERANK_BACKEND="${RERANK_BACKEND:-vllm}"
export REUSE_CANDIDATES="${REUSE_CANDIDATES:-1}"
export RESUME="${RESUME:-1}"

bash "${DOMAIN_ROOT}/slurm/apply_server_codiesp_shared.sh"
