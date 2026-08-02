#!/bin/bash
#SBATCH --job-name=bl_wcov1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=logs/%j_bl_wcov1.txt

# Stage 2 of stage_baselines_wcov1_then_submit.sh: the Qwen3-32B listwise rerank that produces
# tab:main_results' Acc./std columns for the two coverage-enabled baselines. Retrieval is already
# on disk, so this reuses candidates and only reranks.
#
# Rerank settings are copied from the published runs' metrics.json (vllm, Qwen/Qwen3-32B) so the
# Acc. column stays comparable to every other row of the table; the coverage term is the only
# thing that differs from the published baseline.
set -euo pipefail

QUERY_MODE="${1:?usage: $0 <query_mode> <output_dir>}"
OUTPUT_DIR="${2:?usage: $0 <query_mode> <output_dir>}"

FHS_ROOT="${FHS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REPO_ROOT="${FHS_ROOT}"
CONDA_ENV="${CONDA_ENV:-finben_b200}"
TEST_JSONL="${TEST_JSONL:-${REPO_ROOT}/data/test/test.jsonl}"
RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-32B}"
RERANK_BACKEND="${RERANK_BACKEND:-vllm}"
RERANK_LIST_SIZE="${RERANK_LIST_SIZE:-20}"

if [[ ! -f "${OUTPUT_DIR}/bm25_candidates.jsonl" && ! -f "${OUTPUT_DIR}/metrics.json" ]]; then
  echo "no staged retrieval in ${OUTPUT_DIR}; run stage_baselines_wcov1_then_submit.sh first" >&2
  exit 1
fi

source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "${CONDA_ENV}" 2>/dev/null || true

echo "--- listwise rerank, ${QUERY_MODE}, w_cov=1.0 ---"
python "${REPO_ROOT}/src/run_fintagging_grounding_baseline.py" \
  --test-jsonl "${TEST_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --query-mode "${QUERY_MODE}" \
  --label-coverage-weight 1.0 \
  --reuse-candidates \
  --run-rerank \
  --rerank-model "${RERANK_MODEL}" \
  --rerank-backend "${RERANK_BACKEND}" \
  --rerank-list-size "${RERANK_LIST_SIZE}"

echo "Done -> ${OUTPUT_DIR}/metrics.json"
