#!/bin/bash
#SBATCH --job-name=abl_wcov0
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=logs/%j_abl_wcov0.txt

set -euo pipefail

# tab:ablation's "- label coverage (w_cov=0)" Final column.
#
# Unlike every other arm in that table, this one cannot reorder the frozen trace's candidates:
# w_cov weights the retrieval index, so switching it off changes which concepts are retrieved.
# Stage 1 therefore re-retrieves the whole split at w_cov=0 (CPU, slow) before stage 2 applies
# the same listwise reranker the other arms use.

FHS_ROOT="${FHS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REPO_ROOT="${FHS_ROOT}"
cd "${REPO_ROOT}"

# Environment, copied verbatim from apply_server_verifier_ablation_rerank.sh. The first
# submission (19968987) died at exit 127 in three seconds with an empty log because it did
# `source ~/.bashrc; conda activate` instead: `conda` is not a command on a compute node, and
# under `set -euo pipefail` that aborts the job before any Python runs. The B200 needs the
# finben_b200 env specifically -- the default finben env's PyTorch is built for sm_50-sm_90
# and aborts on this node's sm_100 during vLLM engine init.
unset -f conda 2>/dev/null || true
unset -f __conda_activate 2>/dev/null || true
unset -f __conda_reactivate 2>/dev/null || true
unset -f __conda_hashr 2>/dev/null || true

if ! command -v conda >/dev/null 2>&1; then
  conda() { return 0; }
  export -f conda
  _FAKE_CONDA_FOR_PURGE=1
fi

module --force purge || true
if [[ "${_FAKE_CONDA_FOR_PURGE:-0}" == "1" ]]; then
  unset -f conda || true
  unset _FAKE_CONDA_FOR_PURGE
fi

module load StdEnv || true
module load CUDA/12.8.0

export CUDA_HOME
CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export TRITON_CACHE_DIR="/tmp/${USER}/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}"

export SCRATCH_ROOT="${SCRATCH_ROOT:-${TMPDIR:-/tmp}/fhs-scratch}"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"

module load miniconda
if [[ -n "${EBROOTMINICONDA:-}" && -f "${EBROOTMINICONDA}/etc/profile.d/conda.sh" ]]; then
  source "${EBROOTMINICONDA}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  source "$(cd "$(dirname "$(command -v conda)")/.." && pwd)/etc/profile.d/conda.sh"
else
  echo "Failed to initialize conda after loading the miniconda module." >&2
  exit 1
fi
conda activate finben_b200

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs_ags_verifier_ablation/qwen3_32b/rerank_wcov0}"
RANKING="${OUTPUT_DIR}/bm25_candidates.jsonl"
mkdir -p "${OUTPUT_DIR}"

export TEST_JSONL="${TEST_JSONL:-${REPO_ROOT}/data/test/test.jsonl}"
export RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-32B}"
export RERANK_BACKEND="${RERANK_BACKEND:-vllm}"
export RERANK_LIST_SIZE="${RERANK_LIST_SIZE:-20}"

# VERDICTS attaches this arm's OWN verdicts, generated against its OWN w_cov=0 window
# (stage_arm_windows.py on the stage-1 dump -> run_llm_verifier.py --window-tags). Without it
# the rerank term has no LLM support to consume and the row is a re-retrieval with no
# candidate-level verifier, which is not comparable to the other rows of tab:ablation.
if [[ -n "${VERDICTS:-}" && ! -f "${VERDICTS}" ]]; then
  echo "VERDICTS=${VERDICTS} does not exist." >&2
  exit 1
fi

if [[ ! -f "${RANKING}" ]]; then
  echo "--- re-retrieving at w_cov=0 and dumping ranking ---"
  echo "    verdicts: ${VERDICTS:-<none: no candidate-level verifier>}"
  python "${REPO_ROOT}/ags_table5_ablation/dump_index_ablation_ranking.py" \
    --label-coverage-weight 0.0 \
    --beta 0.6 \
    ${VERDICTS:+--llm-verifier-verdicts "${VERDICTS}"} \
    --output "${RANKING}.partial.${SLURM_JOB_ID:-$$}" \
    --summary "${OUTPUT_DIR}/ranking_summary.json" \
    ${LIMIT:+--limit "${LIMIT}"}
  mv "${RANKING}.partial.${SLURM_JOB_ID:-$$}" "${RANKING}"
else
  echo "--- ranking already staged, reusing ---"
fi

echo "--- listwise rerank (w_cov=0) ---"
python "${REPO_ROOT}/run_fintagging_grounding_baseline.py" \
  --test-jsonl "${TEST_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --query-mode frozen_ags \
  --reuse-candidates \
  --run-rerank \
  --rerank-model "${RERANK_MODEL}" \
  --rerank-backend "${RERANK_BACKEND}" \
  --rerank-list-size "${RERANK_LIST_SIZE}" \
  ${LIMIT:+--limit "${LIMIT}"}

echo "Done -> ${OUTPUT_DIR}/metrics.json"
