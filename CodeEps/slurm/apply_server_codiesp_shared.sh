#!/bin/bash
set -euo pipefail

MODE="${MODE:-full}"
DOMAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_SH="${DOMAIN_ROOT}/run_codiesp_grounding_baseline.sh"

for var in CONDA_EXE CONDA_PREFIX CONDA_PREFIX_1 CONDA_PREFIX_2 CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE CONDA_PKGS_DIRS CONDA_ENVS_PATH _CE_CONDA _CE_M _CONDA_EXE _CONDA_ROOT; do
  unset "${var}" || true
done
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

export SCRATCH_ROOT="${SCRATCH_ROOT:-${DOMAIN_ROOT}/.cache}"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}"

module load miniconda

if [[ -n "${EBROOTMINICONDA:-}" && -f "${EBROOTMINICONDA}/etc/profile.d/conda.sh" ]]; then
  source "${EBROOTMINICONDA}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
  CONDA_BASE="$(cd "$(dirname "${CONDA_BIN}")/.." && pwd)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "Failed to initialize conda after loading the miniconda module." >&2
  exit 1
fi

conda activate finben_b200

which nvcc
nvcc --version
which python
python -c "import torch; print('torch cuda:', torch.version.cuda); print('gpus:', torch.cuda.device_count())"
nvidia-smi

cd "${DOMAIN_ROOT}"
[[ -f "${PIPELINE_SH}" ]] || { echo "Missing pipeline script: ${PIPELINE_SH}" >&2; exit 1; }

export TEST_JSONL="${TEST_JSONL:-${DOMAIN_ROOT}/data/codiesp/facts_test_full_exact.jsonl}"
export TAXONOMY_JSONL="${TAXONOMY_JSONL:-${DOMAIN_ROOT}/index/icd10cm_fy2018/icd10cm_fy2018_retrieval.jsonl}"
export NORMALIZATION_MAP="${NORMALIZATION_MAP:-${DOMAIN_ROOT}/schema/icd10cm/normalization_map.json}"
export RUNS_ROOT="${RUNS_ROOT:-${DOMAIN_ROOT}/runs_codiesp_grounding_baseline}"
export TOP_K="${TOP_K:-200}"
export RRF_KAPPA="${RRF_KAPPA:-60.0}"
export REUSE_CANDIDATES="${REUSE_CANDIDATES:-1}"
export TYPE_FILTER="${TYPE_FILTER:-1}"
export RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-32B}"
export RERANK_BACKEND="${RERANK_BACKEND:-vllm}"
export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
export QUERY_CONTEXT_MAX_CHARS="${QUERY_CONTEXT_MAX_CHARS:-12000}"
export QUERY_MAX_INPUT_TOKENS="${QUERY_MAX_INPUT_TOKENS:-16000}"
export QUERY_MAX_NEW_TOKENS="${QUERY_MAX_NEW_TOKENS:-128}"
export FHS_VERIFIER_MAX_NEW_TOKENS="${FHS_VERIFIER_MAX_NEW_TOKENS:-3072}"
export QUERY_TEMPERATURE="${QUERY_TEMPERATURE:-0.0}"
export QUERY_TOP_P="${QUERY_TOP_P:-1.0}"
export BF16="${BF16:-1}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
export CONTEXT_MAX_CHARS="${CONTEXT_MAX_CHARS:-12000}"
export CANDIDATE_DOC_MAX_CHARS="${CANDIDATE_DOC_MAX_CHARS:-320}"
export RERANK_LIST_SIZE="${RERANK_LIST_SIZE:-20}"
export MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-30000}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export TOP_P="${TOP_P:-1.0}"
export RESUME="${RESUME:-1}"
export LOG_EVERY="${LOG_EVERY:-25}"

case "${MODE}" in
  full)
    export RUN_RERANK=1
    ;;
  retrieval)
    export RUN_RERANK=0
    ;;
  dryrun)
    export RUN_RERANK=0
    export LIMIT="${LIMIT:-2}"
    ;;
  *)
    echo "Unknown MODE=${MODE}. Expected full|retrieval|dryrun." >&2
    exit 1
    ;;
esac

echo "============================================================"
echo "DOMAIN_ROOT=${DOMAIN_ROOT}"
echo "PIPELINE_SH=${PIPELINE_SH}"
echo "MODE=${MODE}"
echo "TEST_JSONL=${TEST_JSONL}"
echo "TAXONOMY_JSONL=${TAXONOMY_JSONL}"
echo "NORMALIZATION_MAP=${NORMALIZATION_MAP}"
echo "OUTPUT_DIR=${OUTPUT_DIR:-<runner default>}"
echo "QUERY_MODE=${QUERY_MODE}"
echo "RUN_RERANK=${RUN_RERANK}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "============================================================"

python -m py_compile \
  "${DOMAIN_ROOT}/scripts/run_codiesp_grounding.py" \
  "${DOMAIN_ROOT}/scripts/prepare_codiesp_domain.py" \
  "${DOMAIN_ROOT}/codiesp_pipeline/run_fintagging_grounding_baseline.py" \
  "${DOMAIN_ROOT}/codiesp_pipeline/ags_frozen_grounding.py" \
  "${DOMAIN_ROOT}/codiesp_pipeline/ags_configuration_scoring.py" \
  "${DOMAIN_ROOT}/codiesp_pipeline/ags_symbolic_agreement.py" \
  "${DOMAIN_ROOT}/codiesp_pipeline/run_ags_component_validation.py"
bash "${PIPELINE_SH}"
