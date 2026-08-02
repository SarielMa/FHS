#!/usr/bin/env bash
set -euo pipefail

DOMAIN_ROOT="$(readlink -f "$(dirname "$0")")"
SHARED_ROOT="$(readlink -f "${DOMAIN_ROOT}/codiesp_pipeline")"

TEST_JSONL="${TEST_JSONL:-${DOMAIN_ROOT}/data/codiesp/facts_test_full_exact.jsonl}"
TAXONOMY_JSONL="${TAXONOMY_JSONL:-${DOMAIN_ROOT}/index/icd10cm_fy2018/icd10cm_fy2018_retrieval.jsonl}"
NORMALIZATION_MAP="${NORMALIZATION_MAP:-${DOMAIN_ROOT}/schema/icd10cm/normalization_map.json}"
RUNS_ROOT="${RUNS_ROOT:-${DOMAIN_ROOT}/runs_codiesp_grounding_baseline}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${DOMAIN_ROOT}/.cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"

QUERY_MODE="${QUERY_MODE:-direct_retrieval}"
case "${QUERY_MODE}" in
  direct|direct_retrieval)
    DEFAULT_METHOD_DIR="qwen3_32b_direct_retrieval"
    ;;
  llm_description|one_pass_grounding)
    DEFAULT_METHOD_DIR="qwen3_32b_one_pass_grounding"
    ;;
  gold_label_definition|gold_label_definition_retrieval|oracle_definition|oracle_retrieval)
    DEFAULT_METHOD_DIR="qwen3_32b_gold_label_definition_retrieval"
    ;;
  intrinsic|intrinsic_self_refinement|self_refinement)
    DEFAULT_METHOD_DIR="qwen3_32b_intrinsic_self_refinement"
    ;;
  feedback|retrieval_feedback|retrieval_feedback_refinement)
    DEFAULT_METHOD_DIR="qwen3_32b_retrieval_feedback_refinement"
    ;;
  parallel|parallel_sampling)
    DEFAULT_METHOD_DIR="qwen3_32b_parallel_sampling"
    ;;
  ags_j1|one_pass_structured|one_pass_grounding_structured)
    DEFAULT_METHOD_DIR="qwen3_32b_one_pass_structured"
    ;;
  ags|frozen_ags|frozen_ags_grounding)
    DEFAULT_METHOD_DIR="qwen3_32b_frozen_ags"
    ;;
  fhs_j1|frozen_ags_j1)
    DEFAULT_METHOD_DIR="qwen3_32b_fhs_j1"
    ;;
  fhs_no_verifier|frozen_ags_no_verifier)
    DEFAULT_METHOD_DIR="qwen3_32b_fhs_no_verifier"
    ;;
  fhs_j3_wcov0|fhs_j3_wcov1|fhs_j4_wcov0|fhs_j4_wcov1)
    DEFAULT_METHOD_DIR="qwen3_32b_${QUERY_MODE}"
    ;;
  *)
    echo "Unsupported CodiEsp QUERY_MODE=${QUERY_MODE}. Expected direct_retrieval|gold_label_definition_retrieval|one_pass_grounding|intrinsic_self_refinement|retrieval_feedback_refinement|parallel_sampling|one_pass_structured|frozen_ags|fhs_j1|fhs_no_verifier|fhs_j3_wcov0|fhs_j3_wcov1|fhs_j4_wcov0|fhs_j4_wcov1." >&2
    exit 1
    ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/${DEFAULT_METHOD_DIR}}"

TOP_K="${TOP_K:-200}"
RETRIEVAL_ROUNDS="${RETRIEVAL_ROUNDS:-4}"
FEEDBACK_CANDIDATE_COUNT="${FEEDBACK_CANDIDATE_COUNT:-10}"
RRF_KAPPA="${RRF_KAPPA:-60.0}"
MEMORY_TOP_K="${MEMORY_TOP_K:-3}"
BANDIT_INITIAL_GROUNDINGS="${BANDIT_INITIAL_GROUNDINGS:-3}"
BANDIT_POSTERIOR_RIDGE="${BANDIT_POSTERIOR_RIDGE:-1.0}"
BANDIT_POSTERIOR_ALPHA="${BANDIT_POSTERIOR_ALPHA:-0.75}"
BANDIT_SEED="${BANDIT_SEED:-20260728}"
BANDIT_QUERY_OVERLAP_THRESHOLD="${BANDIT_QUERY_OVERLAP_THRESHOLD:-0.85}"
BANDIT_MAX_GATE_REJECTIONS="${BANDIT_MAX_GATE_REJECTIONS:-2}"
BANDIT_REPLAY="${BANDIT_REPLAY:-1}"
BANDIT_REWARD_ALPHA="${BANDIT_REWARD_ALPHA:-0.5}"
REUSE_CANDIDATES="${REUSE_CANDIDATES:-1}"
TYPE_FILTER="${TYPE_FILTER:-1}"
RUN_RERANK="${RUN_RERANK:-1}"
RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-32B}"
RERANK_BACKEND="${RERANK_BACKEND:-vllm}"
QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
QUERY_DESCRIPTION_PATH="${QUERY_DESCRIPTION_PATH:-}"
QUERY_CONTEXT_MAX_CHARS="${QUERY_CONTEXT_MAX_CHARS:-12000}"
QUERY_MAX_INPUT_TOKENS="${QUERY_MAX_INPUT_TOKENS:-16000}"
QUERY_MAX_NEW_TOKENS="${QUERY_MAX_NEW_TOKENS:-128}"
FHS_VERIFIER_MAX_NEW_TOKENS="${FHS_VERIFIER_MAX_NEW_TOKENS:-3072}"
QUERY_TEMPERATURE="${QUERY_TEMPERATURE:-0.0}"
QUERY_TOP_P="${QUERY_TOP_P:-1.0}"
BF16="${BF16:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
CONTEXT_MAX_CHARS="${CONTEXT_MAX_CHARS:-12000}"
CANDIDATE_DOC_MAX_CHARS="${CANDIDATE_DOC_MAX_CHARS:-320}"
RERANK_LIST_SIZE="${RERANK_LIST_SIZE:-20}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-30000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
RESUME="${RESUME:-1}"
LOG_EVERY="${LOG_EVERY:-25}"
LABEL_COVERAGE_POOL_MULTIPLIER="${LABEL_COVERAGE_POOL_MULTIPLIER:-10}"
LABEL_COVERAGE_WEIGHT="${LABEL_COVERAGE_WEIGHT:-}"
FROZEN_AGS_TOP_P="${FROZEN_AGS_TOP_P:-1.0}"

case "${QUERY_MODE}" in
  ags|frozen_ags|frozen_ags_grounding|fhs_j1|frozen_ags_j1|fhs_no_verifier|frozen_ags_no_verifier|fhs_j3_wcov0|fhs_j3_wcov1|fhs_j4_wcov0|fhs_j4_wcov1|ags_j1|one_pass_structured|one_pass_grounding_structured)
    if [[ "${QUERY_MAX_NEW_TOKENS}" == "128" ]]; then
      QUERY_MAX_NEW_TOKENS=512
    fi
    ;;
esac

require_path () {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_path "${DOMAIN_ROOT}/scripts/run_codiesp_grounding.py" "CodiEsp prompt shim"
require_path "${SHARED_ROOT}/run_fintagging_grounding_baseline.py" "vendored grounding runner"
require_path "${TEST_JSONL}" "CodiEsp grounding JSONL"
require_path "${TAXONOMY_JSONL}" "ICD-10-CM retrieval taxonomy"
require_path "${NORMALIZATION_MAP}" "ICD-10-CM symbolic normalization map"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${OUTPUT_DIR}"

ARGS=(
  --test-jsonl "${TEST_JSONL}"
  --taxonomy-jsonl "${TAXONOMY_JSONL}"
  --normalization-map "${NORMALIZATION_MAP}"
  --output-dir "${OUTPUT_DIR}"
  --top-k "${TOP_K}"
  --retrieval-rounds "${RETRIEVAL_ROUNDS}"
  --feedback-candidate-count "${FEEDBACK_CANDIDATE_COUNT}"
  --rrf-kappa "${RRF_KAPPA}"
  --memory-top-k "${MEMORY_TOP_K}"
  --bandit-initial-groundings "${BANDIT_INITIAL_GROUNDINGS}"
  --bandit-posterior-ridge "${BANDIT_POSTERIOR_RIDGE}"
  --bandit-posterior-alpha "${BANDIT_POSTERIOR_ALPHA}"
  --bandit-seed "${BANDIT_SEED}"
  --bandit-query-overlap-threshold "${BANDIT_QUERY_OVERLAP_THRESHOLD}"
  --bandit-max-gate-rejections "${BANDIT_MAX_GATE_REJECTIONS}"
  --bandit-reward-alpha "${BANDIT_REWARD_ALPHA}"
  --query-mode "${QUERY_MODE}"
  --rerank-model "${RERANK_MODEL}"
  --rerank-backend "${RERANK_BACKEND}"
  --query-generation-model "${QUERY_GENERATION_MODEL}"
  --query-generation-backend "${QUERY_GENERATION_BACKEND}"
  --query-context-max-chars "${QUERY_CONTEXT_MAX_CHARS}"
  --query-max-input-tokens "${QUERY_MAX_INPUT_TOKENS}"
  --query-max-new-tokens "${QUERY_MAX_NEW_TOKENS}"
  --fhs-verifier-max-new-tokens "${FHS_VERIFIER_MAX_NEW_TOKENS}"
  --query-temperature "${QUERY_TEMPERATURE}"
  --query-top-p "${QUERY_TOP_P}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --vllm-batch-size "${VLLM_BATCH_SIZE}"
  --context-max-chars "${CONTEXT_MAX_CHARS}"
  --candidate-doc-max-chars "${CANDIDATE_DOC_MAX_CHARS}"
  --rerank-list-size "${RERANK_LIST_SIZE}"
  --max-input-tokens "${MAX_INPUT_TOKENS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --label-coverage-pool-multiplier "${LABEL_COVERAGE_POOL_MULTIPLIER}"
  --frozen-ags-top-p "${FROZEN_AGS_TOP_P}"
  --log-every "${LOG_EVERY}"
)

if [[ -n "${LABEL_COVERAGE_WEIGHT}" ]]; then
  ARGS+=(--label-coverage-weight "${LABEL_COVERAGE_WEIGHT}")
fi

if [[ -n "${QUERY_DESCRIPTION_PATH}" ]]; then
  ARGS+=(--query-description-path "${QUERY_DESCRIPTION_PATH}")
fi

if [[ "${TYPE_FILTER}" == "1" ]]; then
  ARGS+=(--type-filter)
else
  ARGS+=(--no-type-filter)
fi

if [[ "${RUN_RERANK}" == "1" ]]; then
  ARGS+=(--run-rerank)
fi

if [[ "${BANDIT_REPLAY}" == "1" ]]; then
  ARGS+=(--bandit-replay)
else
  ARGS+=(--no-bandit-replay)
fi

if [[ "${BF16}" == "1" ]]; then
  ARGS+=(--bf16)
fi

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  ARGS+=(--trust-remote-code)
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi

if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  ARGS+=(--attn-implementation "${ATTN_IMPLEMENTATION}")
fi

if [[ "${RESUME}" == "1" ]]; then
  ARGS+=(--resume)
fi

if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

echo "============================================================"
echo "TASK                  : codiesp_grounding_experiment"
echo "DOMAIN_ROOT           : ${DOMAIN_ROOT}"
echo "SHARED_ROOT           : ${SHARED_ROOT}"
echo "TEST_JSONL            : ${TEST_JSONL}"
echo "TAXONOMY_JSONL        : ${TAXONOMY_JSONL}"
echo "NORMALIZATION_MAP     : ${NORMALIZATION_MAP}"
echo "OUTPUT_DIR            : ${OUTPUT_DIR}"
echo "QUERY_MODE            : ${QUERY_MODE}"
echo "RUN_RERANK            : ${RUN_RERANK}"
echo "QUERY_MAX_NEW_TOKENS  : ${QUERY_MAX_NEW_TOKENS}"
echo "FHS_MAX_NEW_TOKENS    : ${FHS_VERIFIER_MAX_NEW_TOKENS}"
echo "MAX_NEW_TOKENS        : ${MAX_NEW_TOKENS}"
echo "TOP_K                 : ${TOP_K}"
echo "RRF_KAPPA             : ${RRF_KAPPA}"
echo "TYPE_FILTER           : ${TYPE_FILTER}"
echo "LABEL_COVERAGE_WEIGHT : ${LABEL_COVERAGE_WEIGHT:-<default>}"
echo "LABEL_COVERAGE_POOL   : ${LABEL_COVERAGE_POOL_MULTIPLIER}"
echo "HF_HOME               : ${HF_HOME}"
echo "============================================================"

PYTHONPATH="${SHARED_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python "${DOMAIN_ROOT}/scripts/run_codiesp_grounding.py" "${ARGS[@]}"
