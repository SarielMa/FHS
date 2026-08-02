#!/usr/bin/env bash
set -euo pipefail

# Grounding experiments:
#   Method-specific query generation -> BM25 candidates -> shared Qwen reranking/eval.

REPO_ROOT="${FHS_ROOT:-$(readlink -f "$(dirname "$0")/..")}"
PROJECT_ROOT="$(readlink -f "${REPO_ROOT}/..")"

TEST_JSONL="${TEST_JSONL:-${REPO_ROOT}/data/test/test.jsonl}"
TAXONOMY_JSONL="${TAXONOMY_JSONL:-${PROJECT_ROOT}/retrieval_data/us_gaap_2024_enriched/us_gaap_2024_enriched_retrieval.jsonl}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${TMPDIR:-/tmp}/fhs-scratch}"
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
  intrinsic|self_refinement|intrinsic_self_refinement)
    DEFAULT_METHOD_DIR="qwen3_32b_intrinsic_self_refinement"
    ;;
  feedback|retrieval_feedback|retrieval_feedback_refinement)
    DEFAULT_METHOD_DIR="qwen3_32b_retrieval_feedback_refinement"
    ;;
  parallel|parallel_sampling)
    DEFAULT_METHOD_DIR="qwen3_32b_parallel_sampling"
    ;;
  decomposed|decomposed_retrieval)
    DEFAULT_METHOD_DIR="qwen3_32b_decomposed_retrieval"
    ;;
  operator|operator_refinement)
    DEFAULT_METHOD_DIR="qwen3_32b_operator_refinement"
    ;;
  memory|memory_guided_refinement)
    DEFAULT_METHOD_DIR="qwen3_32b_memory_guided_refinement"
    ;;
  ags|frozen_ags|frozen_ags_grounding)
    DEFAULT_METHOD_DIR="qwen3_32b_frozen_ags"
    ;;
  ags_seq|ags_sequential)
    DEFAULT_METHOD_DIR="qwen3_32b_ags_seq"
    ;;
  ags_seq_random|ags_sequential_random)
    DEFAULT_METHOD_DIR="qwen3_32b_ags_seq_random"
    ;;
  *)
    DEFAULT_METHOD_DIR="qwen3_32b_${QUERY_MODE}"
    ;;
esac
DEFAULT_OUTPUT_DIR="${REPO_ROOT}/runs_fintagging_grounding_baseline/${DEFAULT_METHOD_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

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
# SHARED GENERATION BUDGET. This was 128 with a per-mode bump to 512 for the frozen family
# only, which starved four baselines the paper compares against: measured at the 128 cap,
# one_pass_structured truncated 8.5% of its generations (11/200 parse failures),
# parallel_sampling_diversity 33.1% (397/1200), intrinsic_self_refinement 5.9% and
# retrieval_feedback_refinement 4.2% -- every failure sat exactly at 127-128 tokens, while
# frozen_ags at 512 never touched its cap (longest output 161) and ags_seq never did either
# (longest 245). A method-dependent output budget is not a budget-matched comparison, so the
# cap is now uniform and set well above every measured length (longest observed: frozen_ags
# 161, ags_seq 245, decomposed 235, parallel_sampling 102). 512 would probably have been enough,
# but the four truncated methods' true output lengths are unobservable from truncated data, so
# the cap is set high rather than guessed and the run reports any call that still reaches it --
# vLLM stops at EOS, so a generous cap costs nothing when the model finishes on its own.
QUERY_MAX_NEW_TOKENS="${QUERY_MAX_NEW_TOKENS:-2048}"
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
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
RESUME="${RESUME:-1}"
LOG_EVERY="${LOG_EVERY:-25}"
LIMIT="${LIMIT:-}"
NORMALIZATION_MAP="${NORMALIZATION_MAP:-}"
LABEL_COVERAGE_POOL_MULTIPLIER="${LABEL_COVERAGE_POOL_MULTIPLIER:-0}"
FROZEN_AGS_TOP_P="${FROZEN_AGS_TOP_P:-1.0}"
AGS_SEQ_MAX_ROUNDS="${AGS_SEQ_MAX_ROUNDS:-4}"
AGS_SEQ_SLATE_LIMIT="${AGS_SEQ_SLATE_LIMIT:-6}"
AGS_SEQ_FEEDBACK_TOP_M="${AGS_SEQ_FEEDBACK_TOP_M:-10}"
# Spec section 3: the gate stays off so the sequential arm reaches its full B=4 budget.
AGS_SEQ_NOVELTY_GATE="${AGS_SEQ_NOVELTY_GATE:-0}"
AGS_SEQ_NOVELTY_THRESHOLD="${AGS_SEQ_NOVELTY_THRESHOLD:-0.02}"
AGS_SEQ_REWARD_ALPHA="${AGS_SEQ_REWARD_ALPHA:-0.5}"
AGS_SEQ_POSTERIOR_RIDGE="${AGS_SEQ_POSTERIOR_RIDGE:-1.0}"
AGS_SEQ_POSTERIOR_SIGMA="${AGS_SEQ_POSTERIOR_SIGMA:-1.0}"
AGS_SEQ_POSTERIOR_NU="${AGS_SEQ_POSTERIOR_NU:-0.75}"
AGS_SEQ_POSTERIOR_FORGETTING="${AGS_SEQ_POSTERIOR_FORGETTING:-0.995}"
AGS_SEQ_SEED="${AGS_SEQ_SEED:-20260724}"

# frozen_ags produces its own final ranking deterministically (fuse + agree rerank), but the
# downstream listwise rerank still runs so the method is comparable to every other query mode
# at the same evaluation stage. Use MODE=retrieval on the apply_server_* wrapper for the
# retrieval-only variant (those wrappers export RUN_RERANK unconditionally per MODE).
# Structured hypotheses need more than the 128-token query default; bump it if unset.
case "${QUERY_MODE}" in
  ags|frozen_ags|frozen_ags_grounding|ags_seq|ags_sequential|ags_seq_random|ags_sequential_random)
    if [[ "${QUERY_MAX_NEW_TOKENS}" == "128" ]]; then
      QUERY_MAX_NEW_TOKENS=512
    fi
    ;;
  # seq_verifier calls the candidate-level verifier INSIDE the loop, so this cap also bounds a
  # verdict array over K_v candidates x every judged dimension -- not just one hypothesis. At 128
  # (and at 512) every response is cut off, salvage recovers only the first few candidates, and
  # the loop then sees "no unsupported dimension" and stops after round one: measured 0/171 clean
  # parses and 70/75 facts stopping at round 1 before this was raised.
  seq_verifier)
    if [[ "${QUERY_MAX_NEW_TOKENS}" == "128" || "${QUERY_MAX_NEW_TOKENS}" == "512" ]]; then
      QUERY_MAX_NEW_TOKENS=2816
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

require_path "${REPO_ROOT}/run_fintagging_grounding_baseline.py" "grounding baseline script"
require_path "${TEST_JSONL}" "context tagging test JSONL"
require_path "${TAXONOMY_JSONL}" "enriched retrieval taxonomy"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

ARGS=(
  --test-jsonl "${TEST_JSONL}"
  --taxonomy-jsonl "${TAXONOMY_JSONL}"
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
  --ags-seq-max-rounds "${AGS_SEQ_MAX_ROUNDS}"
  --ags-seq-slate-limit "${AGS_SEQ_SLATE_LIMIT}"
  --ags-seq-feedback-top-m "${AGS_SEQ_FEEDBACK_TOP_M}"
  --ags-seq-novelty-threshold "${AGS_SEQ_NOVELTY_THRESHOLD}"
  --ags-seq-reward-alpha "${AGS_SEQ_REWARD_ALPHA}"
  --ags-seq-posterior-ridge "${AGS_SEQ_POSTERIOR_RIDGE}"
  --ags-seq-posterior-sigma "${AGS_SEQ_POSTERIOR_SIGMA}"
  --ags-seq-posterior-nu "${AGS_SEQ_POSTERIOR_NU}"
  --ags-seq-posterior-forgetting "${AGS_SEQ_POSTERIOR_FORGETTING}"
  --ags-seq-seed "${AGS_SEQ_SEED}"
  --log-every "${LOG_EVERY}"
)

if [[ "${AGS_SEQ_NOVELTY_GATE}" == "1" ]]; then
  ARGS+=(--ags-seq-novelty-gate)
else
  ARGS+=(--no-ags-seq-novelty-gate)
fi

if [[ -n "${NORMALIZATION_MAP}" ]]; then
  ARGS+=(--normalization-map "${NORMALIZATION_MAP}")
fi

if [[ "${REUSE_CANDIDATES}" == "1" ]]; then
  ARGS+=(--reuse-candidates)
fi

# Forces w_cov for any query mode. Unset by default, so every existing run behaves exactly as
# before; the python side refuses to override the frozen_ags family, which pins its own value.
if [[ -n "${LABEL_COVERAGE_WEIGHT:-}" ]]; then
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

if [[ -n "${LIMIT}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

echo "============================================================"
echo "TASK                 : grounding_experiment"
echo "TEST_JSONL           : ${TEST_JSONL}"
echo "TAXONOMY_JSONL       : ${TAXONOMY_JSONL}"
echo "OUTPUT_DIR           : ${OUTPUT_DIR}"
echo "HF_HOME              : ${HF_HOME}"
echo "HF_HUB_CACHE         : ${HF_HUB_CACHE}"
echo "TOP_K                : ${TOP_K}"
echo "RETRIEVAL_ROUNDS     : ${RETRIEVAL_ROUNDS}"
echo "FEEDBACK_CANDIDATE_COUNT: ${FEEDBACK_CANDIDATE_COUNT}"
echo "RRF_KAPPA            : ${RRF_KAPPA}"
echo "MEMORY_TOP_K         : ${MEMORY_TOP_K}"
echo "BANDIT_INITIAL_GROUNDINGS: ${BANDIT_INITIAL_GROUNDINGS}"
echo "BANDIT_POSTERIOR_ALPHA: ${BANDIT_POSTERIOR_ALPHA}"
echo "BANDIT_QUERY_OVERLAP_THRESHOLD: ${BANDIT_QUERY_OVERLAP_THRESHOLD}"
echo "BANDIT_REPLAY        : ${BANDIT_REPLAY}"
echo "QUERY_MODE           : ${QUERY_MODE}"
echo "AGS_SEQ_MAX_ROUNDS   : ${AGS_SEQ_MAX_ROUNDS}"
echo "AGS_SEQ_NOVELTY_GATE : ${AGS_SEQ_NOVELTY_GATE}"
echo "AGS_SEQ_SEED         : ${AGS_SEQ_SEED}"
echo "REUSE_CANDIDATES     : ${REUSE_CANDIDATES}"
echo "TYPE_FILTER          : ${TYPE_FILTER}"
echo "RUN_RERANK           : ${RUN_RERANK}"
echo "RERANK_MODEL         : ${RERANK_MODEL}"
echo "RERANK_BACKEND       : ${RERANK_BACKEND}"
echo "QUERY_GENERATION_MODEL: ${QUERY_GENERATION_MODEL}"
echo "QUERY_GENERATION_BACKEND: ${QUERY_GENERATION_BACKEND}"
echo "QUERY_DESCRIPTION_PATH: ${QUERY_DESCRIPTION_PATH:-<default>}"
echo "QUERY_MAX_NEW_TOKENS : ${QUERY_MAX_NEW_TOKENS}"
echo "TENSOR_PARALLEL_SIZE : ${TENSOR_PARALLEL_SIZE}"
echo "GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION}"
echo "MAX_NUM_SEQS         : ${MAX_NUM_SEQS}"
echo "VLLM_BATCH_SIZE      : ${VLLM_BATCH_SIZE}"
echo "CONTEXT_MAX_CHARS    : ${CONTEXT_MAX_CHARS}"
echo "CANDIDATE_DOC_MAX_CHARS: ${CANDIDATE_DOC_MAX_CHARS}"
echo "MAX_INPUT_TOKENS     : ${MAX_INPUT_TOKENS}"
echo "MAX_NEW_TOKENS       : ${MAX_NEW_TOKENS}"
echo "RESUME               : ${RESUME}"
echo "LIMIT                : ${LIMIT:-<none>}"
echo "============================================================"

python "${REPO_ROOT}/run_fintagging_grounding_baseline.py" "${ARGS[@]}"
