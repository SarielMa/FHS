#!/bin/bash
#SBATCH --job-name=ags_table5_llm_verifier
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=logs/%j_ags_table5_llm_verifier_qwen3_b200.txt

set -euo pipefail

# Table 5 row 3.9: + LLM verification layer (comparing_methods/ags_table5_ablation_spec.md).
# The only row needing new generation: ~2 x 2,509 batched calls re-judging FAMILY/ROLE/EVENT
# on AGS's own top-M=10 cluster representatives per hypothesis.
#
# Standalone by design, modeled on apply_server_fintagging_frozen_ags.sh's environment setup
# but calling ags_table5_ablation/run_llm_verifier.py directly -- it does not go through
# run_fintagging_grounding_baseline.sh's --query-mode dispatch, which other experiments
# (e.g. the AGS+Seq sequential-arm jobs) run against concurrently. Reads AGS's already-
# computed test-split trace read-only; writes only to its own runs_ags_table5_ablation/ tree.
#
# After this completes, re-run apply_server_ags_table5_offline.sh (or just
# ags_table5_ablation/run_test_rows.py) so ablation.csv picks up the verdicts file and folds
# row 3.9 into the table.
#
# Examples:
#   sbatch apply_server_ags_table5_llm_verifier.sh
#   sbatch --export=ALL,LIMIT=20 apply_server_ags_table5_llm_verifier.sh   # smoke test

FHS_ROOT="${FHS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REPO_ROOT="${FHS_ROOT}"

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

export SCRATCH_ROOT="${SCRATCH_ROOT:-${TMPDIR:-/tmp}/fhs-scratch}"
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

cd "${REPO_ROOT}"

if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
export TOP_M="${TOP_M:-10}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"
# fused = window cut from the fused retrieval ranking, before either verifier. The old
# behaviour ("deployed") cut it from final_candidates, which is already deterministically
# reranked, so the deterministic verifier chose what the LLM was shown even in the arms that
# remove it. Keep this at fused for anything feeding a verifier ablation.
export WINDOW_SOURCE="${WINDOW_SOURCE:-fused}"
# vLLM only batches if the script hands it more than one prompt at a time; the old loop called
# generate_one, so a B200 ran at batch size 1.
export GENERATION_CHUNK="${GENERATION_CHUNK:-64}"
# SYMBOLIC_HINT=1 hands the deterministic D- verdict to the LLM as prompt context instead of
# using it as a scoring term. Score the resulting verdicts with verifier_mode=llm_drop.
# HINT_DIMENSIONS selects what the hint may name. "llm" (default) restricts it to
# FAMILY/ROLE/EVENT -- the dimensions the LLM already judges -- which is what the first hint
# run used, and which withholds exactly the dimensions the symbolic layer uniquely covers.
# "all" passes every resolved dimension and widens the hypothesis shown in the prompt to match.
HINT_ARGS=()
if [[ "${SYMBOLIC_HINT:-0}" == "1" ]]; then
  HINT_ARGS=(--symbolic-hint --hint-dimensions "${HINT_DIMENSIONS:-llm}")
fi
# JUDGE_DIMENSIONS=all asks the LLM for QUALIFIER/SCOPE/TEMPORAL as well -- the control for the
# hint experiment. A verdict entry roughly doubles in length, so the output cap has to grow with
# it: at 3 dimensions an entry is ~110-130 tokens and top_m=10 fits in 1536, at 6 it does not,
# and an undersized cap truncates every response mid-array (that failure mode once produced a
# 100% parse failure across a full run).
JUDGE_ARGS=(--judge-dimensions "${JUDGE_DIMENSIONS:-llm}")
# ASK_DECISIVE=1 adds one output field: which dimensions actually discriminate among the
# candidates shown. No extra call, and the per-dimension verdicts are unchanged, so the file
# stays a superset of a normal run.
if [[ "${ASK_DECISIVE:-0}" == "1" ]]; then
  JUDGE_ARGS+=(--ask-decisive-dimensions)
fi
# The cap must follow the SIZE of the judged set, not the spelling of the flag. Since
# 2026-07-30 the default judged set is all six dimensions, so the default cap is the wide one, and
# an undersized cap truncates responses mid-array, which reads as total abstention rather than as
# an error. Only the legacy three-dimension mode may use the narrow cap.
# MEASURED on the finished ask-6 runs (2026-07-30, 5,018 + 2,752 calls, top_m=10): mean completion
# 887-891 tokens, max 1,137, and hit_token_cap = 0. Ask-3 was 610 mean / 765 max. So six dimensions
# cost ~46% more completion, not the ~2x an earlier comment here estimated, and the responses would
# in fact have fitted in 1536; 2816 is headroom that costs nothing, not a requirement. Do not use
# that old estimate to size anything -- read hit_token_cap out of the call log instead.
if [[ "${JUDGE_DIMENSIONS:-llm}" == "legacy" ]]; then
  QUERY_MAX_NEW_TOKENS="${QUERY_MAX_NEW_TOKENS:-1536}"
fi
QUERY_MAX_NEW_TOKENS="${QUERY_MAX_NEW_TOKENS:-2816}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs_ags_table5_ablation/qwen3_32b}"
RESUME_ARGS=()
if [[ "${RESUME:-1}" == "1" ]]; then
  RESUME_ARGS=(--resume)
fi
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

echo "============================================================"
echo "REPO_ROOT=${REPO_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "QUERY_GENERATION_MODEL=${QUERY_GENERATION_MODEL}"
echo "QUERY_GENERATION_BACKEND=${QUERY_GENERATION_BACKEND}"
echo "TOP_M=${TOP_M}"
echo "WINDOW_SOURCE=${WINDOW_SOURCE}"
echo "GENERATION_CHUNK=${GENERATION_CHUNK}"
echo "LIMIT=${LIMIT:-<none>}"
echo "============================================================"

# Pre-flight: the flag values this wrapper passes must exist in the script it calls. A renamed
# choice (--judge-dimensions llm -> legacy happened once) otherwise costs a full GPU allocation
# and fails in argparse a minute later, after the queue wait.
for _flag in "--judge-dimensions ${JUDGE_DIMENSIONS:-llm}" "--hint-dimensions ${HINT_DIMENSIONS:-llm}"; do
  set -- ${_flag}
  if ! python "${VERIFIER_PY:-${REPO_ROOT}/ags_table5_ablation/run_llm_verifier.py}" --help 2>/dev/null \
       | tr -d " \n" | grep -q "$(echo "$1" | sed 's/^--//')"; then continue; fi
  if ! python "${VERIFIER_PY:-${REPO_ROOT}/ags_table5_ablation/run_llm_verifier.py}" "$1" "$2" --help >/dev/null 2>&1; then
    echo "FATAL: ${1} ${2} is not accepted by run_llm_verifier.py. Its choices changed; fix the wrapper." >&2
    exit 2
  fi
done

python -m py_compile ags_table5_ablation/run_llm_verifier.py

# Defaults to the gold-instance test trace. Point it at a fulltagging run's
# bm25_candidates.jsonl to verify the extractor-driven pipeline instead; the two traces share
# the frozen_ags_hypotheses / final_candidates schema this script reads.
TRACE_ARGS=()
if [[ -n "${TEST_TRACE:-}" ]]; then
  TRACE_ARGS=(--test-trace "${TEST_TRACE}")
fi

# WINDOW_TAGS points at a per-arm window file from stage_arm_windows.py. With it the verifier
# judges that arm's OWN fused head instead of the deployed one, which is what lets an ablation
# row be scored by the LLM-only verifier. Without it nothing changes.
WINDOW_TAG_ARGS=()
if [[ -n "${WINDOW_TAGS:-}" ]]; then
  if [[ ! -f "${WINDOW_TAGS}" ]]; then
    echo "WINDOW_TAGS=${WINDOW_TAGS} does not exist; run stage_arm_windows.py first." >&2
    exit 1
  fi
  WINDOW_TAG_ARGS=(--window-tags "${WINDOW_TAGS}")
  echo "WINDOW_TAGS=${WINDOW_TAGS}"
fi

python ags_table5_ablation/run_llm_verifier.py \
  --output-dir "${OUTPUT_DIR}" \
  "${TRACE_ARGS[@]}" \
  --top-m "${TOP_M}" \
  --window-source "${WINDOW_SOURCE}" \
  "${WINDOW_TAG_ARGS[@]}" \
  --query-max-new-tokens "${QUERY_MAX_NEW_TOKENS}" \
  "${HINT_ARGS[@]}" \
  "${JUDGE_ARGS[@]}" \
  --generation-chunk "${GENERATION_CHUNK}" \
  --query-generation-model "${QUERY_GENERATION_MODEL}" \
  --query-generation-backend "${QUERY_GENERATION_BACKEND}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --vllm-batch-size "${VLLM_BATCH_SIZE}" \
  "${RESUME_ARGS[@]}" \
  "${LIMIT_ARGS[@]}"
