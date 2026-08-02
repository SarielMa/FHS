#!/usr/bin/env bash
# Run/submit the oracle plus the extra CodiEsp arms added by the revised
# experiment matrix. Gold oracle is run once; GPU arms are submitted for both
# coverage settings.
#
# Keep the interpretation clean:
# - `parallel_sampling` is an independent comparison baseline, not an ablation.
# - Ablations are only the FHS-family variants, all backed by
#   ags_frozen_grounding.build_frozen_ags_method_record.
set -euo pipefail

DOMAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-${DOMAIN_ROOT}/runs_codiesp_grounding_baseline}"
TEST_JSONL="${TEST_JSONL:-${DOMAIN_ROOT}/data/codiesp/facts_test_full_exact.jsonl}"
STATS_JSON="${STATS_JSON:-${DOMAIN_ROOT}/data/codiesp/stats_full_exact.json}"
RELOCATIONS_JSONL="${RELOCATIONS_JSONL:-${DOMAIN_ROOT}/data/codiesp/evidence_relocations_full_exact.jsonl}"
SPOTCHECK_TSV="${SPOTCHECK_TSV:-${DOMAIN_ROOT}/data/codiesp/spotcheck_50_full_exact.tsv}"
DOCS_TXT="${DOCS_TXT:-${DOMAIN_ROOT}/data/codiesp/test_docs_full_exact.txt}"
EXPECTED_FACTS="${EXPECTED_FACTS:-3144}"

cd "${DOMAIN_ROOT}"
mkdir -p logs "${RUNS_ROOT}"

python scripts/validate_codiesp_full_outputs.py \
  --facts-jsonl "${TEST_JSONL}" \
  --stats-json "${STATS_JSON}" \
  --relocations-jsonl "${RELOCATIONS_JSONL}" \
  --spotcheck-tsv "${SPOTCHECK_TSV}" \
  --docs-txt "${DOCS_TXT}" \
  --expected-facts "${EXPECTED_FACTS}" \
  --min-exact-rate 1.0 \
  --min-parse-rate 1.0

# #2: oracle retrieval diagnostic. CPU-only: no Slurm and no LLM rerank.
MODE=retrieval \
QUERY_MODE=gold_label_definition_retrieval \
TEST_JSONL="${TEST_JSONL}" \
OUTPUT_DIR="${RUNS_ROOT}/qwen3_32b_gold_label_definition_retrieval_full_exact" \
LABEL_COVERAGE_WEIGHT=0.0 \
RUN_RERANK=0 \
REUSE_CANDIDATES=0 \
bash "${DOMAIN_ROOT}/run_codiesp_grounding_baseline.sh"

submit_gpu_arm () {
  local query_mode="$1"
  local output_name="$2"
  local time_limit="$3"
  local label_coverage_weight="$4"
  local suffix="$5"
  shift 5

  sbatch \
    --partition=gpu_b200 \
    --gpus=b200:1 \
    --nodes=1 \
    --cpus-per-task=1 \
    --mem=256G \
    --time="${time_limit}" \
    --job-name="codiesp_${query_mode}_full_${suffix}" \
    --output="${DOMAIN_ROOT}/logs/%j_codiesp_${query_mode}_full_${suffix}.txt" \
    --export="ALL,TEST_JSONL=${TEST_JSONL},QUERY_MODE=${query_mode},OUTPUT_DIR=${RUNS_ROOT}/${output_name},LABEL_COVERAGE_WEIGHT=${label_coverage_weight},RESUME=1,REUSE_CANDIDATES=1,$*" \
    "${DOMAIN_ROOT}/apply_server_codiesp_shared.sh"
}

submit_fhs_ablation () {
  local query_mode="$1"
  case "${query_mode}" in
    fhs_j1|fhs_no_verifier) ;;
    *)
      echo "FHS ablation submitter received non-FHS query_mode=${query_mode}" >&2
      exit 1
      ;;
  esac
  submit_gpu_arm "$@"
}

# #5: independent comparison baseline: two free-text samples, matching FHS J=2
# only in sample count. This is not an FHS ablation.
submit_gpu_arm parallel_sampling qwen3_32b_parallel_sampling_n2_full_wcov0 05:00:00 0.0 wcov0 "RETRIEVAL_ROUNDS=2,RUN_RERANK=1"
submit_gpu_arm parallel_sampling qwen3_32b_parallel_sampling_n2_full_wcov1 05:00:00 1.0 wcov1 "RETRIEVAL_ROUNDS=2,RUN_RERANK=1"

# FHS-family ablations. These share frozen_ags's structured hypothesis,
# retrieval/fusion, and candidate record builder; only the frozen constants
# below differ.

# #7: FHS with J=1, candidate-level verifier retained.
submit_fhs_ablation fhs_j1 qwen3_32b_fhs_j1_full_wcov0 05:00:00 0.0 wcov0 "RUN_RERANK=1"
submit_fhs_ablation fhs_j1 qwen3_32b_fhs_j1_full_wcov1 05:00:00 1.0 wcov1 "RUN_RERANK=1"

# #8: FHS with J=2 and candidate-level verifier disabled.
submit_fhs_ablation fhs_no_verifier qwen3_32b_fhs_no_verifier_full_wcov0 04:00:00 0.0 wcov0 "RUN_RERANK=1"
submit_fhs_ablation fhs_no_verifier qwen3_32b_fhs_no_verifier_full_wcov1 04:00:00 1.0 wcov1 "RUN_RERANK=1"
