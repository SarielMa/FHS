#!/bin/bash
#SBATCH --job-name=fintag_one_pass_structured
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=logs/%j_fintag_one_pass_structured_qwen3_b200.txt

set -euo pipefail

# Baseline: One-pass grounding (structured) == AGS with J=1.
#
#   ONE greedy structured hypothesis (six dimensions, AGS's own generator prompt and schema)
#   -> the SAME renderer (def+lab for table, def only for text)
#   -> the SAME retrieval (K=200, w_cov=1.0, same index and datatype filter)
#   -> the SAME fusion (sum-RRF, kappa=60) -> range-normalize
#   -> NO consensus rerank (beta=0; one hypothesis has nothing to reach consensus with),
#      so the output ranking IS the normalized fused score.
#
# It isolates the contribution of the structured representation from the ensemble + verifier
# machinery. This is not a reimplementation: query_mode=one_pass_structured runs the identical
# ags_frozen_grounding.build_frozen_ags_method_record that frozen_ags runs, with the J=1
# variant of FrozenAgsConfig. The variant is frozen and asserted at startup exactly like AGS
# (J=1, beta=0, kappa=60, w_cov=1.0, K=200, temperature=0) -- see _FROZEN_VARIANTS.
#
# WHY GREEDY: frozen_ags samples at temperature 0.8 because it needs J independent draws. With
# J=1 there is nothing to decorrelate, so sampling would only add variance to a baseline whose
# job is to be a clean reference point.
#
# --cpus-per-task=8 is set deliberately. It is absent from the older method scripts, which
# therefore get Slurm's default of 1 CPU and starve the GPU through a single-lane driver loop.
#
#   sbatch apply_server_fintagging_one_pass_structured.sh
#   sbatch --export=ALL,LIMIT=25 apply_server_fintagging_one_pass_structured.sh   # smoke test
#
# Report R@10/50/200, MRR and top-1 accuracy, table and text separately -- the metrics.json
# this writes already splits by modality.
#
# NOTE ON THE TABLE 5 CROSS-CHECK. This row is NOT numerically identical to the Table 5
# ablation row as that row is currently computed; see compare_one_pass_structured_vs_table5.py,
# which quantifies the gap instead of asserting an equality that cannot hold. Run it after this
# job finishes.

FHS_ROOT="${FHS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REPO_ROOT="${FHS_ROOT}"

export SCRATCH_ROOT="${SCRATCH_ROOT:-${TMPDIR:-/tmp}/fhs-scratch}"
export RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs/runs_fintagging_grounding_baseline}"
export QUERY_MODE="${QUERY_MODE:-one_pass_structured}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/qwen3_32b_one_pass_structured}"
export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
# Full eval: retrieval stage + listwise rerank, same as every other method, so the row is
# scored at the same stage (bm25_retrieval + qwen_reranked in metrics.json).
export RUN_RERANK="${RUN_RERANK:-1}"
export REUSE_CANDIDATES="${REUSE_CANDIDATES:-1}"
export RESUME="${RESUME:-1}"

bash "${REPO_ROOT}/scripts/slurm/apply_server_fintagging_direct_retrieval.sh"
