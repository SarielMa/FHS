#!/bin/bash
#SBATCH --job-name=fintag_ags_seq
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=logs/%j_fintag_ags_seq_qwen3_b200.txt

set -euo pipefail

# AGS+Seq: the Thompson-Sampling sequential control arm (ags_seq_arms_spec.md).
#
#   round one  = frozen AGS, byte-identical (J=2 -> def/lab render -> K=200 coverage
#                retrieval -> sum-RRF -> range-normalize -> agree rerank, beta=0.6), with a
#                per-fact assertion that round one reproduces the AGS ranking exactly
#   rounds 2-B = symbolic feedback over cluster representatives -> admissible slate (L=6)
#                -> theta_o ~ N(mu_o, nu^2 Sigma_o), argmax psi^T theta_o -> preserve-set
#                enforced revision -> render -> retrieve -> re-consolidate over the union
#
# The novelty gate is OFF (spec section 3), so realized rounds should reach ~4 rather than
# the appendix diagnostic's 2.34. Run with AGS_SEQ_NOVELTY_GATE=1 for the gate-on
# configuration; report which one the table describes.
#
# The uniform-selection control is apply_server_fintagging_ags_seq_random.sh, which must
# run with the same TEST_JSONL and instance order.
#
# Cost: an episode issues 8 LLM calls (2 hypotheses + 3 revisions + 3 replays) and up to 16
# retrievals, against frozen AGS's 2 and 4. RESUME=1 is on, so a job that hits the wall
# clock continues from grounding_traces.jsonl on resubmission -- including the posteriors,
# which are replayed from the stored per-round credits.
#
# Examples:
#   sbatch apply_server_fintagging_ags_seq.sh
#   sbatch --export=ALL,LIMIT=20 apply_server_fintagging_ags_seq.sh
#   sbatch --export=ALL,AGS_SEQ_NOVELTY_GATE=1 apply_server_fintagging_ags_seq.sh

FHS_ROOT="${FHS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REPO_ROOT="${FHS_ROOT}"

export SCRATCH_ROOT="${SCRATCH_ROOT:-${TMPDIR:-/tmp}/fhs-scratch}"
export RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs/runs_fintagging_grounding_baseline}"
export QUERY_MODE="${QUERY_MODE:-ags_seq}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/qwen3_32b_ags_seq}"
export QUERY_GENERATION_MODEL="${QUERY_GENERATION_MODEL:-Qwen/Qwen3-32B}"
export QUERY_GENERATION_BACKEND="${QUERY_GENERATION_BACKEND:-vllm}"
# Full eval: AGS retrieval stage + listwise rerank, scored at the same stage as every
# other query mode. REUSE_CANDIDATES=1 makes a rerun rerank-only.
export RUN_RERANK="${RUN_RERANK:-1}"
export REUSE_CANDIDATES="${REUSE_CANDIDATES:-1}"
export RESUME="${RESUME:-1}"

export AGS_SEQ_MAX_ROUNDS="${AGS_SEQ_MAX_ROUNDS:-4}"
export AGS_SEQ_NOVELTY_GATE="${AGS_SEQ_NOVELTY_GATE:-0}"
export AGS_SEQ_SEED="${AGS_SEQ_SEED:-20260724}"

bash "${REPO_ROOT}/scripts/slurm/apply_server_fintagging_direct_retrieval.sh"
