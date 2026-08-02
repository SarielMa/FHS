#!/bin/bash
#SBATCH --job-name=codiesp_free_text
#SBATCH --mail-type=ALL
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=logs/%j_codiesp_one_pass_grounding_qwen3_b200.txt

set -euo pipefail

DOMAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QUERY_MODE="${QUERY_MODE:-one_pass_grounding}"
export OUTPUT_DIR="${OUTPUT_DIR:-${DOMAIN_ROOT}/runs_codiesp_grounding_baseline/qwen3_32b_one_pass_grounding}"

bash "${DOMAIN_ROOT}/slurm/apply_server_codiesp_shared.sh"
