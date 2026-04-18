#!/bin/bash
#SBATCH --job-name=vllm_deeptest
#SBATCH --output=exp_logs/exp_%A_%a.out
#SBATCH --error=exp_logs/exp_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=196G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=pi_tpoggio
#SBATCH --array=0-1

# Array search: 4 models × 3 tool modes = 12 tasks.
# Submit:         sbatch run_exp.sh
# Subset:         sbatch --array=3,6,9 run_exp.sh
# Pass-through:   sbatch run_exp.sh --max-records 500 --shuffle
# Single dataset: sbatch run_exp.sh --input experiments_llm_diagnostic.json

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p exp_logs

MODELS=(
    "Qwen/Qwen3-4B-Thinking-2507"
    "Qwen/Qwen3-4B-Instruct-2507"
)
gs_small=(0 1 3 7 15 31)      # use experiments_llm_adversarial.json
gs_large=(63 127)             # use experiments_llm_large_adversarial.json
# "--no-tools"
# "--tools"

TOOL_MODES=(
    "--tools --tool-revise --max-revisions 10"
)

N_MODES=${#TOOL_MODES[@]}              # 3
model_idx=$(( SLURM_ARRAY_TASK_ID / N_MODES ))
mode_idx=$((  SLURM_ARRAY_TASK_ID % N_MODES ))
MODEL="${MODELS[$model_idx]}"
MODE="${TOOL_MODES[$mode_idx]}"

# 2× A100-80GB.
# 30B-A3B: TP=2 (1 replica over both GPUs) — KV cache room for long contexts.
# 4B:      TP=1 (2 data-parallel replicas, one per GPU) — 2× throughput.
if [[ "$MODEL" == *"30B"* ]]; then
    TP=2
else
    TP=1
fi

echo "========================================================"
echo "array_task = $SLURM_ARRAY_TASK_ID / ${SLURM_ARRAY_TASK_MAX:-?}"
echo "model      = $MODEL"
echo "mode       = $MODE"
echo "tp/replica = $TP (replicas auto = visible_gpus // TP)"
echo "cpus       = ${SLURM_CPUS_PER_TASK:-?} (sandbox pool auto-sizes)"
echo "========================================================"

source .venv/bin/activate

G_CSV_SMALL=$(IFS=, ; echo "${gs_small[*]}")
G_CSV_LARGE=$(IFS=, ; echo "${gs_large[*]}")

uv run vllm_deeptest.py \
    --model "$MODEL" \
    $MODE \
    --tensor-parallel-size "$TP" \
    --yes \
    --shuffle \
    --input experiments_llm_adversarial.json \
    --g-values "$G_CSV_SMALL" \
    --server-log "exp_logs/server_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_small.log" \
    "$@"

uv run vllm_deeptest.py \
    --model "$MODEL" \
    $MODE \
    --tensor-parallel-size "$TP" \
    --yes \
    --shuffle \
    --input experiments_llm_large_adversarial.json \
    --g-values "$G_CSV_LARGE" \
    --server-log "exp_logs/server_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_large.log" \
    "$@"
