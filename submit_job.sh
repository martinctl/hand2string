#!/bin/bash
# Usage: sbatch submit_job.sh <config_file> <wandb_key> <num_gpus>
# Example:
#   sbatch submit_job.sh configs/transformer.yaml YOUR_WANDB_KEY 2

#SBATCH --job-name=hand2string
#SBATCH --time=06:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2                 # adjust via NUM_GPUS arg below
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ── Args ──────────────────────────────────────────────────────────────────────
CONFIG_FILE=${1:-configs/transformer.yaml}
WANDB_KEY=${2:-""}
NUM_GPUS=${3:-2}

mkdir -p logs

echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Config       : $CONFIG_FILE"
echo "Num GPUs     : $NUM_GPUS"
echo "Start        : $(date)"
echo "=========================================="

# ── Environment ───────────────────────────────────────────────────────────────
conda activate hand2string

# Install wandb if not already present (non-interactive)
pip install -q wandb 2>/dev/null || true

if [ -n "$WANDB_KEY" ]; then
    export WANDB_API_KEY=$WANDB_KEY
fi

# ── Training ──────────────────────────────────────────────────────────────────
export OMP_NUM_THREADS=4

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$(( RANDOM % 10000 + 20000 )) \
    scripts/train_improved.py \
    --config $CONFIG_FILE

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
