#!/bin/bash
# Usage: sbatch submit_extract.sh [output_dir] [limit]
# Example:
#   sbatch submit_extract.sh /scratch/izar/banuls/how2sign_landmarks
#   sbatch submit_extract.sh /scratch/izar/banuls/how2sign_landmarks 20   # smoke-test

#SBATCH --job-name=extract_landmarks
#SBATCH --time=12:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1          # required by cs-503 QOS policy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ── Args ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR=${1:-/scratch/izar/banuls/how2sign_landmarks}
LIMIT=${2:-""}

mkdir -p logs

echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Output dir   : $OUTPUT_DIR"
echo "Limit        : ${LIMIT:-none (full dataset)}"
echo "Start        : $(date)"
echo "=========================================="

# ── Environment ───────────────────────────────────────────────────────────────
# Source conda directly — 'conda init' only writes to ~/.bashrc (interactive
# shells) and has no effect in batch scripts.
source /home/banuls/anaconda3/etc/profile.d/conda.sh
conda activate hand2string

# Hide GPU from MediaPipe: the GPU is allocated above to satisfy the QOS
# policy, but MediaPipe's EGL backend segfaults in headless GPU environments.
# CPU inference is fast enough; real speedup comes from parallel workers below.
export CUDA_VISIBLE_DEVICES=""

# ── Extraction ────────────────────────────────────────────────────────────────
LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

python scripts/extract_landmarks.py \
    --output "$OUTPUT_DIR" \
    --workers 8 \
    $LIMIT_ARG

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
