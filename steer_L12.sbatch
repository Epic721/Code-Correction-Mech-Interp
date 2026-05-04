#!/bin/bash
#SBATCH -J steer_L12
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=131072M
#SBATCH --time=11:00:00
#SBATCH -o slurm_logs/steer_L12_%j.out
#SBATCH -e slurm_logs/steer_L12_%j.err
# --mail-user, --mail-type, --gres are injected by submit.sh from
# steer_config.sh -- this file stays per-user-agnostic.

# -----------------------------------------------------------------
# step 2 of 3 in the steering pipeline (friend role / solo chain).
#
# what this does:
#   - layer-depth ablation at L12:
#       caa,sae,learned x {1,2,4,8} x {surgical,continuous}
#
# why L12: SAE t-stat is highest there (5.61 vs 5.19 at L20), and the
# learned vector trained best there too (delta_val_loss = -0.547 vs
# -0.428 at L20). this is the strongest second-layer comparison.
#
# expected wall-clock: ~8 hr
#
# DO NOT submit this file directly. use:
#   ./submit.sh friend
# (or ./submit.sh solo) so per-user config gets injected.
# -----------------------------------------------------------------

set -e
cd "$SLURM_SUBMIT_DIR"

if [ ! -f eval_steering.py ]; then
    echo "ERROR: eval_steering.py not found in $(pwd)"
    exit 1
fi

if [ ! -f steer_config.sh ]; then
    echo "ERROR: steer_config.sh not found. submit via ./submit.sh"
    exit 1
fi
source steer_config.sh

mkdir -p logs

TOKEN_FILE="${STEER_HF_TOKEN_FILE:-$HOME/.hf_token}"
if [ ! -f "$TOKEN_FILE" ]; then
    echo "ERROR: HF token file not found at $TOKEN_FILE"
    exit 1
fi
export HF_TOKEN=$(cat "$TOKEN_FILE")

module load anaconda3/2023.03
module load cuda/11.7.0
module load cudnn/8.5.
conda activate "$STEER_CONDA_ENV"

python -c "import torch, transformer_lens, sae_lens; print('imports ok')"

echo "================================="
echo "Job ID    : $SLURM_JOB_ID  (L12 focal sweep)"
echo "Node      : $SLURMD_NODENAME"
echo "Started   : $(date)"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Conda env : $STEER_CONDA_ENV"
echo "================================="

echo ""
echo "[$(date '+%H:%M:%S')] focal L12 sweep"
python eval_steering.py \
    --datasets mbpp \
    --methods caa,sae,learned \
    --layers 12 \
    --alphas 1,2,4,8 \
    --modes surgical,continuous \
    --log-file logs/steering_focal_L12_${SLURM_JOB_ID}.log
echo "[$(date '+%H:%M:%S')] L12 sweep done"

echo ""
echo "================================="
echo "Pipeline complete: $(date)"
echo "================================="
