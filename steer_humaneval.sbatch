#!/bin/bash
#SBATCH -J steer_he
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=131072M
#SBATCH --time=03:00:00
#SBATCH -o slurm_logs/steer_he_%j.out
#SBATCH -e slurm_logs/steer_he_%j.err
# --mail-user, --mail-type, --gres are injected by submit.sh.

# -----------------------------------------------------------------
# step 3 of 3 in the steering pipeline.
#
# what this does:
#   - humaneval OOD generalization at L20 only:
#       caa,sae,learned x {1,2,4} x continuous
#
# rationale: 164 problems vs 500 -> noisier signal, so we lean on
# only the most informative (mode, alpha) combos and skip surgical
# here (the report's surgical-vs-continuous claim is established
# from the mbpp L20+L12 runs).
#
# expected wall-clock: ~2 hr
#
# DO NOT submit this file directly. use ./submit.sh user (or solo).
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
echo "Job ID    : $SLURM_JOB_ID  (humaneval OOD)"
echo "Started   : $(date)"
echo "Conda env : $STEER_CONDA_ENV"
echo "================================="

# sanity: humaneval baseline must exist before we can compute correction/corruption
if [ ! -f outputs/humaneval/generations.jsonl ]; then
    echo "ERROR: outputs/humaneval/generations.jsonl missing."
    echo "run 'python eval_sandbox.py --dataset humaneval' first."
    exit 1
fi

echo ""
echo "[$(date '+%H:%M:%S')] humaneval OOD steering"
python eval_steering.py \
    --datasets humaneval \
    --methods caa,sae,learned \
    --layers 20 \
    --alphas 1,2,4 \
    --modes continuous \
    --log-file logs/steering_humaneval_${SLURM_JOB_ID}.log
echo "[$(date '+%H:%M:%S')] humaneval done"

echo ""
echo "================================="
echo "Pipeline complete: $(date)"
echo "================================="
