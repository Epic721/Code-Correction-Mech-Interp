#!/bin/bash
#SBATCH -J steer_L20
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=131072M
#SBATCH --time=11:00:00
#SBATCH -o slurm_logs/steer_L20_%j.out
#SBATCH -e slurm_logs/steer_L20_%j.err
# NOTE: --mail-user, --mail-type, --gres are NOT specified here.
# they get injected by submit.sh from steer_config.sh, so this file
# is the same on every user's machine -- no per-user edits required.

# -----------------------------------------------------------------
# step 1 of 3 in the steering pipeline (user role / solo chain).
#
# what this does:
#   - closes out caa L25 alpha=8 continuous (the one missing point)
#   - runs the focal L20 sweep:
#       caa,sae,learned x {1,2,4,8} x {surgical,continuous}
#
# expected wall-clock: ~25 min + ~8 hr = ~8.5 hr
#
# DO NOT submit this file directly with `sbatch`. use:
#   ./submit.sh user
# (or ./submit.sh solo) so the per-user config (email, conda env,
# gpu type) gets pulled from steer_config.sh.
#
# resume-safe: if this hits walltime mid-sweep, just re-run the same
# submit command. eval_steering.py picks up where it left off via
# per_problem.jsonl.
# -----------------------------------------------------------------

set -e
cd "$SLURM_SUBMIT_DIR"

# ---- sanity: project dir + config ----
if [ ! -f eval_steering.py ]; then
    echo "ERROR: eval_steering.py not found in $(pwd)"
    echo "submit this from the code_correction_mech_interp dir"
    exit 1
fi

if [ ! -f steer_config.sh ]; then
    echo "ERROR: steer_config.sh not found. submit via ./submit.sh"
    exit 1
fi
source steer_config.sh

mkdir -p logs

# ---- HF token ----
TOKEN_FILE="${STEER_HF_TOKEN_FILE:-$HOME/.hf_token}"
if [ ! -f "$TOKEN_FILE" ]; then
    echo "ERROR: HF token file not found at $TOKEN_FILE"
    exit 1
fi
export HF_TOKEN=$(cat "$TOKEN_FILE")

# ---- environment ----
module load anaconda3/2023.03
module load cuda/11.7.0
module load cudnn/8.5.
conda activate "$STEER_CONDA_ENV"

# fail fast on missing deps before burning gpu hours
python -c "import torch, transformer_lens, sae_lens; print('imports ok')"

echo "================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Started   : $(date)"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Workdir   : $(pwd)"
echo "Conda env : $STEER_CONDA_ENV"
echo "================================="

# -----------------------------------------------------------------
# step 1: close out caa L25 alpha=8 continuous (~25 min)
# -----------------------------------------------------------------
echo ""
echo "[$(date '+%H:%M:%S')] STEP 1: close caa L25 (alpha=8 continuous)"
python eval_steering.py \
    --datasets mbpp \
    --methods caa \
    --layers 25 \
    --alphas 8 \
    --modes continuous \
    --no-plot \
    --log-file logs/steering_caa_L25_close_${SLURM_JOB_ID}.log
echo "[$(date '+%H:%M:%S')] STEP 1 done"

# -----------------------------------------------------------------
# step 2: focal L20 sweep -- the headline experiment (~8 hr)
# alpha=0 baseline reusable from existing caa L25 alpha=0 data
# (alpha=0 is method/layer-agnostic by construction)
# alpha=0.5 dropped: continuous-caa data shows it interpolates trivially
# -----------------------------------------------------------------
echo ""
echo "[$(date '+%H:%M:%S')] STEP 2: focal L20 sweep"
python eval_steering.py \
    --datasets mbpp \
    --methods caa,sae,learned \
    --layers 20 \
    --alphas 1,2,4,8 \
    --modes surgical,continuous \
    --log-file logs/steering_focal_L20_${SLURM_JOB_ID}.log
echo "[$(date '+%H:%M:%S')] STEP 2 done"

echo ""
echo "================================="
echo "Pipeline complete: $(date)"
echo "================================="
