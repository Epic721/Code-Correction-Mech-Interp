#!/bin/bash
# submit.sh - unified submission wrapper for the steering pipeline.
#
# usage:
#   ./submit.sh user      submit your L25-closeout + L20 + humaneval chain
#                         (paired w/ a friend running L12 in their own dir)
#   ./submit.sh friend    submit L12 only (paired w/ someone running user)
#   ./submit.sh solo      submit L20 -> L12 -> humaneval as one chain (no friend)
#
# all submission-time overrides (mail-user, mail-type, gpu type) come
# from steer_config.sh, so the sbatch files themselves stay portable.

set -e
cd "$(dirname "$0")"

# -------------------------------------------------------------------
# load config
# -------------------------------------------------------------------
if [ ! -f steer_config.sh ]; then
    echo "ERROR: steer_config.sh not found in $(pwd)" >&2
    echo "edit the file first (it lives next to this script)" >&2
    exit 1
fi
source steer_config.sh

if [ -z "${STEER_EMAIL:-}" ] || [ "$STEER_EMAIL" = "your_email@gatech.edu" ]; then
    echo "ERROR: edit STEER_EMAIL in steer_config.sh first." >&2
    exit 1
fi

mkdir -p slurm_logs logs

# common sbatch overrides applied to every submission
SBATCH_COMMON=(
    --mail-user="$STEER_EMAIL"
    --mail-type="$STEER_MAIL_TYPE"
    --gres="gpu:${STEER_GPU}:1"
)

# -------------------------------------------------------------------
# dispatch
# -------------------------------------------------------------------
mode="${1:-}"
case "$mode" in
    user|main)
        JOB1=$(sbatch --parsable "${SBATCH_COMMON[@]}" steer_L25_and_L20.sbatch)
        JOB2=$(sbatch --parsable "${SBATCH_COMMON[@]}" \
            --dependency=afterany:$JOB1 steer_humaneval.sbatch)
        echo "queued (user role):"
        echo "  $JOB1  steer_L25_and_L20.sbatch"
        echo "  $JOB2  steer_humaneval.sbatch  (after any:$JOB1)"
        echo ""
        echo "estimated wall-clock: ~10.5 hr"
        ;;

    friend)
        JOB=$(sbatch --parsable "${SBATCH_COMMON[@]}" steer_L12.sbatch)
        echo "queued (friend role):"
        echo "  $JOB  steer_L12.sbatch"
        echo ""
        echo "estimated wall-clock: ~8 hr"
        ;;

    solo)
        JOB1=$(sbatch --parsable "${SBATCH_COMMON[@]}" steer_L25_and_L20.sbatch)
        JOB2=$(sbatch --parsable "${SBATCH_COMMON[@]}" \
            --dependency=afterany:$JOB1 steer_L12.sbatch)
        JOB3=$(sbatch --parsable "${SBATCH_COMMON[@]}" \
            --dependency=afterany:$JOB2 steer_humaneval.sbatch)
        echo "queued (solo):"
        echo "  $JOB1  steer_L25_and_L20.sbatch"
        echo "  $JOB2  steer_L12.sbatch          (after any:$JOB1)"
        echo "  $JOB3  steer_humaneval.sbatch    (after any:$JOB2)"
        echo ""
        echo "estimated wall-clock: ~18.5 hr"
        ;;

    *)
        cat <<EOF
usage: $0 <role>

  user    submit L25-closeout + L20 + humaneval (parallel w/ a friend on L12)
  friend  submit L12 only (parallel w/ someone running L20 + humaneval)
  solo    submit everything serially as one chain (no friend)

before submitting:
  1. edit steer_config.sh   (email + conda env)
  2. set up your HF token   (echo "hf_..." > ~/.hf_token && chmod 600 ~/.hf_token)
  3. run ./setup_check.sh   (validates 1+2 plus python deps and data prereqs)
  4. ./submit.sh <role>     (this script)

monitoring (once submitted):
  squeue -u \$USER
  tail -f slurm_logs/steer_*_<JOBID>.out

post-run plotting:
  python plot_steering_results.py
EOF
        exit 1
        ;;
esac
