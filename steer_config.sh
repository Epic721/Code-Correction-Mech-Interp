#!/bin/bash
# steer_config.sh - single source of truth for your slurm/conda config.
# edit the values below ONCE, then everything else (sbatch files,
# submit.sh, setup_check.sh) reads from here.
#
# this file is sourced (not executed) so don't put any side-effecting
# commands in here -- just exports.

# -------------------------------------------------------------------
# REQUIRED: slurm email notifications
# -------------------------------------------------------------------
export STEER_EMAIL="your_email@gatech.edu"

# -------------------------------------------------------------------
# REQUIRED: name of the conda env w/ torch + transformer_lens + sae_lens
# (run `conda env list` to find yours)
# -------------------------------------------------------------------
export STEER_CONDA_ENV="hw3_7643"

# -------------------------------------------------------------------
# OPTIONAL: path to your hf token file. the file should contain just
# the token string (no newline tricks). create it with:
#     echo "hf_YOUR_TOKEN" > ~/.hf_token && chmod 600 ~/.hf_token
# -------------------------------------------------------------------
export STEER_HF_TOKEN_FILE="${STEER_HF_TOKEN_FILE:-$HOME/.hf_token}"

# -------------------------------------------------------------------
# OPTIONAL: when slurm should email. valid: BEGIN,END,FAIL,ALL,NONE
# -------------------------------------------------------------------
export STEER_MAIL_TYPE="${STEER_MAIL_TYPE:-BEGIN,END,FAIL}"

# -------------------------------------------------------------------
# OPTIONAL: gpu type for --gres. h100 is the default; override if
# your cluster uses a different label (eg "a100", "v100", "h200")
# -------------------------------------------------------------------
export STEER_GPU="${STEER_GPU:-h100}"
