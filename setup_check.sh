#!/bin/bash
# setup_check.sh - pre-flight validator. run this BEFORE submitting any
# jobs to catch the cheap-to-fix problems (missing config values, wrong
# conda env, missing prereqs) instead of finding them out 7 hours into
# a slurm job.
#
# usage: ./setup_check.sh

set -u
cd "$(dirname "$0")"

problems=()
warns=()

note_fail() { problems+=("$*"); }
note_warn() { warns+=("$*"); }

# -------------------------------------------------------------------
# 1. config file
# -------------------------------------------------------------------
if [ ! -f steer_config.sh ]; then
    echo "FAIL: steer_config.sh not found in $(pwd)"
    exit 1
fi
source steer_config.sh

if [ -z "${STEER_EMAIL:-}" ]; then
    note_fail "STEER_EMAIL is empty in steer_config.sh"
elif [ "$STEER_EMAIL" = "your_email@gatech.edu" ]; then
    note_fail "STEER_EMAIL is still the default placeholder. edit steer_config.sh."
fi

if [ -z "${STEER_CONDA_ENV:-}" ]; then
    note_fail "STEER_CONDA_ENV is empty in steer_config.sh"
fi

# -------------------------------------------------------------------
# 2. hf token
# -------------------------------------------------------------------
TOKEN_FILE="${STEER_HF_TOKEN_FILE:-$HOME/.hf_token}"
if [ ! -f "$TOKEN_FILE" ]; then
    note_fail "HF token file not found at $TOKEN_FILE
    create it with:  echo 'hf_YOUR_TOKEN' > $TOKEN_FILE && chmod 600 $TOKEN_FILE"
else
    perms=$(stat -c '%a' "$TOKEN_FILE" 2>/dev/null || stat -f '%A' "$TOKEN_FILE" 2>/dev/null || echo "?")
    if [ "$perms" != "600" ] && [ "$perms" != "?" ]; then
        note_warn "$TOKEN_FILE has permissions $perms (recommend 600)"
    fi
    tok_first10=$(head -c 10 "$TOKEN_FILE")
    if [ "${tok_first10:0:3}" != "hf_" ]; then
        note_warn "$TOKEN_FILE doesn't start with 'hf_' -- looks malformed"
    fi
fi

# -------------------------------------------------------------------
# 3. conda env -- this one needs to be sourced into the current shell
# (a subshell `conda activate` doesn't tell us if it worked)
# -------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    if [ -f /etc/profile.d/conda.sh ]; then
        source /etc/profile.d/conda.sh
    fi
    if ! command -v conda >/dev/null 2>&1; then
        note_warn "conda not on PATH. you may need 'module load anaconda3' before submit"
    fi
fi

if command -v conda >/dev/null 2>&1; then
    if ! conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$STEER_CONDA_ENV"; then
        note_fail "conda env '$STEER_CONDA_ENV' not found.
    available envs: $(conda env list 2>/dev/null | awk 'NR>2{print $1}' | tr '\n' ' ')"
    fi
fi

# -------------------------------------------------------------------
# 4. python deps -- only check if conda is reachable AND env exists
# -------------------------------------------------------------------
if command -v conda >/dev/null 2>&1 && \
   conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$STEER_CONDA_ENV"; then
    deps_check=$(conda run -n "$STEER_CONDA_ENV" python -c \
        "import torch, transformer_lens, sae_lens; print('ok')" 2>&1 || echo "FAILED")
    if [ "$deps_check" != "ok" ]; then
        note_fail "python deps missing in env '$STEER_CONDA_ENV':
$deps_check"
    fi
fi

# -------------------------------------------------------------------
# 5. data prereqs (vectors + baselines)
# -------------------------------------------------------------------
[ -d outputs/vectors ] || note_fail "outputs/vectors/ missing -- run vector_extraction.py first"
[ -f outputs/mbpp/generations.jsonl ] || note_fail "outputs/mbpp/generations.jsonl missing -- run eval_sandbox.py --dataset mbpp first"
[ -f outputs/humaneval/generations.jsonl ] || note_warn "outputs/humaneval/generations.jsonl missing -- humaneval sweep will fail"

if [ -d outputs/vectors ]; then
    n_caa=$(ls outputs/vectors/caa_layer_*.pt 2>/dev/null | wc -l | tr -d ' ')
    n_sae=$(ls outputs/vectors/sae_layer_*.pt 2>/dev/null | wc -l | tr -d ' ')
    n_lrn=$(ls outputs/vectors/learned_layer_*.pt 2>/dev/null | wc -l | tr -d ' ')
    [ "$n_caa" = "0" ] && note_fail "no caa vectors found in outputs/vectors/"
    [ "$n_sae" = "0" ] && note_warn "no sae vectors found in outputs/vectors/"
    [ "$n_lrn" = "0" ] && note_warn "no learned vectors found in outputs/vectors/"
fi

# -------------------------------------------------------------------
# 6. sbatch / submit / plot scripts present
# -------------------------------------------------------------------
for f in steer_L25_and_L20.sbatch steer_L12.sbatch steer_humaneval.sbatch \
         submit.sh plot_steering_results.py; do
    [ -f "$f" ] || note_fail "$f missing"
done

# -------------------------------------------------------------------
# 7. log dirs writable
# -------------------------------------------------------------------
mkdir -p slurm_logs logs 2>/dev/null
[ -w slurm_logs ] || note_fail "slurm_logs/ is not writable"
[ -w logs ]       || note_fail "logs/ is not writable"

# -------------------------------------------------------------------
# report
# -------------------------------------------------------------------
echo ""
if [ ${#problems[@]} -gt 0 ]; then
    echo "=== FAILURES ==="
    for p in "${problems[@]}"; do
        echo "  - $p"
    done
fi

if [ ${#warns[@]} -gt 0 ]; then
    echo ""
    echo "=== warnings ==="
    for w in "${warns[@]}"; do
        echo "  - $w"
    done
fi

echo ""
if [ ${#problems[@]} -eq 0 ]; then
    echo "OK -- ready to submit."
    echo ""
    echo "  email      : $STEER_EMAIL"
    echo "  conda env  : $STEER_CONDA_ENV"
    echo "  hf token   : $TOKEN_FILE"
    echo "  gpu type   : $STEER_GPU"
    if [ -d outputs/vectors ]; then
        echo "  vectors    : $(ls outputs/vectors/*.pt 2>/dev/null | wc -l | tr -d ' ') .pt files"
    fi
    if [ -f outputs/mbpp/generations.jsonl ]; then
        echo "  mbpp base  : $(wc -l < outputs/mbpp/generations.jsonl | tr -d ' ') problems"
    fi
    if [ -f outputs/humaneval/generations.jsonl ]; then
        echo "  he base    : $(wc -l < outputs/humaneval/generations.jsonl | tr -d ' ') problems"
    fi
    echo ""
    echo "next:  ./submit.sh user|friend|solo"
    exit 0
else
    echo "NOT READY: fix the failures above, then re-run ./setup_check.sh"
    exit 1
fi
