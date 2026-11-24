#!/bin/bash
# Train one stage of the regime (A, B, C or D) as one SLURM job (2026-11-24). It runs until the time limit or until
# the stage has all its steps, whichever comes first; then you decide whether to start it again, and from what.
#
#   cd ~/f1/f1-simulator
#   sbatch scripts/cluster_train.slurm a                       # stage A from scratch
#   sbatch --export=ALL,CONTINUE=1 scripts/cluster_train.slurm a       # carry stage A on from its latest.npz
#   sbatch --export=ALL,FROM=runs/middle/stage-a/best.npz scripts/cluster_train.slurm b   # stage B from that checkpoint
#   sbatch scripts/cluster_train.slurm b                       # stage B from Middle's A best (the default start)
#   squeue -u $USER;  tail -f runs/stage-<x>-<job id>.log
#
# Input variables (with sbatch --export=ALL,NAME=value,...):
#   CONTINUE=1   carry the stage on from its own latest.npz, at its step count (after a time limit or a stop)
#   FROM=<path>  start the stage from this checkpoint (stage D: every personality from it); its steps count from 0
#   STEPS=<n>    the stage's environment steps in all, continuations included (A: 4M, B-D: 5M by default)
#   PRESET=laptop  the config preset (laptop: the laptop runs' settings; cluster: the older cluster preset)
# With neither CONTINUE nor FROM a stage starts fresh: A from scratch, B-D from Middle's best of the stage before.
# A stage that already has checkpoints is not started fresh over them: give CONTINUE=1, or move them away first.
# 15 minutes before the time limit SLURM sends SIGTERM: the trainer stops after its current chunk and saves
# latest.npz, so CONTINUE=1 picks up from there.
#
# Adjust the #SBATCH lines to your cluster (the partition or account, the GPU type).
#SBATCH --job-name=f1-train
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@900
#SBATCH --output=runs/slurm-%j.out
# #SBATCH --partition=<your GPU partition>
# #SBATCH --account=<your account>
set -euo pipefail

STAGE=${1:-${STAGE:-}}
case "$STAGE" in a|b|c|d) ;; *) echo "give the stage: a, b, c or d"; exit 1 ;; esac
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p runs
PRESET=${PRESET:-laptop}
CONTINUE=${CONTINUE:-0}
FROM=${FROM:-}
if [ "$STAGE" = a ]; then STEPS=${STEPS:-4000000}; else STEPS=${STEPS:-5000000}; fi
case "$STAGE" in a|b|c) WHO=middle ;; d) WHO=smooth ;; esac  # the profile whose latest.npz tells where the stage is
LATEST=runs/$WHO/stage-$STAGE/latest.npz

steps_in() {  # a checkpoint's environment steps
    .venv/bin/python -c "import json, numpy as np; print(json.loads(str(np.load('$1')['meta'])).get('env_steps', 0))"
}

ARGS=""
if [ "$CONTINUE" = 1 ] && [ -n "$FROM" ]; then
    echo "give CONTINUE=1 (from the stage's latest.npz) or FROM=<checkpoint>, not both"; exit 1
elif [ "$CONTINUE" = 1 ]; then
    [ -f "$LATEST" ] || { echo "CONTINUE=1, but $LATEST doesn't exist: nothing to carry on"; exit 1; }
    DONE=$(steps_in "$LATEST")
    echo "stage ${STAGE^^}: continuing from $LATEST at $DONE of $STEPS steps"
    [ "$DONE" -lt "$STEPS" ] || { echo "stage ${STAGE^^} already has its $STEPS steps (raise STEPS to train it further)"; exit 0; }
    ARGS="--resume"
else
    if [ -e "runs/$WHO/stage-$STAGE/latest.npz" ] || [ -e "runs/$WHO/stage-$STAGE/best.npz" ]; then
        echo "runs/$WHO/stage-$STAGE already holds a run: CONTINUE=1 to carry it on, or move it away to start again"; exit 1
    fi
    if [ -n "$FROM" ]; then
        [ -f "$FROM" ] || { echo "FROM=$FROM doesn't exist"; exit 1; }
        echo "stage ${STAGE^^}: starting from $FROM ($(steps_in "$FROM") steps in its own stage)"
        if [ "$STAGE" = a ]; then ARGS="--init $FROM"; else ARGS="--init-all $FROM"; fi
    else
        echo "stage ${STAGE^^}: starting fresh"
    fi
fi

.venv/bin/python - <<'E'
import jax
devs = jax.devices()
print("JAX devices:", devs, flush=True)
assert any(d.platform == "gpu" for d in devs), "JAX sees no GPU: check the job got one (--gres) and the CUDA install (jax[cuda12])"
E

LOG=runs/stage-$STAGE-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M)}.log
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
echo "stage ${STAGE^^}: preset $PRESET, $STEPS steps in all; log $LOG"
if [ "$STAGE" = a ]; then
    exec .venv/bin/f1sim-train --tracks tracks --preset "$PRESET" --steps "$STEPS" --patience 0 $ARGS > "$LOG" 2>&1
else
    exec .venv/bin/f1sim-train-b --stage "$STAGE" --tracks tracks --preset "$PRESET" --steps "$STEPS" $ARGS > "$LOG" 2>&1
fi
