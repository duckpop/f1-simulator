#!/bin/bash
# Train one stage of the regime (A, B, C or D) as a SLURM job of at most 24 hours, continuing in follow-up jobs until
# the stage has all its steps (2026-11-24).
#
#   cd ~/f1/f1-simulator
#   sbatch scripts/cluster_train.slurm a          # stage A: Middle alone, from scratch
#   sbatch scripts/cluster_train.slurm b          # stage B, from Middle's A best (once A is done)
#   sbatch scripts/cluster_train.slurm c          # stage C, from Middle's B best
#   sbatch scripts/cluster_train.slurm d          # stage D: the personalities, from Middle's C best
#   squeue -u $USER;  tail -f runs/stage-<x>-<job id>.log
#
# A stage with no checkpoint yet starts fresh; one with a latest.npz carries on from it at its step count (--resume).
# 15 minutes before the time limit SLURM sends SIGTERM: the trainer stops after its current chunk and saves
# latest.npz. Each job first queues its own follow-up (it starts once this one ends, and stops at once if the stage
# is done), up to MAX_JOBS jobs; cancel the stage with scancel. Stage steps: STEPS (A: 4M, B-D: 5M by default).
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
PRESET=${PRESET:-laptop}   # laptop: the settings of the laptop runs; cluster: the older cluster preset
if [ "$STAGE" = a ]; then STEPS=${STEPS:-4000000}; else STEPS=${STEPS:-5000000}; fi
MAX_JOBS=${MAX_JOBS:-10}
JOB_N=${JOB_N:-1}
case "$STAGE" in a|b|c) WHO=middle ;; d) WHO=smooth ;; esac  # the profile whose latest.npz tells where the stage is
LATEST=runs/$WHO/stage-$STAGE/latest.npz

done_steps() {  # the stage's steps so far (0: not started)
    [ -f "$LATEST" ] || { echo 0; return; }
    .venv/bin/python -c "import json, numpy as np; print(json.loads(str(np.load('$LATEST')['meta'])).get('env_steps', 0))"
}
DONE=$(done_steps)
echo "stage ${STAGE^^}: $DONE of $STEPS steps done (job $JOB_N of at most $MAX_JOBS)"
if [ "$DONE" -ge "$STEPS" ]; then echo "stage ${STAGE^^} is done"; exit 0; fi

# the follow-up job, in case this one runs out of time (it stops at once if the stage is done by then)
if [ -n "${SLURM_JOB_ID:-}" ] && [ "$JOB_N" -lt "$MAX_JOBS" ]; then
    NEXT=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" --export=ALL,JOB_N=$((JOB_N + 1)),STEPS="$STEPS",PRESET="$PRESET",MAX_JOBS="$MAX_JOBS" scripts/cluster_train.slurm "$STAGE")
    echo "follow-up job $NEXT queued"
fi

.venv/bin/python - <<'E'
import jax
devs = jax.devices()
print("JAX devices:", devs, flush=True)
assert any(d.platform == "gpu" for d in devs), "JAX sees no GPU: check the job got one (--gres) and the CUDA install (jax[cuda12])"
E

RESUME=""
[ "$DONE" -gt 0 ] && RESUME="--resume"
LOG=runs/stage-$STAGE-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M)}.log
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
echo "stage ${STAGE^^}: preset $PRESET, $STEPS steps in all${RESUME:+ (resuming)}; log $LOG"
if [ "$STAGE" = a ]; then
    exec .venv/bin/f1sim-train --tracks tracks --preset "$PRESET" --steps "$STEPS" --patience 0 $RESUME > "$LOG" 2>&1
else
    exec .venv/bin/f1sim-train-b --stage "$STAGE" --tracks tracks --preset "$PRESET" --steps "$STEPS" $RESUME > "$LOG" 2>&1
fi
