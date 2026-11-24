#!/bin/bash
# Cluster step 3 of 3 (2026-11-24): stage A from scratch on one GPU, as a SLURM job. It checks the GPU first (and stops
# if JAX can't see one), then trains Middle alone for STEPS environment steps, no early stop.
#
#   cd ~/f1/f1-simulator && sbatch scripts/cluster_stage_a.slurm
#   squeue -u $USER                                  # is it running
#   tail -f runs/stage-a-<job id>.log                # follow it
#
# Adjust the #SBATCH lines to your cluster (the partition or account name, the GPU type, the time limit). Stage A
# keeps runs/middle/stage-a/latest.npz at every eval; after a time-limit stop, carry on with
#   STEPS=<the steps left> INIT=runs/middle/stage-a/latest.npz sbatch scripts/cluster_stage_a.slurm
#
#SBATCH --job-name=f1-stage-a
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-%j.out
# #SBATCH --partition=<your GPU partition>
# #SBATCH --account=<your account>
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p runs
PRESET=${PRESET:-laptop}   # laptop: the settings of the laptop run (64 races, replay 4M); cluster: the older preset
STEPS=${STEPS:-4000000}
INIT=${INIT:-}
LOG=runs/stage-a-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M)}.log

.venv/bin/python - <<'E'
import jax
devs = jax.devices()
print("JAX devices:", devs, flush=True)
assert any(d.platform == "gpu" for d in devs), "JAX sees no GPU: check the job got one (--gres) and the CUDA install (jax[cuda12])"
E
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ -z "$INIT" ] && [ -e runs/middle/stage-a/best.npz ]; then
    echo "runs/middle/stage-a already holds a run: move it away to start from scratch, or set INIT to carry it on"
    exit 1
fi
echo "stage A: preset $PRESET, $STEPS steps${INIT:+, from $INIT}; log $LOG"
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
exec .venv/bin/f1sim-train --tracks tracks --preset "$PRESET" --steps "$STEPS" --patience 0 ${INIT:+--init "$INIT"} > "$LOG" 2>&1
