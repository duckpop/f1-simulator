#!/bin/bash
# The checker as a SLURM job (2026-11-24): realistic speeds of stages A-D and the evals on this GPU, and how many
# 24-hour jobs each stage needs (scripts/benchmark.py, ~15 minutes). Results: runs/benchmark-<host>-<time>.json/.txt.
#
#   cd ~/f1/f1-simulator && sbatch scripts/cluster_check.slurm
#
# Set for the openlab cluster (2026-11-25): partition openlab-queue (1 day max, nodes with 4 GPUs), account
# openlab-korisnici, one GPU of any type (the nodes list theirs as gpu:standard).
#SBATCH --job-name=f1-check
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm-%j.out
#SBATCH --partition=openlab-queue
#SBATCH --account=openlab-korisnici
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p runs
.venv/bin/python - <<'E'
import jax
devs = jax.devices()
print("JAX devices:", devs, flush=True)
assert any(d.platform == "gpu" for d in devs), "JAX sees no GPU: check the job got one (--gres) and the CUDA install (jax[cuda12])"
E
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec .venv/bin/python scripts/benchmark.py "$@"
