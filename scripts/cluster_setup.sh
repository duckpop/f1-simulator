#!/bin/bash
# Cluster step 1 of 3 (2026-11-24): clone or pull f1-simulator and the pinned f1env, a virtual environment with JAX
# for CUDA, and a few tests on the CPU. Runs on the login node, no GPU needed. Then: cluster_tracks.sh, then
# sbatch cluster_stage_a.slurm.
#
#   bash cluster_setup.sh
#
# Settings (environment variables, with their defaults):
#   BASE=$HOME/f1                                  where both repos go
#   F1ENV_URL=git@github.com:duckpop/f1env.git     f1env is private: an ssh key on GitHub, or
#                                                  https://<user>:<token>@github.com/duckpop/f1env.git with a token
set -euo pipefail

BASE=${BASE:-$HOME/f1}
SIM=$BASE/f1-simulator
ENVDIR=$BASE/f1env
SIM_URL=https://github.com/duckpop/f1-simulator.git
SIM_BRANCH=v3-safety-first
F1ENV_URL=${F1ENV_URL:-git@github.com:duckpop/f1env.git}

mkdir -p "$BASE"
if [ -d "$SIM/.git" ]; then
    git -C "$SIM" fetch origin
    git -C "$SIM" checkout "$SIM_BRANCH"
    git -C "$SIM" pull --ff-only origin "$SIM_BRANCH"
else
    git clone --branch "$SIM_BRANCH" "$SIM_URL" "$SIM"
fi
# the f1env commit f1sim is pinned to (pyproject.toml), cloned beside it and installed from there
PIN=$(grep -o 'f1env.git@[0-9a-f]\{40\}' "$SIM/pyproject.toml" | cut -d@ -f2)
if [ -d "$ENVDIR/.git" ]; then git -C "$ENVDIR" fetch origin; else git clone "$F1ENV_URL" "$ENVDIR"; fi
git -C "$ENVDIR" checkout -q --detach "$PIN"
echo "f1-simulator $(git -C "$SIM" rev-parse --short HEAD) ($SIM_BRANCH); f1env $(git -C "$ENVDIR" rev-parse --short HEAD) (the pinned commit)"

cd "$SIM"
mkdir -p runs  # the training job writes its SLURM output there (it must exist before the job starts)
if command -v uv >/dev/null; then
    [ -d .venv ] || uv venv --python ">=3.11"
    uv pip install --python .venv/bin/python -e "$ENVDIR"
    uv pip install --python .venv/bin/python --no-deps -e .   # f1env is already in: not fetched again over ssh
    uv pip install --python .venv/bin/python "jax[cuda12]>=0.4.35" "optax>=0.2.3" "numpy>=1.26" "pytest>=8"
else
    [ -d .venv ] || python3 -m venv .venv
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -e "$ENVDIR"
    .venv/bin/pip install -q --no-deps -e .
    .venv/bin/pip install -q "jax[cuda12]>=0.4.35" "optax>=0.2.3" "numpy>=1.26" "pytest>=8"
fi
.venv/bin/python -c "import f1sim, f1env, jax; print('installed: f1sim', f1sim.__file__, '| f1env', f1env.__file__, '| jax', jax.__version__)"

# a few tests on the CPU (the GPU is checked at the start of the training job)
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q tests/test_config.py tests/test_tqc.py tests/test_replay.py tests/test_rewards.py
echo
echo "setup done in $SIM. Next: bash $SIM/scripts/cluster_tracks.sh"
