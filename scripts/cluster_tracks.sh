#!/bin/bash
# Cluster step 2 of 3 (2026-11-24): generate the track pool on the CPU (500 training + 10 eval tracks, 22 m wide all
# the way round, as the laptop's pool). No GPU and no job needed. Then: sbatch cluster_stage_a.slurm.
#
#   bash cluster_tracks.sh
#
# Settings (environment variables, with their defaults):
#   BASE=$HOME/f1       where cluster_setup.sh put the repos
#   WORKERS=$(nproc)    generation processes (lower it on a shared login node, e.g. WORKERS=4)
set -euo pipefail

BASE=${BASE:-$HOME/f1}
SIM=$BASE/f1-simulator
WORKERS=${WORKERS:-$(nproc)}
cd "$SIM"
export JAX_PLATFORMS=cpu  # no GPU here

if [ -f tracks/index.json ]; then
    echo "tracks/index.json exists: keeping the pool (move tracks/ away to generate it again)"
else
    echo "generating the pool in $SIM/tracks with $WORKERS processes"
    time .venv/bin/f1env-tracks --out tracks --train 500 --eval 10 --width 22 --no-wide --workers "$WORKERS"
fi
.venv/bin/python - <<'E'
import json
d = json.load(open("tracks/index.json"))
n_train, n_eval = len(d["train"]), len(d["eval"])
print(f"pool: {n_train} training + {n_eval} eval tracks, width {d.get('width')} m")
assert n_train == 500 and n_eval == 10, "the pool is incomplete"
E
echo
echo "tracks done. Next: cd $SIM && sbatch scripts/cluster_stage_a.slurm"
