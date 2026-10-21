#!/usr/bin/env bash
# Status of a profile's latest training run: eval rounds and training stats
# per 30 minutes, from saved_models/<profile>/evals.csv and metrics.csv.
# Usage: ./status.sh [profile]   (default: defensive)
cd "$(dirname "$0")" || exit 1
P=${1:-defensive}
D=saved_models/$P
R=$(tail -1 "$D/metrics.csv" | cut -d, -f1)
echo "$P · run started $R · now $(date '+%Y-%m-%d %H:%M')"
pgrep -f "python train.py --profile $P" >/dev/null && echo "training: running" || echo "training: NOT running"
nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
free -m | awk '/Mem:/{printf "RAM: %.1f GB available\n", $7 / 1024}'
[ -f "$D/best_meta.json" ] && python3 -c "import json; m = json.load(open('$D/best_meta.json')); print(f\"best eval: {m['score']:.2f} at {m['env_steps'] / 1e6:.1f}M steps of its run\")"

echo
echo "EVALS (per 30 min; score = pts/car; finished/passed per round of all eval races)"
awk -F, -v R="$R" '
NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
index($0, R) == 1 {
    b = int($c["elapsed_s"] / 1800); k = $c["elapsed_s"]; g = $c["grids"]
    if (!((b, k) in seen)) { seen[b, k] = 1; n[b]++; sc[b] += $c["round_points"]; if ($c["round_points"] > mx[b]) mx[b] = $c["round_points"]; nb[b] += $c["new_best"]; st[b] = $c["env_steps"] }
    fin[b] += $c["survived"] * g; pd[b] += $c["passed"] * g; po[b] += $c["passed_by_opp"] * g
    tp[b, $c["map"]] += $c["points_per_car"]; tn[b, $c["map"]]++; maps[$c["map"]] = 1
}
END {
    printf "%-9s %6s %6s %6s %5s %8s %7s %7s", "time", "rounds", "mean", "best", "new", "finished", "passed", "by opp"
    for (m in maps) printf " %6.6s", m; print ""
    for (i = 0; i in n; i++) {
        printf "%3d-%3dm %6d %6.2f %6.2f %5d %8.1f %7.0f %7.0f", i * 30, i * 30 + 30, n[i], sc[i] / n[i], mx[i], nb[i], fin[i] / n[i], pd[i] / n[i], po[i] / n[i]
        for (m in maps) printf " %6.1f", tp[i, m] / tn[i, m]; print ""
    }
}' "$D/evals.csv"

echo
echo "TRAINING (per 30 min; last 200 car-episodes, with exploration noise)"
awk -F, -v R="$R" '
NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
index($0, R) == 1 {
    b = int($c["elapsed_s"] / 1800); n[b]++
    sv[b] += $c["survival_rate"]; cc[b] += $c["car_crash_rate"]; ps[b] += $c["passes_per_car"]
    pd[b] += $c["passed_per_car"]; po[b] += $c["passed_by_opp_per_car"]; sp[b] += $c["steps_per_s"]
    st[b] = $c["env_steps"]; split($c["win_rates"], w, ";"); for (j in w) { split(w[j], kv, "="); wr[b, kv[1]] += kv[2]; names[kv[1]] = 1 }
}
END {
    printf "%-9s %8s %8s %7s %7s %7s %9s", "time", "steps", "survived", "car hit", "passes", "passed", "by opp"
    for (o in names) printf " %12s", "ahead " o; print ""
    for (i = 0; i in n; i++) {
        printf "%3d-%3dm %7.1fM %7.0f%% %6.0f%% %7.2f %7.2f %9.2f", i * 30, i * 30 + 30, st[i] / 1e6, 100 * sv[i] / n[i], 100 * cc[i] / n[i], ps[i] / n[i], pd[i] / n[i], po[i] / n[i]
        for (o in names) printf " %11.0f%%", 100 * wr[i, o] / n[i]; print ""
    }
    printf "speed now: %.0f steps/s\n", sp[i - 1] / n[i - 1]
}' "$D/metrics.csv"
