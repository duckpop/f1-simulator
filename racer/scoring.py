"""
Scoring races: place points (evals), F1 points (head to head), the
profiles' eval bonuses, and averaging over grids, maps and rounds.

A race result is the episode-stats dict from workers.episode_stats.
"""

import numpy as np

F1_POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)


def place_points(cars, position):
    """Every finishing place scores: cars - position + 1 (22 for P1 on 22
    cars, 1 for the last finisher); a crash (position None) scores 0."""
    return 0 if position is None else cars - position + 1


def f1_points(position):
    return F1_POINTS[position - 1] if position is not None and position <= len(F1_POINTS) else 0


def mine(result, key):
    """The trained brain's cars' values of a per-car list."""
    return [v for v, own in zip(result[key], result["learner"]) if own]


def race_points(result, bonus):
    """(points per car of the trained brain in one eval race, what its
    bonuses counted). bonus: the profile's [eval_bonus] - a finisher scores
    place_weight x its place points + f1_weight x its F1 points, per pass on
    an opponent's car, minus per time passed (never below finish_floor); a
    crashed car only crashed_pass_bonus per pass. Plus the race's fastest lap
    by a finishing car of the brain, and the brain's best lap beating the
    opponents' best."""
    n = len(result["positions"])
    own = [i for i, m in enumerate(result["learner"]) if m]
    if not own:
        return 0.0, {"passes_on_opp": 0, "passed_by_opp": 0, "fastest_lap": 0, "faster": 0, "top10": 0}
    finished = [i for i in own if result["positions"][i] is not None]
    top10 = [i for i in finished if result["positions"][i] <= len(F1_POINTS)]
    pts = {i: bonus.crashed_pass_bonus * result["passes_on_opp"][i] for i in own}
    for i in finished:
        pos = result["positions"][i]
        pts[i] = max(bonus.finish_floor, bonus.place_weight * place_points(n, pos) + bonus.f1_weight * f1_points(pos)
                     + bonus.pass_bonus * result["passes_on_opp"][i] - bonus.passed_penalty * result["passed_by_opp"][i])
    laps = result["best_laps"]
    timed = [i for i in range(n) if laps[i]]
    fastest = min(timed, key=lambda i: laps[i]) if timed else None
    if fastest in finished:
        pts[fastest] += bonus.fastest_lap_bonus
    ours = [laps[i] for i in timed if result["learner"][i]]
    theirs = [laps[i] for i in timed if not result["learner"][i]]
    faster = bool(ours and theirs and min(ours) < min(theirs))
    score = float(np.mean(list(pts.values()))) + bonus.pace_bonus * faster
    counted = own if bonus.crashed_pass_bonus else finished
    return score, {"passes_on_opp": sum(result["passes_on_opp"][i] for i in counted),
                   "passed_by_opp": sum(result["passed_by_opp"][i] for i in finished),
                   "fastest_lap": int(fastest in finished), "faster": int(faster), "top10": len(top10)}


def score_round(results, bonus, with_opponents):
    """An eval round: per race the brain's mean distance (not the median:
    once half the grid survives, the median only measures speed) and its
    place points; per map the average over its grids; the round score the
    average over maps. Returns a dict with per-race rows too."""
    races, by_map = [], {}
    for r in results:
        dist = float(np.mean(mine(r, "progress_m")))
        pts, counts = race_points(r, bonus) if with_opponents else (None, None)
        races.append({"result": r, "distance": dist, "points": pts, "counts": counts})
        by_map.setdefault(r["map"], []).append(races[-1])
    distance = {m: float(np.mean([x["distance"] for x in rs])) for m, rs in by_map.items()}
    points = {m: float(np.mean([x["points"] for x in rs])) for m, rs in by_map.items()} if with_opponents else {}
    return {"races": races, "by_map": by_map, "distance": distance, "points": points,
            "round_distance": float(np.mean(list(distance.values()))),
            "round_points": float(np.mean(list(points.values()))) if points else None}


def win_rate(duels_list):
    """(overall share of comparisons the brain's cars finished ahead,
    {opponent: (share, comparisons)}) from per-race {opponent: [wins, n]}."""
    total = {}
    for duels in duels_list:
        for name, (w, k) in duels.items():
            t = total.setdefault(name, [0, 0])
            t[0] += w
            t[1] += k
    n = sum(k for _, k in total.values())
    overall = sum(w for w, _ in total.values()) / n if n else None
    return overall, {name: (w / k, k) for name, (w, k) in total.items() if k}
