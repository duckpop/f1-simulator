import jax
import jax.numpy as jnp
import numpy as np
import pytest

from f1sim import tqc
from f1sim.drivers import env as de
from f1sim.drivers import season
from f1sim.nets import EncoderSpec


def attacks(steps):
    """steps: (under attack, places lost) for one car; (spells, spells lost)."""
    a = season.Attacks(jnp.zeros(1, bool), jnp.zeros(1), jnp.zeros(1))
    for att, lost in steps:
        a = season.track_attacks(a, jnp.array([att]), jnp.array([lost]))
    return int(a.count[0]), int(a.lost[0])


def test_attack_spells():
    assert attacks([(1, 0)] * 30 + [(0, 0)] * 5) == (1, 0)  # held
    assert attacks([(1, 0)] * 10 + [(1, 1)] + [(0, 0)] * 5) == (1, 1)  # passed
    assert attacks([(1, 0)] * 10 + [(1, 1)] + [(1, 0)] * 10 + [(0, 0)]) == (2, 1)  # the next car's attack is a new one
    assert attacks([(1, 0), (0, 0), (1, 0), (0, 0)]) == (2, 0)
    assert attacks([(0, 1)]) == (0, 0)  # not under attack: not an attack lost


def test_seats():
    assert np.bincount(season.seats_for()).tolist() == [6, 6, 5, 5]


def fake(**over):
    r = {
        "points_per_race": 5.0, "finish_rate": 0.6, "out_lap1": 0.1, "disqualified": 0.0, "crashes_per_lap": 0.01, "laps": 100.0, "race_speed_kmh": 200.0,
        "overtakes_per_race": 1.0, "places_lost_per_race": 1.0, "attacks_per_race": 5.0, "attacks_held": 0.6, "km_per_tyre_set": 90.0, "slip_s_per_race": 3.0,
        "slip_share": 0.01, "mean_grid_slot": 11.0, "titles": 0,
    }
    return {**r, **over}


def test_criteria_and_report():
    alone = {p: {"finish_rate": 0.6, "crashes_per_lap": 0.01, "laps": 100.0, "lap_ratio": 1.3, "laps_per_stint": 20.0} for p in ("middle", "aggressive", "smooth", "defensive")}
    alone["smooth"]["crashes_per_lap"] = 0.02  # worse than v2's 1.39%
    profiles = {
        "middle": fake(points_per_race=9.8),
        "aggressive": fake(overtakes_per_race=3.0, points_per_race=10.0),
        "smooth": fake(km_per_tyre_set=120.0, race_speed_kmh=199.0),
        "defensive": fake(attacks_held=0.9),
    }
    sea = {"seasons": 1, "rounds": 24, "profiles": profiles, "out_lap1_field": 0.1, "champions": ["aggressive"]}
    got = {what: ok for what, ok, _ in season.criteria(alone, sea)}
    assert got["middle beats v2 Middle alone"] and not got["smooth beats v2 Middle alone"]
    assert got["fewer cars out on lap 1 than v2 (28%)"]
    assert got["Aggressive makes the most overtakes"] and got["Defensive holds the most attacks"]
    assert got["Smooth's tyres last longest at a similar pace"]
    assert got["Middle scores the most points (or within the noise)"]  # 9.8 vs 10.0
    slow = {**profiles, "smooth": fake(km_per_tyre_set=120.0, race_speed_kmh=180.0)}  # long tyre life by going slowly
    assert not {w: ok for w, ok, _ in season.criteria(alone, {**sea, "profiles": slow})}["Smooth's tyres last longest at a similar pace"]
    text = season.report(alone, sea, {p: f"runs/{p}" for p in profiles}, {"rounds": 24})
    assert "| ✗ | smooth beats v2 Middle alone |" in text and "| v2 middle | | 36% | 28% |" in text and "## Style" in text
    assert [ok for _, ok, _ in season.criteria(None, None)] == [None] * 5


@pytest.mark.slow
def test_one_short_weekend(tracks):
    from f1env.env import F1Env
    from f1env.weekend import WeekendConfig

    cfg = tqc.TQCConfig(obs_dim=de.OBS, act_dim=de.ACTIONS, encoder=EncoderSpec(lidar_beams=50, conv=((4, 5, 5),), other_hidden=(8,)), actor_hidden=(8,), critic_hidden=(8,))
    actors = [tqc.init(jax.random.PRNGKey(p), cfg).actor for p in range(4)]
    env = F1Env(tracks, WeekendConfig(race_km=2.0, quali_sessions=((120.0, 6), (120.0, 6), (120.0, 0))))
    res = season.season(tracks, actors, cfg, rounds=1, env=env, log=None)
    assert res["rounds"] == 1 and len(res["champions"]) == 1
    assert sum(r["mean_grid_slot"] * n for r, n in zip(res["profiles"].values(), (6, 6, 5, 5))) == pytest.approx(sum(range(1, 23)))
    for r in res["profiles"].values():
        assert 0 <= r["finish_rate"] <= 1 and 0 <= r["out_lap1"] <= 1


def test_a_tie_is_not_a_lead_and_no_laps_reads_plainly():
    tied = {p: fake(attacks_held=1.0) for p in ("middle", "aggressive", "smooth", "defensive")}
    sea = {"seasons": 1, "rounds": 1, "profiles": tied, "out_lap1_field": 1.0, "champions": ["middle"]}
    assert not {w: ok for w, ok, _ in season.criteria(None, sea)}["Defensive holds the most attacks"]
    alone = {"middle": {"finish_rate": 0.0, "crashes_per_lap": 10.0, "laps": 0.0, "lap_ratio": 0.0, "laps_per_stint": 0.0}}
    text = season.report(alone, None, {}, {})
    assert "no lap done" in text and "1000" not in text
