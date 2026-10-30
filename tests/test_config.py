import pytest

from f1sim.drivers import rewards
from f1sim.drivers.config import CONFIGS, PROFILES, load


@pytest.mark.parametrize("preset", ["laptop", "cluster"])
def test_every_profile_loads(preset):
    for p in PROFILES:
        s = load(p, preset)
        assert s.tqc.obs_dim == 209 and s.tqc.act_dim == 5 and s.tqc.critics * s.tqc.quantiles == 125


def test_profiles_match_the_spec():
    spec = {  # d, H0, gamma, start from, (overtake, place lost, holding, inside line, tyre wear, smoothness)
        "middle": (2, -2.0, 0.994, "", (1, 1, 1, 0, 1, 1)),
        "aggressive": (1, -1.5, 0.990, "middle", (2, 0.6, 0.5, 0, 0.5, 0.5)),
        "smooth": (3, -2.5, 0.996, "middle", (0.5, 1, 0.5, 0, 2, 1.5)),
        "defensive": (2, -2.0, 0.992, "middle", (0.5, 1.8, 2, 1, 1, 1)),
    }
    for name, (d, h0, gamma, start, mult) in spec.items():
        s = load(name)
        assert (s.tqc.drop, s.tqc.target_entropy, s.tqc.gamma, s.start_from) == (d, h0, gamma, start)
        assert tuple(s.styles[k] for k in rewards.STYLES) == mult


def test_a_profile_can_replace_base_values():
    crash = rewards.TERMS.index("crash")
    assert {p: float(load(p).weights[crash]) for p in PROFILES} == {"middle": -70.0, "aggressive": -40.0, "smooth": -200.0, "defensive": -120.0}


def test_unknown_reward_terms_are_refused(tmp_path):
    (tmp_path / "profiles").mkdir()
    (tmp_path / "drivers.toml").write_text((CONFIGS / "drivers.toml").read_text())
    (tmp_path / "profiles" / "smooth.toml").write_text((CONFIGS / "profiles" / "smooth.toml").read_text() + "crashes = -1.0\n")
    with pytest.raises(ValueError, match="unknown reward terms"):
        load("smooth", configs=tmp_path)
