import pytest

from f1sim.drivers import rewards
from f1sim.drivers.config import PROFILES, load


@pytest.mark.parametrize("preset", ["laptop", "cluster"])
def test_every_profile_loads(preset):
    for p in PROFILES:
        s = load(p, preset)
        assert s.tqc.obs_dim == 209 and s.tqc.act_dim == 5 and s.tqc.critics * s.tqc.quantiles == 125


def test_profiles_match_the_spec():
    spec = {  # d, H0, gamma, start from, (overtake, place lost, holding, inside line, tyre wear, smoothness)
        "middle": (2, -2.0, 0.995, "", (1, 1, 1, 0, 1, 1)),
        "aggressive": (1, -1.5, 0.985, "middle", (2, 0.5, 0.5, 0, 0.5, 0.5)),
        "smooth": (3, -2.5, 0.997, "middle", (0.5, 1, 0.5, 0, 2, 2)),
        "defensive": (2, -2.0, 0.990, "middle", (0.5, 2, 2, 1, 1, 1)),
    }
    for name, (d, h0, gamma, start, mult) in spec.items():
        s = load(name)
        assert (s.tqc.drop, s.tqc.target_entropy, s.tqc.gamma, s.start_from) == (d, h0, gamma, start)
        assert tuple(s.styles[k] for k in rewards.STYLES) == mult
