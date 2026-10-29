import os
import time

import jax
import pytest

from f1sim import tqc
from f1sim.drivers import env as de
from f1sim.drivers import watch
from f1sim.nets import EncoderSpec


def test_newest_best(tmp_path):
    a = tmp_path / "middle" / "stage-a"
    a.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        watch.newest_best(tmp_path, "middle")
    (a / "latest.npz").touch()
    assert watch.newest_best(tmp_path, "middle").name == "latest.npz"
    (a / "best-26.npz").touch()
    time.sleep(0.01)
    (a / "best-20.npz").touch()
    os.utime(a / "best-26.npz", (1, 1))  # older
    assert watch.newest_best(tmp_path, "middle").name == "best-20.npz"  # the curriculum's current width
    (a / "best.npz").touch()
    assert watch.newest_best(tmp_path, "middle").name == "best.npz"
    (tmp_path / "middle" / "stage-b").mkdir()
    (tmp_path / "middle" / "stage-b" / "best.npz").touch()
    assert watch.newest_best(tmp_path, "middle").parent.name == "stage-b"


def test_record(tracks, tmp_path):
    from f1env.viewer import write_html

    cfg = tqc.TQCConfig(obs_dim=de.OBS, act_dim=de.ACTIONS, encoder=EncoderSpec(lidar_beams=50, conv=((4, 5, 5),), other_hidden=(8,)), actor_hidden=(8,), critic_hidden=(8,))
    actor = tqc.init(jax.random.PRNGKey(0), cfg).actor
    rec, race = watch.record(tracks[0], actor, cfg, cars=3, laps=1, seconds=12.0)
    assert race.clock[0] >= 12.0 and len(rec.frames) > 40  # lights, then driving, a frame every 4 steps
    write_html(tmp_path / "race.html", rec.replay())
    assert (tmp_path / "race.html").stat().st_size > 10_000


def test_live_window_offscreen(tracks):
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no window on the screen
    cfg = tqc.TQCConfig(obs_dim=de.OBS, act_dim=de.ACTIONS, encoder=EncoderSpec(lidar_beams=50, conv=((4, 5, 5),), other_hidden=(8,)), actor_hidden=(8,), critic_hidden=(8,))
    actor = tqc.init(jax.random.PRNGKey(0), cfg).actor
    rec, race = watch.live(tracks[0], actor, cfg, cars=3, laps=1, seconds=8.0, speed=8.0, save=True, close_when_done=True)
    assert race.clock[0] >= 8.0 and len(rec.frames) > 30
