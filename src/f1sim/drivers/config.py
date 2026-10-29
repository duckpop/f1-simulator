"""Reads configs/drivers.toml and configs/profiles/<profile>.toml into the brain's, the races' and the rewards' settings."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..nets import EncoderSpec
from ..tqc import TQCConfig
from . import env as de
from . import rewards

CONFIGS = Path(__file__).resolve().parents[3] / "configs"
PROFILES = ("middle", "aggressive", "smooth", "defensive")


@dataclass
class Settings:
    profile: str
    description: str
    tqc: TQCConfig
    env: de.EnvConfig
    weights: np.ndarray  # (terms,) reward weight per term
    styles: dict  # the profile's multiplier per style term
    start_from: str  # the profile whose stage A networks this one starts from ("": from scratch)
    stage_a: dict
    preset: dict


def load(profile: str, preset: str = "laptop", configs=CONFIGS) -> Settings:
    configs = Path(configs)
    shared = tomllib.loads((configs / "drivers.toml").read_text())
    own = tomllib.loads((configs / "profiles" / f"{profile}.toml").read_text())
    if preset not in shared["preset"]:
        raise ValueError(f"no preset {preset!r} in drivers.toml (have: {', '.join(shared['preset'])})")
    p = shared["preset"][preset]
    t, e = shared["tqc"], shared["encoder"]
    encoder = EncoderSpec(lidar_beams=de.LIDAR_BEAMS, conv=tuple(tuple(c) for c in e["conv"]), other_hidden=tuple(e["other_hidden"]))
    cfg = TQCConfig(
        obs_dim=de.OBS,
        act_dim=de.ACTIONS,
        encoder=encoder,
        critics=t["critics"],
        quantiles=t["quantiles"],
        drop=own["drop"],
        gamma=own["gamma"],
        n_step=t["n_step"],
        target_entropy=own["target_entropy"],
        lr=t["lr"],
        batch=t["batch"],
        tau=t["tau"],
        actor_hidden=tuple(t["actor_hidden"]),
        critic_hidden=tuple(t["critic_hidden"]),
        init_alpha=t["init_alpha"],
    )
    ev = shared["env"]
    env = de.EnvConfig(
        races=p["races"],
        cars=1,
        min_laps=ev["min_laps"],
        race_km=ev["race_km"],
        pit_share=ev["pit_share"],
        pit_km=ev["pit_km"],
        time_factor=ev["time_factor"],
        worn=tuple(ev["worn"]),
    )
    unknown = set(own.get("weights", {})) - set(rewards.STYLES)
    if unknown:
        raise ValueError(f"{profile}: unknown style terms {sorted(unknown)} (have: {', '.join(rewards.STYLES)})")
    styles = {s: float(own.get("weights", {}).get(s, 1.0)) for s in rewards.STYLES}
    return Settings(
        profile=profile,
        description=own.get("description", ""),
        tqc=cfg,
        env=env,
        weights=rewards.weights(shared.get("rewards"), styles),
        styles=styles,
        start_from=own.get("start_from", ""),
        stage_a=shared["stage_a"],
        preset=p,
    )
