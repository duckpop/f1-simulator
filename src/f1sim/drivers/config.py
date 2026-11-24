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
    weights: np.ndarray  # (terms,) reward weight per term, the reward scale included
    styles: dict  # the profile's multiplier per style term
    stage_a: dict
    stage_b: dict
    preset: dict
    reward_scale: float = 1.0  # [rewards] scale: the weighted sum x this (already in `weights`)
    warm_start: dict = None  # [warm_start]: fill (share of the replay), critic_steps and actor_ramp (gradient steps), for a run from a checkpoint
    stage_c: dict = None  # [stage_c]: what differs from [stage_b] in stage C (team orders) and D
    stage_d: dict = None  # [stage_d]: what differs from [stage_c] in stage D (the personalities)


def load(profile: str, preset: str = "laptop", configs=CONFIGS) -> Settings:
    configs = Path(configs)
    shared = tomllib.loads((configs / "drivers.toml").read_text())
    own = tomllib.loads((configs / "profiles" / f"{profile}.toml").read_text())
    if preset not in shared["preset"]:
        raise ValueError(f"no preset {preset!r} in drivers.toml (have: {', '.join(shared['preset'])})")
    p = shared["preset"][preset]
    t, e = shared["tqc"], shared["encoder"]
    encoder = EncoderSpec(lidar_beams=de.LIDAR_BEAMS, lidar_channels=de.LIDAR_CHANNELS, conv=tuple(tuple(c) for c in e["conv"]), other_hidden=tuple(e["other_hidden"]), bf16=bool(e.get("bf16", False)))
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
        shared_encoder=bool(t.get("shared_encoder", False)),
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
        race=de.EnvConfig().race._replace(respawn=bool(ev.get("respawn", True))),
        sensor_noise=float(ev.get("sensor_noise", 0.0)),
        crash_left_km=float(ev.get("crash_left_km", 0.0)),
        march_steps=int(ev.get("march_steps", de.MARCH_STEPS)),
    )
    unknown = set(own.get("weights", {})) - set(rewards.STYLES)
    if unknown:
        raise ValueError(f"{profile}: unknown style terms {sorted(unknown)} (have: {', '.join(rewards.STYLES)})")
    styles = {s: float(own.get("weights", {}).get(s, 1.0)) for s in rewards.STYLES}
    unknown = set(own.get("rewards", {})) - set(rewards.TERMS)
    if unknown:
        raise ValueError(f"{profile}: unknown reward terms {sorted(unknown)} (have: {', '.join(rewards.TERMS)})")
    shared_rewards = dict(shared.get("rewards", {}))
    scale = float(shared_rewards.pop("scale", 1.0))
    base = {**shared_rewards, **own.get("rewards", {})}  # a profile's own base values replace the shared ones
    return Settings(
        profile=profile,
        description=own.get("description", ""),
        tqc=cfg,
        env=env,
        weights=rewards.weights(base, styles, scale),
        styles=styles,
        stage_a=shared["stage_a"],
        stage_b=shared["stage_b"],
        preset=p,
        reward_scale=scale,
        warm_start=dict(shared["warm_start"]),
        stage_c=dict(shared.get("stage_c", {})),
        stage_d=dict(shared.get("stage_d", {})),
    )
