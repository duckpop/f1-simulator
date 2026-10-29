"""The scripted training principal (Phase 2, section 3.1): it only calls pit stops.

- Box when the tyres are at ~10% life: the threshold is drawn per stint from 8-12%, so drivers don't learn one exact
  lap. A car is called for the pit entry where its life will be closest to the threshold (from its wear per km on this
  set); not on the last lap.
- Weather: wets when rain arrives within a lap (the exact forecast) or it's raining; slicks (mediums) when the track
  will be dry enough within a lap. One stop covers both a worn set and a weather change.
- Compound: the next-harder at each dry stop (soft -> medium -> hard, hard -> medium); a car still on one dry compound
  in a dry race comes in from the second-to-last lap (the two-compound rule).
- Pace normal, no team orders, ERS "deploy": set once at the start of each race (`fixed_orders`).

It decides once a second, like f1env's principal; a standing call is changed only when the weather changes the
compound it wants.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from f1env import orders as od
from f1env import pits as pt
from f1env import tyres as ty
from f1env import weather as wx
from f1env.race import RACING, RaceConfig

NEXT_HARDER = (ty.MEDIUM, ty.HARD, ty.MEDIUM, ty.MEDIUM)  # soft, medium, hard, wet (wets: back to slicks on mediums)


def draw_worn(key, shape, worn=(0.08, 0.12)):
    """The tyre life each stint is boxed at."""
    return jax.random.uniform(key, shape, minval=worn[0], maxval=worn[1])


def slick_crossover(p: ty.TyreParams) -> float:
    """Track wetness below which mediums grip better than wets."""
    dry_m, wet_m = p.dry_grip[ty.MEDIUM], p.wet_grip[ty.MEDIUM]
    dry_w, wet_w = p.dry_grip[ty.WET], p.wet_grip[ty.WET]
    return (dry_m - dry_w) / ((dry_m - wet_m) - (dry_w - wet_w))


def calls(bank, state, worn_at, cfg: RaceConfig = RaceConfig()):
    """(R, C) pit calls for `f1env.race.pit_orders`: -1 (no change) or 1 + compound. worn_at: (R, C) this stint's
    threshold."""
    total = bank.length[state.tid][:, None]
    leader_lap = (jnp.max(jnp.floor(state.line / total), axis=-1) + 1).astype(jnp.int32)
    last_lap = (leader_lap >= state.target_laps)[:, None]
    late = (leader_lap >= state.target_laps - 1)[:, None]

    w = state.weather
    rain_soon = (~w.raining & (wx.laps_to_change(w, leader_lap) <= 1))[:, None]
    wet_in_a_lap = (w.wetness - 1.0 / cfg.weather.dry_laps)[:, None] >= slick_crossover(cfg.tyres)
    wet_track = w.raining[:, None] | rain_soon | wet_in_a_lap

    t = state.tyres
    on_wets = t.compound == ty.WET
    # life at this lap's pit entry, less half a lap: at or below the threshold, this entry is the closest to it
    to_entry = jnp.mod(bank.pit_entry_s[state.tid][:, None] - state.s, total)
    at_entry = t.life - ty.wear_per_km(t, cfg.tyres) * (to_entry + 0.5 * total) / 1000.0
    worn = ((t.life <= worn_at) | (at_entry <= worn_at)) & ~last_lap
    wrong_tyres = wet_track != on_wets
    second_compound = ~w.rained[:, None] & ~ty.two_compounds(t) & ~on_wets & late
    want = 1 + jnp.where(wet_track, ty.WET, jnp.asarray(NEXT_HARDER)[t.compound])

    p = state.pits
    standing = jnp.where(p.call > 0, p.call, p.committed)  # 0: no stop called
    free = (state.status == RACING) & (p.stage == pt.OFF)
    return jnp.where(free & (worn | wrong_tyres | second_compound) & (standing != want), want, -1)


def fixed_orders(state):
    """Pace normal, no team order, ERS deploy, for every car."""
    o = state.orders
    one = jnp.ones_like(o.pace)
    return state._replace(orders=o._replace(pace=one * od.NORMAL, team=one * od.NO_ORDER, ers=one * od.DEPLOY))
