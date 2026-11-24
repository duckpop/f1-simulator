"""Close calls (since 2026-11-06, after v2's): how near a car is to hitting another car or a barrier, before it does.

A close call's level is 0..1 from the time to contact at the current closing speed: 1 - ttc / TTC_S, clipped, and 0
unless the gap shrinks faster than MIN_CLOSING. A crash is the only signal a driver gets otherwise, and it comes too late
to say which of the steps before it was the mistake; a close call says it a fraction of a second earlier, every time,
even when the car gets away with it. The rewards charge each per second at its level (`car_close`, `wall_close`), and
each profile weighs them by its style (Aggressive less for cars, Defensive and Smooth more).

- **Cars**: every pair's body-to-body gap (each body as f1env's 3 circles) and how fast it closes along the line
  between them; a car takes the worst over the others, wrecks included, and both cars of a pair pay (either can make
  room).
- **Barriers**: the nearest barrier to any point of the body outline (f1env's barrier distance field), and how fast that
  gap shrinks since the last step. The nearest one only: v2 measured that a per-beam time to contact fires on most clean
  steps at speed (every corner's outside wall is "0.4 s ahead").
"""

from __future__ import annotations

import jax.numpy as jnp

from f1env.race import BODY_POINTS, RACING, WRECK
from f1env.sensors import BODY_CENTRE, BODY_LENGTH, BODY_WIDTH, sample_field

TTC_S = 0.4  # s: a contact closer than this in time is a close call (v2's)
MIN_CLOSING = 1.0  # m/s: a gap shrinking slower than this isn't one
DT = 0.05


def level(gap, closing):
    """0..1: how close a gap (m) shrinking at `closing` m/s is to contact within TTC_S."""
    closing = jnp.where(jnp.isfinite(closing) & (closing > MIN_CLOSING), closing, 0.0)
    ttc = jnp.where(closing > 0, jnp.maximum(gap, 0.0) / jnp.maximum(closing, 1e-6), jnp.inf)
    return jnp.clip(1.0 - ttc / TTC_S, 0.0, 1.0)


def cars(race):
    """(R, C) each racing car's car close call: the worst over the other cars on the track (racing or wrecked)."""
    c = race.cars
    r = BODY_WIDTH / 2
    offs = jnp.array([-(BODY_LENGTH / 2 - r), 0.0, BODY_LENGTH / 2 - r])
    co, si = jnp.cos(c.yaw), jnp.sin(c.yaw)
    bx, by = c.x + BODY_CENTRE * co, c.y + BODY_CENTRE * si
    cx = bx[..., None] + offs * co[..., None]  # (R, C, 3)
    cy = by[..., None] + offs * si[..., None]
    dx = cx[:, :, None, :, None] - cx[:, None, :, None, :]  # (R, i, j, a, b)
    dy = cy[:, :, None, :, None] - cy[:, None, :, None, :]
    rr, n = dx.shape[0], dx.shape[1]
    dx, dy = dx.reshape(rr, n, n, 9), dy.reshape(rr, n, n, 9)
    d2 = dx * dx + dy * dy + 1e-9
    k = jnp.argmin(d2, axis=-1)[..., None]
    dmin = jnp.sqrt(jnp.take_along_axis(d2, k, axis=-1)[..., 0])  # the root of the nearest pair only
    nx = jnp.take_along_axis(dx, k, axis=-1)[..., 0] / dmin  # from j towards i
    ny = jnp.take_along_axis(dy, k, axis=-1)[..., 0] / dmin
    gap = dmin - 2 * r
    vx, vy = c.vx * co - c.vy * si, c.vx * si + c.vy * co  # world velocity
    closing = -((vx[:, :, None] - vx[:, None, :]) * nx + (vy[:, :, None] - vy[:, None, :]) * ny)
    racing = race.status == RACING
    body = racing | (race.status == WRECK)
    pair = racing[:, :, None] & body[:, None, :] & ~jnp.eye(n, dtype=bool)[None]
    return jnp.max(jnp.where(pair, level(gap, closing), 0.0), axis=-1)


def barrier_gap(bank, race):
    """(R, C) m from the body outline's nearest point to a barrier (0 or less: touching it)."""
    c = race.cars
    co, si = jnp.cos(c.yaw)[..., None], jnp.sin(c.yaw)[..., None]
    lx, ly = BODY_POINTS[:, 0] + BODY_CENTRE, BODY_POINTS[:, 1]
    px = c.x[..., None] + lx * co - ly * si
    py = c.y[..., None] + lx * si + ly * co
    t = jnp.broadcast_to(race.tid[:, None, None], px.shape)
    return jnp.min(sample_field(bank, bank.barrier, t, px, py), axis=-1)


def walls(gap, prev_gap, racing):
    """(R, C) each racing car's barrier close call from its barrier gap now and a step ago."""
    return jnp.where(racing, level(gap, (prev_gap - gap) / DT), 0.0)
