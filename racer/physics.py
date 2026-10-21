"""
The car: grip limit, traction circle, smooth steering, engine / brakes /
drag, slipstream, DRS and ERS, and the racing-only physics step installed on
the gym's simulator.

The physics model is the gym's kinematic single-track model (KS) with the
[car] parameters - its dynamic model has linear tyres that never saturate
and spun under braking. The grip limit stands in for downforce: the steering
angle is capped each step to what the tyres can hold at that speed, less
what braking / accelerating takes from them (the traction circle).

Every car has its own parameters from its design (design.car_params: numpy
arrays, one value per car - `car` below), and its own row of the gym's
parameters in the integrator.
"""

import numpy as np
from numba import njit

from f1tenth_gym.envs.dynamic_models.kinematic import vehicle_dynamics_ks


# -------------------------------------------------------------------------
# Grip and steering
# -------------------------------------------------------------------------

def max_lateral_accel(cfg, speed, grip_factor=1.0, downforce=1.0):
    """Sideways acceleration (m/s^2) the tyres hold at `speed`: 1.8 g in slow
    corners rising to 5 g in fast ones. downforce: the share of the downforce
    grip available (less with DRS open)."""
    g = cfg.grip
    return 9.81 * grip_factor * np.minimum(
        g.max_g, g.base_g + downforce * g.downforce_g * (np.asarray(speed) / g.ref_speed) ** 2)


def max_steer(cfg, speed, grip_factor=1.0, downforce=1.0, steer_factor=1.0, lateral_share=1.0):
    """Largest steering angle within the grip limit at `speed`
    (steady-state cornering: a = v^2 tan(angle) / wheelbase). grip_factor:
    the design's grip x the episode's grip variation; steer_factor: the
    design's steering factor, on the whole limit (never above its s_max);
    lateral_share: the grip braking / accelerating leaves for cornering."""
    v2 = np.maximum(np.asarray(speed, dtype=np.float64), 1.0) ** 2
    a_lat = max_lateral_accel(cfg, speed, grip_factor, downforce) * lateral_share
    return steer_factor * np.minimum(cfg.vehicle.s_max, np.arctan(cfg.wheelbase * a_lat / v2))


def lateral_share(cfg, car, tyre_accel):
    """Traction circle: the share of the lateral grip left after the tyre's
    longitudinal use this step. tyre_accel: throttle x engine - braking()
    (engine, brakes and engine braking - drag, tow and DRS pull don't use the
    tyres). u = |a_x| / (traction limit when a_x >= 0, braking limit when
    a_x < 0); share = max(combined_floor, sqrt(1 - u^2)). 1 everywhere with
    [grip] traction_circle off."""
    if not cfg.grip.traction_circle:
        return np.ones_like(tyre_accel)
    limit = np.where(tyre_accel >= 0, car.a_trac, car.brake)
    u = np.minimum(1.0, np.abs(tyre_accel) / limit)
    return np.maximum(cfg.grip.combined_floor, np.sqrt(1.0 - u * u))


def smooth_steering(control_dt):
    """The gym's steering function, replaced: the rate that reaches the
    target angle by the end of the control step, held for the whole step
    (the gym steers bang-bang at full rate and overshoots every step)."""
    def steer(target, state, params):
        return float(np.clip((target - state[2]) / control_dt, params.sv_min, params.sv_max))
    return steer


# -------------------------------------------------------------------------
# Longitudinal: engine, brakes, drag, tow, DRS, ERS
# -------------------------------------------------------------------------

def engine_accel(cfg, car, v, pace, ers_on):
    """The engine's available acceleration per car (m/s^2): traction-limited,
    then power-limited. pace: the share of the car's top speed it runs at
    (pace spread x opponent speed): power x pace^3, since top speed ~
    power^(1/3) under v^2 drag; ERS deploying (0/1) x (1 + power_boost)."""
    power = pace ** 3 * (1.0 + cfg.ers.power_boost * ers_on) * car.power
    return np.minimum(car.a_trac, power / np.maximum(np.maximum(v, 0.0), 0.1))


def braking(cfg, car, throttle, brake):
    """Each car's braking deceleration (m/s^2, >= 0): the brake command x the
    car's full braking, plus engine braking - [engine] brake_share of full
    braking - whenever the throttle is off. Together up to [engine]
    brake_limit x full braking off the throttle (lifting while braking stops
    shorter), the brakes alone (full braking) with the throttle on."""
    off = throttle <= 0
    engine_brake = cfg.engine.brake_share * car.brake * off
    limit = car.brake * np.where(off, cfg.engine.brake_limit, 1.0)
    return np.minimum(limit, brake * car.brake + engine_brake)


def longitudinal_accel(cfg, car, v, throttle, decel, engine, pace, tow_boost, drs_open, out):
    """Each car's acceleration (m/s^2). throttle: 0..1 command; decel:
    braking(); engine: engine_accel(); pace: as there; tow_boost:
    0..tow.boost; drs_open: 0/1; out: cars out of the race (they brake to a
    stop)."""
    v = np.maximum(v, 0.0)
    drs_gain = 1.0 + cfg.drs.boost * drs_open
    drag_scale = 1.0 / ((1.0 + tow_boost) * drs_gain) ** 3
    accel = throttle * engine - decel - car.drag * drag_scale * v * v
    # Extra pull in a tow / with DRS on the throttle, fading out over the last
    # 5 m/s below the boosted top speed so it can't push past it.
    top = car.v_top * pace * (1.0 + tow_boost) * drs_gain
    pull = (cfg.tow.accel * tow_boost / cfg.tow.boost if cfg.tow.boost > 0 else 0.0) + cfg.drs.accel * drs_open
    accel = accel + pull * np.clip((top - v) / 5.0, 0.0, 1.0) * (throttle > 0)
    # The limits the gym's KS model used to apply (see param_rows): up to the
    # traction / power curve, down to max(traction, full braking) - or, off the
    # throttle, brake_limit x full braking, so lifting while braking pays.
    a0 = np.maximum(car.a_trac, car.brake)
    up = a0 * np.minimum(1.0, (car.power / car.a_trac) / np.maximum(v, 0.1))
    down = np.where(throttle <= 0, np.maximum(a0, cfg.engine.brake_limit * car.brake), a0)
    accel = np.clip(accel, -down, up)
    return np.where(out, np.where(v > 0.1, -car.brake, 0.0), accel)


def tow_boost(cfg, geo, n):
    """Slipstream per car (0..tow.boost): the strongest wake of a racing car
    ahead along its own heading - full within full_dist ahead and
    +-full_width sideways, fading to nothing at max_dist / +-width - for a
    racing car above min_speed. geo: this step's pairwise geometry."""
    out = np.zeros(n)
    t = cfg.tow
    if geo.count < 2 or t.boost <= 0:
        return out
    along, side = geo.along, geo.side
    dist_fade = np.clip((t.max_dist - along) / (t.max_dist - t.full_dist), 0.0, 1.0)
    side_fade = np.clip((t.width - np.abs(side)) / (t.width - t.full_width), 0.0, 1.0)
    fade = np.where(along > 0, dist_fade * side_fade, 0.0).max(axis=1)
    out[geo.idx] = t.boost * fade * (geo.v >= t.min_speed)
    return out


def ers_update(cfg, car, battery, throttle, brake, ers_request, v, racing, harvest_scale=1.0):
    """One control step of the battery. Deploys (returned as 0/1) when asked,
    on the throttle and with charge left: engine power x (1 + power_boost),
    draining throttle x dt x the car's drain factor. Braking harvests the
    car's harvest rate x harvest_scale x brake x v/v_max x dt (harvest_scale:
    close_harvest_rate / harvest_rate for cars within close_gap_s of a car
    ahead, else 1). Coasting and engine braking harvest nothing. Within
    0..the car's capacity; updates battery in place."""
    dt = cfg.sim.control_dt
    deploy = (ers_request & (throttle > 0) & (battery > 0) & racing).astype(np.float64)
    battery -= throttle * dt * car.ers_drain * deploy
    battery += car.ers_harvest * harvest_scale * brake * (np.maximum(v, 0.0) / cfg.v_max) * dt * racing
    np.clip(battery, 0.0, car.ers_capacity, out=battery)
    return deploy


# -------------------------------------------------------------------------
# Racing-only physics step (installed on the gym's simulator)
# -------------------------------------------------------------------------

# Where the gym's KS dynamics read their constraints in a parameter row.
P_S_MIN, P_S_MAX, P_V_SWITCH, P_A_MAX, P_V_MAX = 8, 9, 12, 13, 15


def param_rows(cfg, base, car):
    """One row of the gym's parameters per car. Its KS constraints clip the
    steering with s_min / s_max (here the car's steering limit) and the speed
    at v_max (the physics ceiling); its acceleration clip (a_max / v_switch)
    is opened up - longitudinal_accel applies the same limits itself, plus
    the deeper braking off the throttle ([engine] brake_limit)."""
    rows = np.repeat(np.asarray(base, dtype=np.float64)[None, :], len(car.v_top), axis=0)
    rows[:, P_S_MIN] = -cfg.vehicle.s_max * car.steer
    rows[:, P_S_MAX] = cfg.vehicle.s_max * car.steer
    rows[:, P_V_SWITCH] = 1.0e4
    rows[:, P_A_MAX] = 1.0e4
    rows[:, P_V_MAX] = cfg.physics_v_max
    return rows


@njit(cache=True)
def _integrate(state, cars, steer_rate, accel, substeps, dt, params):
    """The gym's RK4 over its KS dynamics (constraints included), for the
    given cars, each with its own parameter row (params[i]), all substeps in
    one compiled loop - the gym's Python RK4 cost ~85 us per car-step.
    Writes the float32 state in place."""
    u = np.empty(2)
    for k in range(len(cars)):
        i = cars[k]
        x = state[i].astype(np.float64)
        p = params[i]
        u[0] = steer_rate[k]
        u[1] = accel[k]
        for _ in range(substeps):
            k1 = vehicle_dynamics_ks(x, u, p)
            k2 = vehicle_dynamics_ks(x + dt * (k1 / 2), u, p)
            k3 = vehicle_dynamics_ks(x + dt * (k2 / 2), u, p)
            k4 = vehicle_dynamics_ks(x + dt * k3, u, p)
            x = x + dt * (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x[4] = (x[4] + np.pi) % (2 * np.pi) - np.pi
        for j in range(5):
            state[i, j] = x[j]


def racing_step(env):
    """Replacement for F110Simulator.step: the same as the gym's, but it only
    integrates cars still racing (cleared cars stand still off the map), each
    with its own parameter row (env.param_rows). Steering is the smooth rate
    (smooth_steering) and the acceleration is used as given (ACCL mode).
    Then scans, car collisions and the clock."""

    def step(sim, control_inputs):
        steer = control_inputs[:, 0].astype(np.float32)
        accel = control_inputs[:, 1].astype(np.float32)
        st = sim.state
        st.control_input[:, 1] = accel
        if st.delay_buffer is not None:
            steer = st.push_delay(steer)
        st.control_input[:, 0] = steer
        cars = np.flatnonzero(~env.done)
        if len(cars):
            p = sim.vehicle_params
            rate = np.clip((steer[cars] - st.state[cars, 2]) / sim.time_step, p.sv_min, p.sv_max)
            _integrate(st.state, cars, rate.astype(np.float64), accel[cars].astype(np.float64),
                       sim.substeps, sim.integrator_dt, env.param_rows)
            s = st.state[cars]
            st.standard_state[cars, :5] = s
            st.standard_state[cars, 5:] = 0.0                  # KS reports no yaw rate / slip
            st.poses[cars] = s[:, [0, 1, 4]]
        sim._update_scans()
        sim._update_agent_collisions()
        st.sim_time += sim.time_step

    return step
