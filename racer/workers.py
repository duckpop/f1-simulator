"""
Rollout workers: one process per track, running episodes back to back
with a private numpy copy of the actor. Each turns its trained brain's cars'
steps into n-step transitions and ships them to the learner in chunks;
worker 0 also runs the eval rounds. Every car of a training episode gets its
own design (opponents too: they see theirs); evals race the stock car.
Torch-free.
"""

import gc
import queue
import signal
from collections import deque

import numpy as np

from racer import design
from racer.env import RaceEnv
from racer.policy import ExplorationNoise, FrozenPolicy, WorkerPolicy
from racer.track import train_maps


class NStep:
    """Per-step (s, a, r) of one car -> n-step (s, a, R, s', discount):
    discount = gamma^k, or 0 when the episode ended in a crash (a timeout
    bootstraps). The learner never needs gamma."""

    def __init__(self, n, gamma):
        self.n, self.gamma = n, gamma
        self.buf = deque()

    def push(self, s, a, r, next_s, terminated, truncated, out):
        self.buf.append((s, a, r))
        if terminated or truncated:
            while self.buf:
                self._emit(next_s, terminated, out)
                self.buf.popleft()
        elif len(self.buf) == self.n:
            self._emit(next_s, False, out)
            self.buf.popleft()

    def _emit(self, next_s, terminal, out):
        ret, g = 0.0, 1.0
        for _, _, r in self.buf:
            ret += g * r
            g *= self.gamma
        s0, a0, _ = self.buf[0]
        out.append((s0, a0, ret, next_s, 0.0 if terminal else g))


def flush(chunk, transition_queue, global_steps, stop_event):
    """Ship a chunk as numpy arrays. The put blocks while the queue is full:
    a slow learner makes the workers wait instead of changing the replay ratio."""
    if not chunk:
        return
    s, a, r, ns, d = zip(*chunk)
    payload = (np.stack(s), np.stack(a), np.asarray(r, np.float32), np.stack(ns), np.asarray(d, np.float32))
    while not stop_event.is_set():
        try:
            transition_queue.put(payload, timeout=0.5)
            break
        except queue.Full:
            continue
    with global_steps.get_lock():
        global_steps.value += len(chunk)
    chunk.clear()


def assign_drivers(num_cars, rng, num_opponents, self_share, front_slots=0):
    """Who drives each car: -1 = the brain being trained (max(1,
    round(self_share x cars)) cars on random slots - among the first
    front_slots when > 0; a car's index is its grid slot and its place in a
    rolling start's pack), k = opponent k (the rest, in turn, in a shuffled
    order - a profile listed twice gets twice the cars)."""
    drivers = np.full(num_cars, -1)
    own = min(num_cars, max(1, int(round(self_share * num_cars))))
    if num_opponents == 0 or own == num_cars:
        return drivers
    if front_slots > 0:
        mine = rng.choice(min(num_cars, max(front_slots, own)), size=own, replace=False)
        others = np.setdiff1d(np.arange(num_cars), mine)
    else:
        others = rng.permutation(num_cars)[own:]
    rounds = -(-len(others) // num_opponents)
    drivers[others] = np.concatenate([rng.permutation(num_opponents) for _ in range(rounds)])[:len(others)]
    return drivers


def duels(env, drivers, names):
    """{opponent: [comparisons the brain's cars finished ahead, comparisons]}
    over every (own car, opponent car) pair. Ahead = survived when the other
    didn't, or the same outcome and further up the order."""
    out = {}
    own = np.flatnonzero(drivers < 0)
    if not names or not len(own):
        return out
    survived = np.array([r == "timeout" for r in env.termination_reason], dtype=float)
    key = survived * 1e9 + env.position_scores()
    for j in np.flatnonzero(drivers >= 0):
        rec = out.setdefault(names[drivers[j]], [0, 0])
        rec[0] += int((key[own] > key[j]).sum())
        rec[1] += len(own)
    return out


def episode_stats(env, returns, drivers, names):
    """Per-car lists cover every car; "learner" marks the trained brain's."""
    own = drivers < 0
    steps = np.maximum(env.racing_steps, 1)
    return {
        "map": env.map_name, "cars": env.num_agents, "spawn": env.spawn,
        "learner": own.tolist(),
        "drivers": [env.profile_names[i] if d < 0 else names[d] for i, d in enumerate(drivers)],
        "designs": env.designs.tolist(),
        "duels": duels(env, drivers, names),
        "car_steps": int(env.racing_steps[own].sum()),
        "steps": env.steps,
        "returns": returns.tolist(),
        "progress_m": env.progress.tolist(),
        "laps": env.laps().tolist(),
        "max_speed": env.max_speed.tolist(),
        "avg_speed": env.avg_speed().tolist(),
        "reasons": list(env.termination_reason),
        "positions": list(env.finish_position),
        "overtakes": env.overtakes.tolist(),
        "passed": env.times_overtaken.tolist(),
        "passes_on_opp": env.passes_on[:, drivers >= 0].sum(axis=1).tolist(),
        "passed_by_opp": env.passes_on[drivers >= 0, :].sum(axis=0).tolist(),
        "tow_frac": (env.tow_steps / steps).tolist(),
        "close_calls": env.close_calls.tolist(),
        "purple": env.purple_sectors.tolist(),
        "best_laps": env.best_laps(),
        "lap_times": [t for i, car in enumerate(env.lap_times) if own[i] for t in car],
        "drs_frac": (env.drs_steps / steps).tolist(),
        "ers_frac": (env.ers_steps / steps).tolist(),
        "battery": (env.battery_sum / steps).tolist(),
    }


def worker_main(worker_id, cfg, shared_policy, transition_queue, stats_queue, stop_event, global_steps, seed,
                skip_warmup, car_counts, is_evaluator, profile, opponents=(), eval_grids=1):
    """cfg: the resolved settings (CLI overrides included). car_counts: the
    --cars list, redrawn at every map switch. opponents: (name, profile,
    actor weights) per --opponents entry - they drive greedily with their own
    profile on a car slowed to opponent_speed, and only the trained brain's
    cars send transitions. eval_grids: races per eval map."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)          # only the learner reacts to Ctrl+C
    rng = np.random.default_rng(seed)
    maps = train_maps(cfg)
    names = [name for name, _, _ in opponents]
    opp_profiles = [prof for _, prof, _ in opponents]
    frozen = [FrozenPolicy(params) for _, _, params in opponents]
    self_share = cfg.workers.self_share if frozen else 1.0
    td3, sp = cfg.td3, cfg.spawn
    # Fastest time through each sector per track, for the lap-pace reward:
    # the whole run, shared by this worker's training envs (evals keep their own).
    sector_records = {}

    def build(map_name, num_cars, training=True):
        env = RaceEnv(cfg, num_cars, map_name, profile, seed=seed)
        if training:
            env.sector_record = sector_records.setdefault(map_name, np.full(cfg.race.lap_sectors, np.inf))
        return env

    def data_until_switch():
        return cfg.sim.map_switch_car_steps * rng.uniform(0.75, 1.25)

    map_name = str(rng.choice(maps))
    env = build(map_name, int(rng.choice(car_counts)))
    switch_in = data_until_switch()           # car-steps of the brain's data left on this map
    eval_envs = {}                            # one per eval map, the most cars of the list, built on first use
    policy = WorkerPolicy(shared_policy)
    chunk = []
    stats_queue.put({"ready": worker_id})

    def run_episode(run_env, is_eval, grid=0):
        """Training episodes refresh the policy as they go and stream
        transitions; eval episodes drive one fixed network greedily."""
        n = run_env.num_agents
        if is_eval:
            spawn = "grid"
        elif n > 1 and rng.random() < sp.rolling_probability:
            spawn = "rolling"
        else:
            p = sp.grid_probability_big_field if n >= sp.big_field else sp.grid_probability
            spawn = "grid" if rng.random() < p else "random"
        drivers = assign_drivers(n, np.random.default_rng(grid) if is_eval else rng, len(frozen), self_share,
                                 cfg.workers.front_slots)
        own = drivers < 0
        if frozen:
            run_env.set_profiles([profile if d < 0 else opp_profiles[d] for d in drivers])
            if not is_eval:
                run_env.speed_share[drivers >= 0] *= cfg.variation.opponent_speed
        run_env.set_designs(None if is_eval else design.sample_training(cfg, rng, n))
        # Evals race the full length; training races a random length (more starts per hour).
        run_env.max_steps = cfg.episode_steps if is_eval else int(rng.integers(cfg.train_race_steps[0],
                                                                               cfg.train_race_steps[1] + 1))
        obs = run_env.reset(rng, spawn=spawn, vary_car=not is_eval)
        nstep = [NStep(td3.n_step, td3.gamma) if own[i] else None for i in range(n)]
        returns = np.zeros(n)
        noise = ExplorationNoise(cfg, rng, n)
        since_refresh = 0
        while not run_env.done.all() and not stop_event.is_set():
            if not is_eval:
                if since_refresh >= cfg.workers.policy_refresh_steps:
                    policy.refresh()
                    since_refresh = 0
                since_refresh += 1
            racing = ~run_env.done
            actions = np.zeros((n, cfg.action_dim), dtype=np.float32)
            drive = own & racing
            if not is_eval and not skip_warmup and global_steps.value < td3.warmup_steps:
                actions[own] = rng.uniform(-1.0, 1.0, size=(int(own.sum()), cfg.action_dim))
            else:
                if drive.any():
                    actions[drive] = policy.act(obs[drive])
                if not is_eval:
                    actions[own] = np.clip(actions[own] + noise()[own], -1.0, 1.0)
            for k, opp in enumerate(frozen):
                idx = (drivers == k) & racing
                if idx.any():
                    actions[idx] = opp.act(obs[idx])
            active = np.flatnonzero(racing)
            next_obs, reward, terminated, truncated, info = run_env.step(actions)
            used = info["actions"]                    # incl. the launch override
            returns[active] += reward[active]
            if not is_eval:
                for i in active[own[active]]:
                    nstep[i].push(obs[i], used[i], float(reward[i]), next_obs[i],
                                  bool(terminated[i]), bool(truncated[i]), chunk)
                if len(chunk) >= cfg.workers.transition_chunk:
                    flush(chunk, transition_queue, global_steps, stop_event)
            obs = next_obs
        return returns, drivers

    # On a resume the evaluator scores the loaded driver straight away (policy
    # version 1, before any training): the run's baseline.
    every = cfg.eval.every_publishes
    last_eval_version = 1 - every if skip_warmup else 0
    while not stop_event.is_set():
        policy.refresh()
        if is_evaluator and every > 0 and policy.version - last_eval_version >= every:
            last_eval_version = policy.version
            params = [p.copy() for p in policy.params]
            results = []
            for k, eval_map in enumerate(cfg.eval.maps):
                if eval_map not in eval_envs:
                    eval_envs[eval_map] = build(eval_map, max(car_counts), training=False)
                for g in range(eval_grids):
                    # Every map and grid its own slots, the same every round.
                    returns, drivers = run_episode(eval_envs[eval_map], True, [k, g] if eval_grids > 1 else 0)
                    if stop_event.is_set():
                        break
                    results.append(episode_stats(eval_envs[eval_map], returns, drivers, names))
                if stop_event.is_set():
                    break
            if stop_event.is_set():
                break
            stats_queue.put({"worker": worker_id, "eval_round": True, "results": results,
                             "policy_version": policy.version, "actor_params": params})
            continue

        if switch_in <= 0 and (len(maps) > 1 or len(set(car_counts)) > 1):
            env.close()
            env = None
            gc.collect()                    # the gym env has reference loops: free the old track now
            if len(maps) > 1:
                map_name = str(rng.choice([m for m in maps if m != map_name]))
            env = build(map_name, int(rng.choice(car_counts)))
            switch_in = data_until_switch()
        returns, drivers = run_episode(env, False)
        if stop_event.is_set():
            break
        stats = episode_stats(env, returns, drivers, names)
        switch_in -= stats["car_steps"]
        stats.update(worker=worker_id, eval_round=False)
        stats_queue.put(stats)

    env.close()
    for e in eval_envs.values():
        e.close()
