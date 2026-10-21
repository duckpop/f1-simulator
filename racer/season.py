"""
Seasons for the designer bot: [season] teams x 2 cars, each team one car
design and two drivers. A season races the tracks once each in a random
order; before every race the designer may make up to max_moves point
operations on a team's car, one after the other, then the race pays F1
points (team points = both cars). Races are like eval races: grid start,
the normal car variation off, everyone at full speed, 150 s, the grid order
shuffled each race.

One general designer designs every team's car in training. Test seasons
(fixed seeds) give it half the field - every other team - while the rest
keep their random starting car, or race the stock car; each test season is
raced again with the halves swapped. Its score is its teams' F1 points per
race: above the rivals' means its cars beat keeping a random car (or the
stock car). With every team designed, F1 points are zero-sum (a race pays
101 points whatever the cars), so the field average can't judge it.

The season workers live here too (torch-free, `spawn`): training workers
race seasons with the shared designer weights (epsilon-greedy) and send each
team's (observation, operation, reward, next observation) back; test
workers race test seasons greedily with the weights a task hands them.
"""

import gc
import queue
import signal

import numpy as np

from racer import design
from racer.env import RaceEnv
from racer.policy import FrozenPolicy, WorkerPolicy, greedy_ops
from racer.scoring import F1_POINTS, f1_points
from racer.track import available_maps

RESULT_SIZE = 7                                  # per driver in the designer's inputs
TEAM_POINTS_MAX = F1_POINTS[0] + F1_POINTS[1]    # a one-two: 43
# A test season's runs: the rivals keep their random car / race the stock
# car, with the designer on the even teams (half 0) or the odd ones (half 1).
TEST_RUNS = (("keep", 0), ("keep", 1), ("stock", 0), ("stock", 1))


class Season:
    """One season: every team's car, drivers, the calendar and the results
    so far. rng draws the starting cars (uniformly random legal), each team's
    2 drivers from `profiles` (repeats allowed) and the calendar, in that
    order - so seasons from the same seed match whatever else differs.
    designed: the teams the designer works on (bool per team; default all);
    rivals: what the others race - "keep" (their random starting car) or
    "stock"."""

    def __init__(self, cfg, rng, maps, profiles, designed=None, rivals="keep"):
        s = cfg.season
        self.cfg = cfg
        self.teams = T = s.teams
        self.designs = design.random_legal(cfg, rng, T)
        self.drivers = np.array(profiles, dtype=object)[rng.integers(len(profiles), size=(T, 2))]
        self.profile_idx = np.vectorize(list(s.profiles).index, otypes=[np.int64])(self.drivers)
        maps = sorted(maps)
        order = []
        while len(order) < s.races:
            order += [maps[k] for k in rng.permutation(len(maps))]
        self.calendar = order[:s.races]
        self.designed = np.ones(T, dtype=bool) if designed is None else np.asarray(designed, dtype=bool)
        self.rivals = rivals
        if rivals == "stock":
            self.designs[~self.designed] = design.stock(cfg)
        self.start = self.designs.copy()
        self.race = 0                            # the next race
        self.last = None                         # the last race: per driver (T, 2, RESULT_SIZE), team points (T,)
        self.points = np.zeros((T, 2))
        self.history = []                        # per race: map, the cars it was raced with, points, places

    def done(self):
        return self.race >= len(self.calendar)

    # ---- the designer -----------------------------------------------------------
    def observe(self, k):
        """The designer's inputs before operation k of the next race, (teams,
        designer_obs_dim): the car (levels as design.inputs, points used /
        max_points), each driver's profile one-hot, the last race (per
        driver: finished, place / cars (1 if out), crashed into a wall, into
        a car, stuck, passes made / 10, times passed / 10; then team points
        / 43 and "no race yet") and which operation this is (0 first, 1 last)."""
        cfg = self.cfg
        T, C, P = self.teams, len(cfg.design.categories), len(cfg.season.profiles)
        out = np.zeros((T, cfg.designer_obs_dim), dtype=np.float32)
        out[:, :C] = design.inputs(cfg, self.designs)
        out[:, C] = self.designs.sum(axis=1) / cfg.design.max_points
        k0 = C + 1
        rows = np.arange(T)
        for seat in range(2):
            out[rows, k0 + seat * P + self.profile_idx[:, seat]] = 1.0
        k0 += 2 * P
        if self.last is None:
            out[:, k0 + 2 * RESULT_SIZE + 1] = 1.0
        else:
            results, team_points = self.last
            out[:, k0:k0 + 2 * RESULT_SIZE] = results.reshape(T, -1)
            out[:, k0 + 2 * RESULT_SIZE] = team_points / TEAM_POINTS_MAX
        out[:, -1] = k / max(1, cfg.season.max_moves - 1)
        return out

    def design_step(self, choose):
        """Before the next race: max_moves operations on every designed team's
        car, one after the other (the second sees the car after the first).
        choose(obs, mask) -> one operation per row (the designed teams' rows
        only). Illegal operations are refused. Returns [(obs, ops)] of the
        designed teams per operation."""
        cfg = self.cfg
        taken = []
        idx = np.flatnonzero(self.designed)
        if not len(idx):
            return taken
        for k in range(cfg.season.max_moves):
            obs = self.observe(k)[idx]
            mask = design.op_mask(cfg, self.designs[idx])
            ops = np.asarray(choose(obs, mask), dtype=np.int64)
            self.designs[idx] = design.apply_ops(cfg, self.designs[idx], ops)
            taken.append((obs, ops))
        return taken

    # ---- races ---------------------------------------------------------------------
    def run_race(self, env, policies, rng, stop_event=None):
        """The next race on env (2 x teams cars, on this race's map): the grid
        order shuffled, every car its team's design and its driver's
        profile. policies: {profile: FrozenPolicy}. Returns each team's F1
        points, or None if stopped."""
        cfg, T = self.cfg, self.teams
        n = 2 * T
        entry = rng.permutation(n)               # grid slot -> entry: team entry // 2, seat entry % 2
        team, seat = entry // 2, entry % 2
        profiles = self.drivers[team, seat]
        env.set_profiles([str(p) for p in profiles])
        env.set_designs(self.designs[team])
        groups = [(policies[p], profiles == p) for p in dict.fromkeys(profiles)]
        obs = env.reset(rng, spawn="grid")
        while not env.done.all():
            if stop_event is not None and stop_event.is_set():
                return None
            actions = np.zeros((n, cfg.action_dim), dtype=np.float32)
            racing = ~env.done
            for policy, mine in groups:
                idx = mine & racing
                if idx.any():
                    actions[idx] = policy.act(obs[idx])
            obs, *_ = env.step(actions)

        results = np.zeros((T, 2, RESULT_SIZE), dtype=np.float32)
        points = np.zeros((T, 2))
        places = np.zeros((T, 2), dtype=np.int64)            # 0 = out
        reasons = np.empty((T, 2), dtype=object)
        for i in range(n):
            pos, why = env.finish_position[i], env.termination_reason[i]
            results[team[i], seat[i]] = (pos is not None, pos / n if pos else 1.0, why == "collision",
                                         why == "contact", why == "no_progress", min(env.overtakes[i] / 10, 1.0),
                                         min(env.times_overtaken[i] / 10, 1.0))
            points[team[i], seat[i]] = f1_points(pos)
            places[team[i], seat[i]] = pos or 0
            reasons[team[i], seat[i]] = why
        team_points = points.sum(axis=1)
        self.last = (results, team_points)
        self.points += points
        self.history.append({"map": env.map_name, "designs": self.designs.copy(), "points": points,
                             "places": places, "reasons": reasons})
        self.race += 1
        return team_points

    # ---- results -----------------------------------------------------------------
    def points_per_race(self, teams):
        """Average F1 points per team per race of the given teams (bool mask)."""
        teams = np.asarray(teams, dtype=bool)
        if not self.race or not teams.any():
            return None
        return float(self.points[teams].sum() / (teams.sum() * self.race))

    def summary(self):
        """A JSON-able record: calendar, drivers, which teams were designed,
        each team's car and points at every race, and the points per race of
        the designer's teams and of the rivals."""
        return {"calendar": self.calendar[:self.race], "drivers": self.drivers.tolist(),
                "designed": self.designed.tolist(), "rivals": self.rivals,
                "designs": [h["designs"].tolist() for h in self.history],
                "points": [h["points"].tolist() for h in self.history],
                "designer_ppr": self.points_per_race(self.designed),
                "rival_ppr": self.points_per_race(~self.designed)}


def transitions(cfg, season, taken, team_points):
    """After a race: every operation before it, credited with its team's F1
    points in that race, discounted by gamma to the next race's first
    decision (also after the season's last race: the designer has no clock,
    so the end of a season bootstraps like a driver's timeout).
    (obs, op, reward, next obs, discount, next legal ops) as arrays."""
    idx = np.flatnonzero(season.designed)
    next_obs = season.observe(0)[idx]
    next_mask = design.op_mask(cfg, season.designs[idx])
    k = len(taken)
    obs = np.concatenate([o for o, _ in taken])
    ops = np.concatenate([a for _, a in taken])
    reward = np.tile(team_points[idx], k).astype(np.float32)
    discount = np.full(len(ops), cfg.designer.gamma, dtype=np.float32)
    return obs, ops, reward, np.tile(next_obs, (k, 1)), discount, np.tile(next_mask, (k, 1))


class EnvCache:
    """One race env at a time, rebuilt when the map changes (every race of a
    season is on another track)."""

    def __init__(self, cfg, cars, seed):
        self.cfg, self.cars, self.seed = cfg, cars, seed
        self.env = None

    def get(self, map_name):
        if self.env is None or self.env.map_name != map_name:
            self.close()
            self.env = RaceEnv(self.cfg, self.cars, map_name, self.cfg.season.profiles[0], seed=self.seed)
        return self.env

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            gc.collect()                         # the gym env has reference loops


def play(cfg, season, policies, envs, choose, rng_for, stop_event=None, on_race=None):
    """Races the rest of a season: rng_for(race index) (called first), the
    designer's operations (choose(obs, mask) -> ops; None: the cars stay as
    they are), then the race. on_race(taken, team_points) after each.
    Returns False if stopped."""
    while not season.done():
        r = season.race
        rng = rng_for(r)
        taken = season.design_step(choose) if choose is not None else []
        points = season.run_race(envs.get(season.calendar[r]), policies, rng, stop_event)
        if points is None:
            return False
        if on_race is not None:
            on_race(taken, points)
    return True


def test_season(cfg, index, rivals, half, params, policies, envs, stop_event=None, on_race=None):
    """Test season `index` (seed [test_seed, index]: the same starting cars,
    drivers, calendar and grids every time), the designer greedy on the even
    (half 0) or odd (half 1) teams, the rest racing their random starting car
    (rivals "keep") or the stock car. None if stopped."""
    seed = cfg.designer.test_seed
    profiles = [p for p in cfg.season.profiles if p in policies]
    designed = np.arange(cfg.season.teams) % 2 == half
    season = Season(cfg, np.random.default_rng([seed, index]), available_maps(cfg), profiles, designed, rivals)
    ok = play(cfg, season, policies, envs, lambda obs, mask: greedy_ops(params, obs, mask),
              lambda r: np.random.default_rng([seed, index, r]), stop_event, on_race)
    return season if ok else None


# -------------------------------------------------------------------------
# Season workers (torch-free)
# -------------------------------------------------------------------------

def _put(q, item, stop_event):
    """A blocking put that gives up when the run stops."""
    while not stop_event.is_set():
        try:
            q.put(item, timeout=0.5)
            return
        except queue.Full:
            continue


def train_worker_main(worker_id, cfg, shared, epsilon, decision_queue, stats_queue, stop_event, seed, drivers):
    """Races training seasons back to back, every team designed by the shared
    designer, epsilon-greedy (epsilon: a shared value the learner lowers),
    and sends each race's decisions to the learner. drivers: {profile: actor
    weights} - teams draw their 2 drivers from these profiles."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)          # only the learner reacts to Ctrl+C
    rng = np.random.default_rng(seed)
    maps = available_maps(cfg)
    policies = {p: FrozenPolicy(w) for p, w in drivers.items()}
    profiles = [p for p in cfg.season.profiles if p in policies]
    designer = WorkerPolicy(shared)
    envs = EnvCache(cfg, 2 * cfg.season.teams, seed)
    stats_queue.put({"ready": worker_id})

    def choose(obs, mask):
        ops = greedy_ops(designer.params, obs, mask)
        for t in np.flatnonzero(rng.random(len(ops)) < epsilon.value):
            ops[t] = rng.choice(np.flatnonzero(mask[t]))
        return ops

    def race_rng(r):
        designer.refresh()                       # the latest weights before every race's decisions are made
        return rng

    def on_race(taken, points):
        _put(decision_queue, transitions(cfg, season, taken, points), stop_event)
        ops = np.concatenate([a for _, a in taken])
        stats_queue.put({"worker": worker_id, "race": season.race, "map": season.history[-1]["map"],
                         "team_points": points.tolist(), "designs": season.designs.tolist(),
                         "ops": np.bincount(ops, minlength=design.NUM_OPS).tolist()})

    try:
        while not stop_event.is_set():
            season = Season(cfg, rng, maps, profiles)
            play(cfg, season, policies, envs, choose, race_rng, stop_event, on_race)
    finally:
        envs.close()


def test_worker_main(worker_id, cfg, task_queue, stats_queue, stop_event, drivers):
    """Races test seasons for the learner: each task is (round, test season
    index, rivals, half, designer weights); the finished season's summary
    goes back on the stats queue."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    policies = {p: FrozenPolicy(w) for p, w in drivers.items()}
    envs = EnvCache(cfg, 2 * cfg.season.teams, 4242 + worker_id)
    stats_queue.put({"ready": worker_id})
    try:
        while not stop_event.is_set():
            try:
                round_id, index, rivals, half, params = task_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            season = test_season(cfg, index, rivals, half, params, policies, envs, stop_event)
            if season is None:
                break
            stats_queue.put({"test": True, "round": round_id, "index": index, "rivals": rivals, "half": half,
                             "season": season.summary()})
    finally:
        envs.close()
