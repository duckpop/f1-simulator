"""
Car designs: 5 integer levels in a fixed order (engine, battery, chassis,
brakes, wing), each within [design] levels (1-5), at most max_points (17) in
total. The stock car (3, 3, 3, 3, 3) is the v1 car exactly.

Each level moves its category's gains and costs by `step` (5%); effects of
different categories on the same quantity multiply, and every resulting
quantity stays within +-max_change (15%) of the stock car:

    engine    +top speed, +acceleration (traction and power)   -braking
    battery   +ERS capacity, +ERS harvest rate                   -steering angle (the whole limit)
    chassis   +cornering grip (base and downforce)               -top speed, -acceleration
    brakes    +braking                                           +drag
    wing      -drag                                              +ERS drain, -braking

Also here: legality, sampling, the per-car physics parameters a design gives
(numpy arrays, one value per car), the designer's point operations (keep /
add / move) and their masks, and the designs file (configs/designs.toml).
Illegal designs and operations are refused wherever they're read.
"""

import itertools
import tomllib
import types

import numpy as np

ENGINE, BATTERY, CHASSIS, BRAKES, WING = range(5)
CATEGORIES = ("engine", "battery", "chassis", "brakes", "wing")


# -------------------------------------------------------------------------
# Legality and sampling
# -------------------------------------------------------------------------

def stock(cfg, n=None):
    """The stock design, or n copies of it (n, 5)."""
    d = np.array(cfg.design.stock, dtype=np.int64)
    return d if n is None else np.tile(d, (n, 1))


def legal(cfg, designs):
    """Per design (..., 5): every level within [design] levels and the sum
    at most max_points."""
    d = np.asarray(designs)
    lo, hi = cfg.design.levels
    return np.all((d >= lo) & (d <= hi), axis=-1) & (d.sum(axis=-1) <= cfg.design.max_points)


def check(cfg, design, what="design"):
    """A legal design as an int array (5,); ValueError otherwise."""
    try:
        d = np.asarray(design, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{what} {design!r}: expected {len(CATEGORIES)} integer levels") from None
    if d.shape != (len(CATEGORIES),) or not np.all(np.isfinite(d)) or not np.all(d == np.round(d)):
        raise ValueError(f"{what} {design!r}: expected {len(CATEGORIES)} integer levels "
                         f"({', '.join(CATEGORIES)})")
    d = d.astype(np.int64)
    if not legal(cfg, d):
        lo, hi = cfg.design.levels
        raise ValueError(f"{what} {text(d)} is illegal: every level {lo}-{hi} and at most "
                         f"{cfg.design.max_points} points in total (it has {int(d.sum())})")
    return d


def per_car(cfg, designs, n):
    """None (stock), one design or one per car -> (n, 5) legal designs."""
    if designs is None:
        return stock(cfg, n)
    d = np.asarray(designs)
    if d.ndim == 1:
        return np.tile(check(cfg, d), (n, 1))
    if d.shape != (n, len(CATEGORIES)):
        raise ValueError(f"expected one design or {n} designs, got shape {d.shape}")
    return np.stack([check(cfg, x, f"car {i}'s design") for i, x in enumerate(d)])


_LEGAL = {}


def all_legal(cfg):
    """Every legal design (2,438 with the default settings), (k, 5)."""
    lo, hi = cfg.design.levels
    key = (lo, hi, cfg.design.max_points)
    if key not in _LEGAL:
        every = np.array(list(itertools.product(range(lo, hi + 1), repeat=len(CATEGORIES))), dtype=np.int64)
        _LEGAL[key] = every[legal(cfg, every)]
        _LEGAL[key].flags.writeable = False
    return _LEGAL[key]


def random_legal(cfg, rng, n):
    """n uniformly random legal designs."""
    every = all_legal(cfg)
    return every[rng.integers(len(every), size=n)].copy()


def sample_training(cfg, rng, n):
    """A training episode's cars: the stock car with probability
    train_stock_share, otherwise a uniformly random legal design."""
    out = random_legal(cfg, rng, n)
    out[rng.random(n) < cfg.design.train_stock_share] = stock(cfg)
    return out


# -------------------------------------------------------------------------
# Effects: per-car physics parameters
# -------------------------------------------------------------------------

def factors(cfg, designs):
    """The capped multipliers (each array (cars,), 1 for the stock car) of
    every quantity a design moves."""
    dz = cfg.design
    delta = np.asarray(designs, dtype=np.float64).reshape(-1, len(CATEGORIES)) - np.asarray(dz.stock)
    g = 1.0 + dz.step * delta                      # a gain
    c = 1.0 - dz.step * delta                      # a cost

    def clip(x):
        return np.clip(x, 1.0 - dz.max_change, 1.0 + dz.max_change)

    drag = clip(g[:, BRAKES] * c[:, WING])
    return types.SimpleNamespace(
        accel=clip(g[:, ENGINE] * c[:, CHASSIS]),
        drag=drag,
        top_speed=clip(g[:, ENGINE] * c[:, CHASSIS] * drag ** (-1.0 / 3.0)),
        brake=clip(c[:, ENGINE] * g[:, BRAKES] * c[:, WING]),
        grip=clip(g[:, CHASSIS]),
        steer=clip(c[:, BATTERY]),
        ers=clip(g[:, BATTERY]),
        ers_drain=clip(g[:, WING]),
    )


def car_params(cfg, designs):
    """Per-car physics parameters (numpy arrays, one value per car):

        v_top        top speed: full throttle tops out exactly here (m/s)
        a_trac       traction limit (m/s^2)
        power        engine power: power-limited acceleration = power / v
        drag         drag constant: power / v_top^3
        brake        full braking (m/s^2)
        grip         cornering grip factor (x the episode's grip variation)
        steer        steering factor, on the steering cap and s_max
        ers_capacity battery (s of full deployment), ers_harvest, ers_drain

    The stock design gives the v1 car exactly."""
    f = factors(cfg, designs)
    v = cfg.vehicle
    power = v.a_max * v.v_switch * f.accel
    v_top = cfg.v_max * f.top_speed
    return types.SimpleNamespace(
        v_top=v_top, a_trac=v.a_max * f.accel, power=power, drag=power / v_top ** 3,
        brake=cfg.engine.brake_max * f.brake, grip=f.grip, steer=f.steer,
        ers_capacity=cfg.ers.capacity_s * f.ers, ers_harvest=cfg.ers.harvest_rate * f.ers, ers_drain=f.ers_drain)


def inputs(cfg, designs):
    """The design as network inputs: (level - stock) / half the level range
    (stock = 0, range -1..+1), (cars, 5) float32."""
    lo, hi = cfg.design.levels
    d = np.asarray(designs, dtype=np.float32).reshape(-1, len(CATEGORIES))
    return (d - np.asarray(cfg.design.stock, dtype=np.float32)) / ((hi - lo) / 2.0)


def text(design):
    """'4-3-2-3-5'."""
    return "-".join(str(int(x)) for x in np.asarray(design).ravel())


# -------------------------------------------------------------------------
# The designer's point operations
# -------------------------------------------------------------------------
# 0 = keep; 1..5 = add 1 to a category; then move 1 point from category i
# to j (i != j): 1 + 5 + 20 = 26 operations.

OP_FROM = np.array([-1] * (1 + len(CATEGORIES))
                   + [i for i, j in itertools.permutations(range(len(CATEGORIES)), 2)], dtype=np.int64)
OP_TO = np.array([-1] + list(range(len(CATEGORIES)))
                 + [j for i, j in itertools.permutations(range(len(CATEGORIES)), 2)], dtype=np.int64)
NUM_OPS = len(OP_FROM)


def op_text(op):
    op = int(op)
    if op == 0:
        return "keep"
    if OP_FROM[op] < 0:
        return f"+{CATEGORIES[OP_TO[op]]}"
    return f"{CATEGORIES[OP_FROM[op]]}->{CATEGORIES[OP_TO[op]]}"


def op_mask(cfg, designs):
    """(teams, NUM_OPS) bool: which operations keep each design legal.
    keep always; add j if below max_points and j below the top level; move
    i -> j if i above the bottom level and j below the top."""
    d = np.asarray(designs).reshape(-1, len(CATEGORIES))
    lo, hi = cfg.design.levels
    can_add = d.sum(axis=1, keepdims=True) < cfg.design.max_points
    up = d < hi                                          # (teams, 5)
    down = d > lo
    mask = np.zeros((len(d), NUM_OPS), dtype=bool)
    mask[:, 0] = True
    add = OP_FROM < 0
    add[0] = False
    mask[:, add] = up[:, OP_TO[add]] & can_add
    move = OP_FROM >= 0
    mask[:, move] = down[:, OP_FROM[move]] & up[:, OP_TO[move]]
    return mask


def apply_ops(cfg, designs, ops):
    """The designs after one operation each; ValueError for an illegal one."""
    d = np.asarray(designs, dtype=np.int64).reshape(-1, len(CATEGORIES)).copy()
    ops = np.asarray(ops, dtype=np.int64).reshape(-1)
    if len(ops) != len(d) or np.any((ops < 0) | (ops >= NUM_OPS)):
        raise ValueError(f"expected {len(d)} operations in 0..{NUM_OPS - 1}, got {ops.tolist()}")
    ok = op_mask(cfg, d)[np.arange(len(d)), ops]
    if not ok.all():
        k = int(np.flatnonzero(~ok)[0])
        raise ValueError(f"illegal operation {op_text(ops[k])} on {text(d[k])}")
    rows = np.arange(len(d))
    src, dst = OP_FROM[ops], OP_TO[ops]
    take = src >= 0
    d[rows[take], src[take]] -= 1
    give = dst >= 0
    d[rows[give], dst[give]] += 1
    return d


# -------------------------------------------------------------------------
# Designs file
# -------------------------------------------------------------------------

def read_designs(cfg, path=None):
    """[(team name, design)] from the designs file ([[team]] tables with
    name and design); illegal designs are refused."""
    from racer.config import resolve
    path = resolve(path or cfg.design.designs_file)
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except OSError as e:
        raise ValueError(f"can't read the designs file '{path}': {e.strerror}") from None
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"designs file '{path}': {e}") from None
    teams = data.get("team", [])
    if not teams:
        raise ValueError(f"designs file '{path}' has no [[team]] entries")
    out = []
    for k, t in enumerate(teams):
        unknown = set(t) - {"name", "design"}
        if unknown or "design" not in t:
            raise ValueError(f"designs file '{path}', team {k + 1}: expected name and design, got {sorted(t)}")
        name = str(t.get("name", f"Team {k + 1}"))
        out.append((name, check(cfg, t["design"], f"{path.name}: {name!r}")))
    return out


def write_designs(cfg, teams, path=None, note=""):
    """Writes [(team name, design)] as the designs file."""
    from racer.config import resolve
    path = resolve(path or cfg.design.designs_file)
    lines = ["# Car designs for watch.py / compare.py --designs (season.py --save-designs writes this file).",
             f"# Levels per category, in this order: {', '.join(CATEGORIES)}; each "
             f"{cfg.design.levels[0]}-{cfg.design.levels[1]}, at most {cfg.design.max_points} points in total."]
    if note:
        lines.append(f"# {note}")
    for name, d in teams:
        d = check(cfg, d, repr(name))
        lines += ["", "[[team]]", f"name = {_toml_str(name)}",
                  f"design = [{', '.join(str(int(x)) for x in d)}]"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(path)
    return path


def _toml_str(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'
