# F1 Racers Sim — v2 specification: car design, car-aware drivers, designer bot

Hand-off document for the second version. It builds on `project.md` and on the code in this
repository (`racer/`, `train.py`, `watch.py`, `compare.py`, `report.py`). Everything not mentioned
here stays as it is. Where this document gives a number, use it.

**What changes** (everything else: keep):

1. **Physics** (§1): steering is scaled to the grip limit, and braking / accelerating and cornering
   share one grip budget (traction circle).
2. **Car design** (§2): every car has a design of 5 categories (engine, battery, chassis, brakes,
   wing), levels 1–5, at most 17 points. Each level moves its gain and its cost by 5%; every
   resulting quantity is capped at ±15% of the stock car.
3. **Sight distance** (§3): lidar range and track preview stretched to the fastest legal car.
4. **Car-aware drivers** (§5): a driver sees its car's design as 5 inputs, trains on random legal
   cars, and is evaluated on the stock car.
5. **Designer bot** (§6): one general, learned designer. Each team starts a season with a random
   car; before every race the designer sees its drivers and their last results and adds or moves
   0–2 points to fit the car to them.
6. **Seasons** (§6.1): 11 teams × 2 cars = 22 cars, 11 different designs on track.
7. **New observation layout** (§5.1): every v1 checkpoint is refused; v1 models move to
   `saved_models_v1/` and all styles retrain (§8).

**Not in v2:** a Team Principal bot (who drives which car). Teams get their drivers as described in
§6.1.

---

## 1. Physics changes

### 1.1 Steering scaled to the grip limit

- **Problem (measured, v1 middle's best, Spielberg):** the steering output maps to ±`s_max` (±34°),
  but the grip limit leaves ±1.6° at 75+ m/s, i.e. 4.7% of the output range. Anything beyond is
  clipped, so at speed the steering sits on the limit 100% of the time and flips side ~8 times a
  second (a weave). It costs no speed in the kinematic model, so training never removed it.
- **Change:** the steering output is a share of the steering available right now:
  `target angle = steer × cap`, where `cap` is the steering limit of §1.2 (grip limit with the
  traction circle, DRS, the car's design and the episode's grip variation), never above the car's
  `s_max`.
  - The full output range is usable at every speed; 0 = straight, ±1 = at the grip limit.
  - The smooth-steering rate limit (`sv_min` / `sv_max`) is unchanged.
  - The "previous steer output" input now means "share of the available steering". The input
    "steering angle / s_max" stays.
- **Check:** at 75+ m/s the steering is at the limit well under 50% of the time and flips side
  less than once a second on straights (same measurement as above).

### 1.2 Traction circle

- **Problem:** braking (2 g) and cornering (up to 5 g) use separate budgets, so a car can brake
  fully while turning at the limit; trail-braking costs nothing.
- **Change:** the tyre's longitudinal and lateral use share one budget per car per step:
  - `a_x` = the tyre's longitudinal acceleration this step: `throttle × engine(v) − brake × B`
    (engine and brakes only; drag, tow and DRS pull are aerodynamic and don't use the tyres).
  - `A_x` = the car's traction limit `a_trac` when `a_x ≥ 0`, its braking limit `B` when `a_x < 0`.
  - `u = min(1, |a_x| / A_x)`; lateral share `= max(combined_floor, sqrt(1 − u²))`.
  - The lateral grip limit used for the steering cap is multiplied by that share.
- **Floor:** `combined_floor = 0.2`: a car keeps 20% of its lateral grip at full braking, and during
  the forced full-throttle launch of a grid start.
- **Effect:** brake in a straight line, then turn; a car that brakes hard into a corner runs wide.
  Full throttle at high speed uses little of the budget (power-limited: at 80 m/s about 44% of
  `a_trac`, leaving ~90% of the lateral grip).
- **Check:** at full braking the steering cap is `combined_floor` × the free-rolling cap; with
  `a_x = 0` it equals the cap without the circle.

---

## 2. Car design

### 2.1 The design vector

- 5 integer levels in this fixed order: **(engine, battery, chassis, brakes, wing)**, each 1–5.
- **Stock car = (3, 3, 3, 3, 3)** = today's car exactly (15 points).
- **Legal design:** every level 1–5 and the sum ≤ **17** (3 per category + 2), so a car has two
  free upgrade levels and pays for more with downgrades. 2,438 legal designs; 320 use all 17.
- Illegal designs are refused wherever a design is read (designs file, CLI, designer actions).

### 2.2 Effects

For each category `Δ = level − 3` (−2…+2). A **gain** multiplies by `(1 + 0.05 Δ)`, a **cost** by
`(1 − 0.05 Δ)`; drag and drain costs multiply by `(1 + 0.05 Δ)`. Every level moves its quantities
by 5%. Effects of different categories on the same quantity multiply, and **every resulting
quantity is capped at ±15% of the stock car** (0.85…1.15).

| Category | Gain per level | Cost per level |
|---|---|---|
| **engine** | +5% top speed, +5% acceleration (traction and power) | −5% braking |
| **battery** | +5% ERS capacity, +5% ERS recharge (harvest rate) | −5% steering angle (the whole steering limit) |
| **chassis** | +5% cornering grip (whole grip limit: base and downforce) | −5% top speed, −5% acceleration |
| **brakes** | +5% braking | +5% drag |
| **wing** | −5% drag | +5% ERS drain, −5% braking |

### 2.3 Per-car parameters

With `g(Δ) = 1 + 0.05 Δ`, `c(Δ) = 1 − 0.05 Δ`, `clip(x) = min(1.15, max(0.85, x))` and today's
values (`a_max` 20, `v_switch` 35, `v_max` 86.1):

| Parameter | Formula | Stock |
|---|---|---|
| acceleration factor `fa` | `clip(g(Δeng) · c(Δcha))` | 1 |
| drag factor `fd` | `clip(g(Δbrk) · c(Δwing))` | 1 |
| top speed `v_top` | `v_max · clip(g(Δeng) · c(Δcha) · fd^(−1/3))` | 86.1 m/s |
| traction limit `a_trac` | `a_max · fa` | 20 m/s² |
| engine power `P` (power-limited accel = `P / v`) | `a_max · v_switch · fa` | 700 |
| drag constant `K` | `P / v_top³` (full throttle tops out exactly at `v_top`) | as today |
| braking `B` | `a_max · clip(c(Δeng) · g(Δbrk) · c(Δwing))` | 20 m/s² |
| grip factor | `clip(g(Δcha))` × the episode's grip variation | 1 |
| steering factor | `clip(c(Δbat))` on the steering cap and `s_max` | 1 |
| ERS capacity / harvest | `[ers] capacity_s · clip(g(Δbat))` / `harvest_rate · clip(g(Δbat))` | 4 s / 0.5 |
| ERS drain | `throttle × dt × clip(g(Δwing))` per deploying step | as today |

- **Ranges over all legal designs:** top speed 263–356 km/h; braking 17.0–23.0 m/s²; acceleration
  ±15%; grip, steering and ERS ±10% (one category each, so the cap doesn't bind).
- **Engine:** `engine(v) = min(a_trac, power × P / max(v, 0.1))`, where `power` is today's pace /
  opponent-speed factor (`(cap / v_max)³`), unchanged.
- **Pace spread and opponent speed** apply on top, as today.
- **Gym constraint:** the gym's KS dynamics clip the acceleration with its own `a_max` / `v_switch`.
  The racing-only step integrates each car with its own parameter row: `a_max = max(a_trac, B)`,
  `v_switch = P / a_trac`, `v_max` = the physics ceiling below.
- **Physics ceiling:** `356 km/h (99.0 m/s) × tow × DRS × ERS gains × 1.05 ≈ 131 m/s`
  (today 113.6).
- **Speed budget:** a 22-car race must still step at ≤ 6 ms/step (per-car parameters are numpy
  arrays; no new per-car Python in the hot path).

### 2.4 Designs file

`configs/designs.toml` holds named designs for `watch.py` / `compare.py --designs`; `season.py`
can write the designs a season ended with (§6.4):

```toml
[[team]]
name = "Team 1"
design = [4, 3, 2, 3, 5]         # engine, battery, chassis, brakes, wing
```

---

## 3. Sight distance

- **Rule kept from v1:** lidar range ≈ 1.35 × the stopping distance from top speed, track preview
  ≈ 1.6 × it (v1: 185 m stop, 250 m lidar, 300 m preview).
- **Worst legal car:** design (5, 5, 2, 1, 4): 99.0 m/s (356 km/h) with 17.0 m/s² braking stops in
  **288 m** (1.56 × stock).
- **New values:** `lidar_range = 390` m; `track_preview_m = [25, 50, 100, 150, 200, 300, 470]`
  (one point added at the far end, so slow cars keep the close-range detail).
- The lidar inputs stay `sqrt(clip(d / lidar_range, 0, 1))`.

---

## 4. Settings (`configs/settings.toml` changes)

```toml
[grip]
# ... existing keys ...
steering_scaled = true      # steer output x the available steering (§1.1)
traction_circle = true      # braking / accelerating and cornering share the tyre (§1.2)
combined_floor = 0.2        # lateral grip kept at full braking / full traction

[sensors]
lidar_range = 390.0         # (was 250) fitted to the fastest legal car (§3)
track_preview_m = [25.0, 50.0, 100.0, 150.0, 200.0, 300.0, 470.0]   # (one point added)

[design]
categories = ["engine", "battery", "chassis", "brakes", "wing"]
levels = [1, 5]
stock = [3, 3, 3, 3, 3]
max_points = 17
step = 0.05                 # per level, gains and costs
max_change = 0.15           # every resulting quantity within +-15% of the stock car
train_stock_share = 0.2     # share of training cars on the stock car; the rest uniform random legal designs
designs_file = "configs/designs.toml"

[season]
teams = 11                  # 2 cars each: 22 cars, 11 designs on track
races = 23                  # one per track, in a random order each season
max_moves = 2               # point operations the designer may make before each race

[variation]
mass = 0.0                  # (was 0.10) the kinematic model ignores mass: it did nothing
```

`config.py` validates `[design]` (stock legal, levels within 1–5, `max_points` ≥ the stock sum) and
`[season]`.

---

## 5. Car-aware drivers

### 5.1 Observation

- Track preview: 6 → 7 inputs (§3).
- Append 5 inputs at the end: the car's levels as `(level − 3) / 2` (stock = 0, range −1…+1).
- Layout: 127 → **133** inputs; layout tag `racer-v2-133obs-5act`. v1 checkpoints are refused
  (never upgraded). Before the first v2 run, move `saved_models/` to `saved_models_v1/` (the v1 code
  can still watch them with `[run] save_dir = "saved_models_v1"`).

### 5.2 Training

- Every car of every training episode gets its own design: the stock car with probability
  `train_stock_share` (0.2), otherwise a uniformly random legal design.
- Frozen opponents get random designs too (they see theirs).
- Everything else as today (rewards, profiles, spawns, opponents, warm-ups).

### 5.3 Evaluation

- Eval rounds race every car on the **stock car** (3, 3, 3, 3, 3), so eval scores stay comparable
  across runs.
- The run signature adds the `[design]` settings, the physics flags of §1 and the sight distances.
- `compare.py`: stock cars by default; `--designs` races the teams of `configs/designs.toml`
  (each team's 2 cars on one design, seats swapped as today).
- `watch.py`: `--designs` gives car pairs the teams' designs; the standings panel shows each car's
  design vector next to its number.

---

## 6. Designer bot

One **general** designer: the same learned policy designs every team's car. Its job is to fit the
car to the two drivers it has, over a season.

### 6.1 Seasons

- **Field:** 11 teams × 2 cars. At the start of a season each team gets a **uniformly random legal
  car** and its two drivers.
- **Drivers:** in designer training each team gets **2 random profiles** (middle / aggressive /
  defensive, repeats allowed), driven by those profiles' trained car-aware checkpoints (frozen,
  torch-free, as opponents are today).
- **Calendar:** the 23 tracks once each, in a random order per season.
- **Races:** like eval races: grid start, normal car variation off, everyone at full speed, 150 s,
  grid order shuffled each race.
- **Points:** F1 points per car (25-18-15-12-10-8-6-4-2-1, crash 0); team points = both cars.

### 6.2 What the designer sees (before each race)

| Block | Size | Content |
|---|---|---|
| car | 6 | the 5 levels as `(level − 3) / 2`; points used / 17 |
| drivers | 6 | each driver's profile, one-hot over the 3 profiles |
| last race | 2 × 7 + 2 | per driver: finished (0/1), place / 22 (1 if out), crashed into a wall (0/1), into a car (0/1), stuck (0/1), passes made / 10, times passed / 10 (both clipped to 1); then team points / 43 and "no race yet" (1 before the season's first race, all other last-race inputs 0) |
| this decision | 1 | which point operation this is (0 = first, 1 = second) |

Total: 29 inputs. It does **not** see the next track.

### 6.3 What it can do

- Up to `max_moves` = **2 point operations** before each race, decided one after the other (the
  second sees the car after the first). Each operation is one of 26:
  - **keep** (no change);
  - **add** 1 to a category (only if the car is below 17 points and the category below 5): 5;
  - **move** 1 point from category i to j (i above 1, j below 5): 20.
- Illegal operations are masked out.
- **Reward:** the team's F1 points in the next race, credited to both operations before it.
- **Learning:** double DQN with a small network (29 → 128 → 128 → 26), discount 0.9 per race
  (a change can pay off over the next races), replay buffer, ε-greedy exploration decaying
  1.0 → 0.05. The network is tiny; it can train on the CPU next to the race workers.
- **Data:** season workers (torch-free, `spawn`, one 22-car env each, the design-policy weights
  through shared memory like the driver's) run races and send each team's (observation,
  operation, reward, next observation) back. About 13 s per race per worker → with 12 workers
  ~1 race/s → 22 designer decisions per race → ~80k decisions an hour.

### 6.4 Evaluating the designer

- **Fixed test seasons** (fixed seeds: random starting cars, profiles and calendars), the same for
  every check.
- **Score:** average team points per race over the season, compared with the same seasons when
  teams keep their random starting car, and when every team races the stock car.
- **Best designer** saved by that score (`saved_models/designer/designer_best.pt`), with
  `metrics.csv` and a season report: each team's car before each race and its points.
- `season.py --seed N` runs one season with the best designer and prints the championship
  (drivers and teams) and each team's car evolution; `--save-designs` writes the final 11 designs
  to `configs/designs.toml`.

---

## 7. Code changes (by module)

| Module | Change |
|---|---|
| `racer/config.py` | `[design]`, `[season]`, validation, layout v2 (133 inputs) |
| `racer/design.py` **(new)** | design vectors: legality, sampling, effects → capped per-car parameters (§2.3), point operations and their masks (§6.3), designs file I/O |
| `racer/physics.py` | per-car parameters in `longitudinal_accel`, `max_steer`, `ers_update`; steering scaled to the cap; traction circle; per-car parameter rows in `_integrate` |
| `racer/env.py` | `set_designs(designs)` per car (like `set_profiles`), per-car parameters, design inputs in the observation |
| `racer/sensors.py` | `build_obs` appends the design block; 7 preview points |
| `racer/workers.py` | random designs per training car, stock in evals; designs in episode stats |
| `racer/season.py` **(new)** | seasons: teams, calendar, races, points, designer observations / operations, season workers |
| `racer/agent.py` | the designer's DQN next to TD3 (networks, replay, checkpoints) |
| `train.py` | header, signature |
| `train_designer.py` **(new entry point)** | learner + season workers for the designer |
| `season.py` **(new entry point)** | runs and prints a season with the trained designer |
| `watch.py`, `compare.py` | `--designs` |

No test files (as in v1).

---

## 8. Training plan v2

1. **Move `saved_models/` to `saved_models_v1/`.**
2. **middle from scratch** on random cars, `--cars 1,4,8,12,16,22`, left alone 12–16 h (a harder
   task than v1: every car is different).
3. **aggressive and defensive** from middle's best with `--opponents` as in v1 (new-layout
   opponents only), 5–10 h each.
4. **Designer** with the three styles' best checkpoints as the drivers, until its test-season score
   stops improving (hours; ~80k decisions an hour).
5. Judge drivers with `compare.py` on stock cars and on `--designs`; judge the designer with its
   test seasons and `season.py`.

---

## 9. Done when

- **Physics:** the checks of §1.1 and §1.2 pass; the stock car still measures 0–100 km/h 1.45 s,
  0–200 3.4 s, braking 86→20 m/s in 175 m (in a straight line).
- **Design:** each level moves exactly its quantities by ±5% until the ±15% cap (checked on the
  per-car parameters for every category); illegal designs and operations are refused; the stock
  design gives today's parameters; top speed spans 263–356 km/h.
- **Sight:** lidar 390 m, 7 preview points; the fastest legal car brakes for corners it sees
  (survives a solo race on Monza and Spa).
- **Drivers:** 133 inputs; random designs in training, stock in evals; a car-aware middle survives
  and races on the 11 designs (`compare.py --designs`).
- **Designer:** never makes an illegal car; on the test seasons it scores more points per race than
  teams keeping their random car; `season.py` prints a season and its car changes.
- **Speed:** a 22-car race ≤ 6 ms/step; driver training still GPU-bound.
