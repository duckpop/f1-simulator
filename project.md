# F1 Racers Sim — rebuild specification

Hand-off document for rebuilding the project from scratch. It describes everything the current
code does (so it can be replicated), the changes to make, and how the new code should be laid
out. Where this document gives a number, use it: every value here was measured or tuned in the
current project. The current code lives in `src/` of the old repo (`duckpop/F1_RACERS_SIM`, branch
`td3-training-fixes`) and can be read as a reference implementation, but do not copy its file layout.

**What changes compared with the current code** (everything else: replicate as described):

1. **New file layout**: one package of modules grouped by category (§3), settings and driving-style
   profiles in TOML files (§4), and three entry points: train, watch, compare (§9).
2. **New actions**: steer, throttle, brake, DRS, ERS (§5.2). The current car has one combined
   throttle/brake output plus DRS.
3. **DRS only on straights** (§5.4): computed DRS zones replace "anywhere within 1 s of the car
   ahead".
4. **ERS** (§5.5): a battery per car that deploys extra power and recharges under braking.
5. **Compact and fast**: no test files, no backwards compatibility with old checkpoints, and every
   speed-up listed in §10 built in from the start.

---

## 1. The project

Reinforcement learning (TD3) that teaches F1-like cars to race wheel to wheel on the 23 real F1
circuit layouts of the F1TENTH racetracks, in the F1TENTH gym. Up to 22 cars share a track.
One neural network drives every car of a "brain", from lidar-like perception. The cars race
from a staggered grid, a rolling start, or random spots on the track.

- **Driving styles (profiles):** `middle` (the baseline), `aggressive` and `defensive`. They differ
  only in reward weights and eval bonuses. Each profile trains its own network.
- **Opponents:** a profile can train against frozen copies of other profiles' networks.
- **Evals:** the best checkpoint of a run is picked by race results (place points) on fixed eval
  tracks.
- **Comparison:** a head-to-head tool races two networks against each other with F1 points.
- **Machine it must run well on:** 16 CPU cores, ~14 GB RAM, RTX 3050 Ti laptop GPU (4 GB). Linux.
  The GPU learner is the bottleneck (~950 updates/s); the simulator workers can supply ~6,500
  env steps/s.

**Training plan** (the reason the tools exist):
1. `middle` from scratch with `--cars 1,4,8,12,16,22`. Leave it running untouched for 8+ hours: the
   old middle's best came from one uninterrupted 7.6 h run.
2. `aggressive` and `defensive` from `middle`'s best, with `--opponents middle` (aggressive with
   `--opponent-speed 0.9`), `--lr 1e-4`, `--cars 4,8,12,16,22`.
3. Rounds: each style against the latest of the other two, 2–3 times.
4. Judge every result with the compare tool (§9.3), not just the eval score.

---

## 2. Dependencies

| What | Where | Notes |
|---|---|---|
| Python 3.14 | `.venv` made with `uv` | Run everything from the project root. |
| F1TENTH gym | `git clone -b dev-humble https://github.com/duckpop/f1tenth_gym.git f1tenth_gym` then `uv pip install -e f1tenth_gym` | The user's fork of `f1tenth/f1tenth_gym` (dev-humble). It adds 12 more car colours to the renderer's palette (needed for 22 cars) and a safer ST-model threshold. Use it unmodified; the project patches behaviour at runtime (§5.9). |
| Race tracks | `git clone --depth 1 https://github.com/f1tenth/f1tenth_racetracks.git maps/f1tenth_racetracks` | GPL-3.0: keep it out of git. Each track folder has `<Name>_map.yaml`, `<Name>_map.png`, `<Name>_centerline.csv`. |
| numpy, gymnasium 0.29, numba, scipy, PyYAML, PyQt6, pyqtgraph, PyOpenGL | come with the gym | |
| torch (CUDA) | `uv run --with torch ...` per run | Only the learner, the checkpoint loaders and the watcher import torch. Workers must not (it costs ~500 MB RAM per process). |
| rich | `uv run --with rich ...` | Live dashboard; optional, with a plain-text fallback. |
| matplotlib | `uv run --with matplotlib ...` | Progress plots only. |
| tomllib | stdlib | Reads the config files. |

`.gitignore`: `saved_models/`, `__pycache__/`, `.venv/`, `maps/f1tenth_racetracks/`.

Typical commands:
```bash
uv run --with torch --with rich train.py --profile middle --cars 1,4,8,12,16,22 --steps 200000000
uv run --with torch watch.py --cars 9 --profiles middle,aggressive,defensive --map Monza
uv run --with torch compare.py --a aggressive --b middle
```

---

## 3. File layout

One package, `racer/`, with one module per category of functions. Entry points sit at the project
root. Do not create a file for a single function, and do not write test files.

```
project.md
configs/
  settings.toml          all base settings (§4.1)
  profiles/
    middle.toml          driving-style overrides (§4.2)
    aggressive.toml
    defensive.toml
racer/
  config.py      load settings + profiles, validate, derived constants, paths
  track.py       maps, gym env construction + gym patches, centerline tracker, DRS zones, spawns
  physics.py     car parameters, grip limit, steering, engine/brake/drag/tow/DRS/ERS, racing-only sim step
  sensors.py     racing-only lidar update (walls-only copy, wall/car crash flags, ray-cast cull),
                 relative geometry, observation building (lidar, state, race, DRS/ERS, mirrors,
                 preview, alongside)
  env.py         RaceEnv: reset/step, race order, passes, points, rewards, crashes, laps/sectors, stats
  scoring.py     place points, F1 points, eval bonuses, per-map/round aggregation (evals and compare)
  policy.py      torch-free: numpy actor (forward pass), shared-memory weights, frozen policies,
                 exploration noise
  agent.py       torch: networks, GPU replay buffer, TD3 with CUDA graphs, checkpoints
  workers.py     rollout worker process: episodes, n-step transitions, drivers per car, map/car-count
                 rotation, eval rounds, episode stats
  console.py     number formatting, training-stats summary, printed status panels, rich dashboard
  records.py     metrics.csv / evals.csv, lap records, best-checkpoint metadata
train.py         learner + workers (§9.1)
watch.py         rendered races (§9.2)
compare.py       head to head with F1 points, and the lap-record leaderboard (§9.3)
report.py        progress plots from the CSV logs (§9.4)
saved_models/<profile>/   checkpoints and logs (git-ignored)
```

**Import rules**
- `config`, `track`, `physics`, `sensors`, `env`, `scoring`, `policy`, `workers` must never import
  torch.
- Only `agent.py` and the entry points do. The entry points import torch lazily, inside `main()`:
  `spawn` re-imports the main module in every worker.

---

## 4. Configuration files

### 4.1 `configs/settings.toml`

The base settings. These values are the current tuned ones; `[drs]` and `[ers]` are new.
`racer/config.py` loads this file into a read-only namespace and builds every derived constant
from it:
- observation layout and beam indices,
- vehicle parameters (gym's `FULLSCALE_VEHICLE_PARAMETERS` with `[car]` applied),
- progress scale and drag constant.

Workers receive the resolved settings (including CLI overrides) as arguments. Never mutate module
globals to pass settings around.

```toml
[car]                      # overrides the gym's FULLSCALE_VEHICLE_PARAMETERS
m = 798.0                  # 2024 F1 minimum weight with driver
I = 1200.0                 # yaw inertia, kg m^2
lf = 1.6
lr = 2.0                   # 3.6 m wheelbase
h = 0.30
mu = 1.8
a_max = 20.0               # ~2 g: engine traction limit (below v_switch) and full brakes
v_switch = 35.0            # above it the engine is power-limited: a = a_max * v_switch / v
v_max = 86.1               # 310 km/h, reached when engine accel = drag
v_min = 0.0                # no reversing
s_min = -0.6
s_max = 0.6                # steering angle, rad
sv_min = -1.5
sv_max = 1.5               # steering rate, rad/s
width = 2.0
length = 5.6
collision_body_center_x = 1.8

[grip]                     # max sideways accel = 9.81 * min(max_g, base_g + downforce * downforce_g * (v / ref_speed)^2)
base_g = 1.8
downforce_g = 3.2
ref_speed = 70.0
max_g = 5.0

[sim]
maps_dir = "maps/f1tenth_racetracks"
map_scale = 12             # tracks are 2.2 m wide at scale 1: ~26 m wide, laps ~3-6.5 km
train_maps = "all"         # or a list of names
map_switch_car_steps = 100000   # a worker changes map + car count after this much training data (+-25%)
control_dt = 0.05          # 20 Hz policy
integrator_dt = 0.01
episode_steps = 3000       # 150 s races
lidar_noise_std = 0.01

[sensors]
lidar_beams = 108          # simulated over the FOV
lidar_fov_deg = 270.0
lidar_range = 250.0        # braking from 86 m/s at 2 g takes ~185 m
obs_bands = [[25.0, 1], [60.0, 2], [135.0, 4]]   # (up to |angle| deg, every Nth beam) -> 50 of 108 beams
mirror_range = 60.0
mirror_width = 10.0
track_preview_m = [25.0, 50.0, 100.0, 150.0, 200.0, 300.0]
alongside_range = 6.0
max_cars_obs = 22          # the car-count input is divided by this

[spawn]
grid_probability = 0.5            # share of non-rolling training episodes that start on the grid
grid_probability_big_field = 0.8  # ... with big_field+ cars
big_field = 12
grid_slot_gap = 12.0       # m between consecutive cars (8 m left the pack too tight after the launch)
grid_columns = 2
grid_column_gap = 6.0
grid_max_turn_deg = 10.0   # the grid must sit on a straight
grid_clear_ahead = 60.0
rolling_probability = 0.35 # share of training episodes with 2+ cars
rolling_gap = [8.0, 16.0]
rolling_speed = [35.0, 60.0]
rolling_speed_jitter = 0.05
rolling_clear_ahead = 150.0
random_lateral_jitter = 1.0
random_heading_jitter_deg = 8.0
random_speed = [0.0, 15.0]
launch_s = 2.0             # grid starts: full throttle forced for 2 s
no_progress_timeout_s = 6.0
no_progress_blocked_gap = 30.0

[variation]                # training episodes only; evals and watching use the normal car
grip = 0.10                # tyre grip x U(0.9, 1.1), the same for every car on the track
mass = 0.10                # mass and yaw inertia x U(0.9, 1.1)
pace_spread = 0.04         # each car's top speed (engine power) cut by U(0, 4%)
opponent_speed = 0.95      # frozen opponents' top speed in training (CLI --opponent-speed)

[tow]                      # slipstream
boost = 0.10               # up to +10% top speed: drag x 1/(1+boost)^3
accel = 10.0               # up to +10 m/s^2 pull on the throttle
full_dist = 15.0
max_dist = 120.0
full_width = 1.5
width = 3.0
min_speed = 20.0

[drs]                      # NEW rules, same effect as before (§5.4)
zone_max_turn_deg = 4.0    # a point is in a DRS zone if the track turns < 4 deg from it to zone_lookahead_m ahead
zone_lookahead_m = 200.0   # so every zone ends ~200 m before the next corner (the braking distance)
min_zone_length_m = 250.0  # shorter straights get no zone
gap_s = 1.0                # also needs the car ahead within 1 s; 0 = any car on a straight
from_s = 30.0              # not in the first 30 s of a race
min_speed = 20.0
boost = 0.06               # +6% top speed while open
accel = 2.0                # +2 m/s^2 pull while open, on the throttle
grip_loss = 0.4            # share of the downforce grip lost while open

[ers]                      # NEW (§5.5)
capacity_s = 4.0           # battery holds 4 s of full deployment
start_charge = 0.5         # share of capacity at a grid start (rolling/random spawns: U(0.2, 1.0))
power_boost = 0.25         # engine power x 1.25 while deploying (~+7.7% top speed)
harvest_rate = 0.5         # s of boost gained per s of full braking at top speed (scales with brake x v / v_max)

[race]
overtake_margin = 2.0      # an order change counts only once the gap is > 2 m (hysteresis)
pass_hold_s = 1.0          # a pass pays once held 1 s with both cars racing
start_phase_s = 30.0
start_overtake_mult = 1.5  # passes in the first 30 s of a grid start pay 1.5x
gain_range = 150.0
lap_sectors = 10
sector_pace_window = 0.05
close_call_ttc = 0.4
close_call_min_closing = 1.0

[rewards]                  # middle's values; profiles override any of these
progress_weight = 1.0      # x metres of progress x 1 / (v_max * control_dt): top speed for a step = +1
wall_crash = -70.0         # crash into a wall, or no progress (terminal)
car_crash = -70.0          # crash into another car (terminal)
per_place = 5.0            # per race point gained/lost, paid as it changes (§6.3)
overtake = 20.0
overtaken = 20.0           # must be >= overtake (otherwise two cars farm points by swapping)
gain_per_m = 0.1
car_close_call = 0.15
wall_close_call = 0.15
steer_change = 0.05        # x |change of the steering output| per step
fastest_sector_lap = 20.0  # per lap: purple sectors pay this / lap_sectors each
sector_pace = 3.0
sector_record = 5.0

[eval_bonus]               # added to place points in evals with opponents; profiles override
pass_bonus = 1.0           # per pass on an opponent's car, finishing cars
crashed_pass_bonus = 0.0   # per pass on an opponent's car, cars that crashed later
passed_penalty = 0.0       # per time passed by an opponent's car, finishing cars
fastest_lap_bonus = 3.0    # the race's fastest lap, by a finishing car of the trained brain
pace_bonus = 1.0           # per race: its best lap beats the opponents' best lap

[td3]
hidden = 384               # 512 halves updates/s on the laptop GPU
gamma = 0.99               # ~5 s horizon at 20 Hz; longer makes every style MORE careful (§12)
n_step = 3
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2
batch = 256
lr_actor = 3e-4
lr_critic = 3e-4           # --lr 1e-4 to fine-tune a trained driver
replay_size = 5000000      # on the GPU, ~2.7 GB at 127 inputs
updates_per_transition = 0.25
allow_tf32 = true
warmup_steps = 20000                # fresh run: random actions first
resume_warmup_steps = 1000000       # resume: refill with the loaded driver
resume_critic_only_updates = 150000 # then train only the critic
actor_lr_ramp_updates = 100000      # then ramp the actor's lr 0 -> full
exploration_sigma = [0.03, 0.4]     # per car per episode, log-uniform
exploration_correlation = 0.8       # AR(1) between steps

[workers]
count = 6                  # max(1, min(6, cpu_count - 4)) if left out
transition_chunk = 64
policy_refresh_steps = 100
publish_every_updates = 250
self_share = 0.3333        # with --opponents: the trained brain's share of each grid (>= 1 car)
default_cars = [3]

[eval]
maps = ["Spielberg", "Monza", "Silverstone", "Spa", "Zandvoort", "Catalunya"]
grids = 3                  # with opponents: races per map, the brain's cars on different slots
every_publishes = 40       # a round starts after this many new policy versions (or right after the last)

[run]
total_steps = 3000000
save_every_updates = 50000
keep_last = 5              # periodic checkpoints a run keeps (its own only)
save_dir = "saved_models"
default_profile = "middle"
resume_tag = "best"        # what a bare --resume loads
log_every_s = 60.0
dashboard = true
dashboard_refresh_s = 2.0
write_logs = true

[watch]
tag = "best"
map = "Spielberg"
profiles = ["middle"]
cars = 3
episodes = 5
random_spawn = false
playback_speed = 3.0

[colors]                   # per profile, window + console; never red (crashed cars turn red)
middle = "#377eb8"
aggressive = "#ff7f00"
defensive = "#4daf4a"
```

### 4.2 `configs/profiles/*.toml`

A profile is one file. It holds a description plus overrides of `[rewards]` and `[eval_bonus]`,
and nothing else. Adding a style means adding a file.

`config.py` must:
- reject unknown keys,
- reject `overtaken < overtake`,
- expose `profile(name)` returning the merged rewards + eval bonuses.

```toml
# middle.toml
description = "The baseline racer: passes when the risk is small. Trained first, from scratch; the other styles start from it."
[rewards]
[eval_bonus]
```
```toml
# aggressive.toml
description = "Bold in traffic, disciplined alone: bigger passes, car contact cheap, walls cost what they cost middle."
[rewards]
overtake = 35.0
overtaken = 35.0
car_crash = -30.0
car_close_call = 0.0
steer_change = 0.0
fastest_sector_lap = 0.0
sector_record = 0.0
[eval_bonus]
pass_bonus = 3.0
crashed_pass_bonus = 3.0
```
```toml
# defensive.toml
description = "Few risks to gain places, hates losing them: being passed costs 4x what a pass earns."
[rewards]
overtake = 6.0
overtaken = 24.0
gain_per_m = 0.05
wall_crash = -200.0
car_crash = -200.0
car_close_call = 0.25
wall_close_call = 0.25
fastest_sector_lap = 10.0
steer_change = 0.1
sector_pace = 1.5
sector_record = 2.5
[eval_bonus]
pass_bonus = 0.0
passed_penalty = 1.0
fastest_lap_bonus = 0.0
```

**Risk levels these give.** A pass is worth trying up to roughly
(0.8 × overtake + per_place) / (|crash| + per_place × points held + ~45).
- The 0.8 is the network's discount of a pass paid 1 s later.
- The 45 is the progress a crash throws away within the ~5 s horizon.

Running P11 of 22 (12 points), against a crash into a car:

| Profile | Pass worth trying up to |
|---|---|
| aggressive | ~24% (P3 ~19%, P20 ~37%) |
| middle | ~12% |
| defensive | ~3% |

Front-runners are more careful than backmarkers by design.

---

## 5. Simulation

### 5.1 Car and physics

- **Physics model:** the gym's kinematic single-track model (`DynamicModel.KS`), with the `[car]`
  parameters. The gym's dynamic model has linear tyres that never saturate (7 g sideways from a
  1.7° input at 80 m/s) and spun under braking: don't use it.
- **Grip limit** (stands in for downforce): the steering angle is capped each step at
  `min(s_max, atan(wheelbase * a_lat / max(v, 1)^2))`, where
  `a_lat = 9.81 * grip_factor * min(max_g, base_g + downforce * downforce_g * (v/ref_speed)^2)`.
  - `grip_factor` is the episode's grip variation.
  - `downforce = 1 - drs.grip_loss` while DRS is open, else 1.
  - That gives 1.8 g in slow corners rising to 5 g in fast ones: fast corners go flat out, slow ones
    need braking.
- **Smooth steering:** the gym steers bang-bang (full rate whenever the angle isn't exactly on
  target, overshooting every step). Replace its steering function with one that holds the rate
  needed to reach the target by the end of the control step:
  `clip((target - delta) / control_dt, sv_min, sv_max)`.
- **Longitudinal model:** the gym runs in `ACCL` longitudinal mode, so the env computes each car's
  acceleration itself.
  - `cap` = the car's top speed = `v_max x pace factor x opponent speed factor`.
  - `power = (cap / v_max)^3`. A slower car has less engine power, because top speed ~ power^(1/3)
    under v² drag.
  - With ERS deploying, power is multiplied by `1 + ers.power_boost`.
  - `engine = min(a_max, power * a_max * v_switch / max(v, 0.1))`: traction-limited, then
    power-limited.
  - `DRAG = a_max * v_switch / v_max^3`, so full throttle tops out exactly at v_max.
  - `drag_scale = 1 / ((1 + tow_boost) * (1 + drs.boost * drs_open))^3`.
  - `accel = throttle * engine - brake * a_max - DRAG * drag_scale * v^2`.
  - **Tow/DRS pull:** `+ (tow.accel * tow_boost / tow.boost + drs.accel * drs_open)`, faded linearly
    to 0 over the last 5 m/s below the boosted top speed `cap * (1 + tow_boost) * (1 + drs.boost * open)`,
    and only while throttle > 0.
  - Cars out of the race brake to a stop.
- **No reversing:** after each step, clamp any negative speed to 0. An integrator substep can
  overshoot slightly below 0 and creep backwards.
- **Physics speed ceiling:** the gym's `v_max` is set to
  `v_max * (1 + tow.boost) * (1 + drs.boost) * (1 + ers top-speed gain) * 1.05`. The real top
  speed comes from drag.
- **Yaw rate for the network:** `v * tan(delta) / wheelbase`. The KS model reports 0.
- **Measured with the current car:** 0–100 km/h 1.45 s, 0–200 3.4 s; lifting from 86 → 72 m/s
  takes 2 s; braking from 86 → 20 m/s takes 175 m.

### 5.2 Actions (NEW)

The actor outputs 5 tanh values in [-1, 1]. The env maps them to commands:

| Output | Range | Command |
|---|---|---|
| 0 steer | [-1, 1] | target angle = steer × s_max, capped by the grip limit |
| 1 throttle | [-1, 1] | > 0: that share of the engine's available acceleration; ≤ 0: off (coasting, only drag slows the car) |
| 2 brake | [0, 1] | `max(output, 0)` × a_max of braking. Both pedals may be pressed: the net acceleration is their difference. |
| 3 DRS | 1 or 0 | 1 when output > 0: request to open. Only opens where available (§5.4). |
| 4 ERS | 1 or 0 | 1 when output > 0: deploy the battery (§5.5) |

- **Why the dead zones:** throttle ≤ 0 and brake ≤ 0 map to "off", so a tanh output can let go of
  a pedal without having to saturate at −1.
- **Grid starts:** for the first `launch_s` the env forces throttle = 1 and brake = 0 for every car.
  `step()` returns the actions actually used (`info["actions"]`), and those are what get stored for
  training.

### 5.3 Tow (slipstream) — keep as is

- **Who gets a tow:** a racing car above `tow.min_speed`, with another car ahead along its own
  heading.
  - Full effect within 15 m ahead and ±1.5 m sideways.
  - Fading linearly to nothing at 120 m / ±3 m.
  - The strongest tow counts.
- **What it gives:** `tow_boost = tow.boost × fade` cuts drag (up to +10% top speed, ~340 km/h)
  and adds up to 10 m/s² pull (§5.1).
- **When it ends:** leaving the wake brings the drag back, so the extra speed bleeds off naturally.
  There's no special braking rule.
- **Why these numbers:** the strong pull is what keeps the pack together. A weak tow (6%, 1 m/s²,
  40 m) left cars ~50 m apart, and +60% top speed sent cars past 370 km/h.

### 5.4 DRS (CHANGED: straights only)

- **DRS zones:** precomputed per track from the centerline. A point at arc length `s` is in a zone
  if the track heading stays within `zone_max_turn_deg` of its heading at `s`, all the way to
  `s + zone_lookahead_m`.
  - Keep zones at least `min_zone_length_m` long; drop shorter ones.
  - Store a boolean mask at 2 m resolution plus, for each point, the distance to the zone's end.
  - Because the lookahead is about one braking distance, every zone ends before the braking zone
    of the next corner.
- **Available** when all of these hold:
  - the car is in a zone,
  - race time ≥ `from_s`,
  - speed ≥ `min_speed`,
  - and (if `gap_s > 0`) the car directly ahead in the race order is within `gap_s × max(v, 1)`
    metres.
- **Open** = the DRS output is 1, AND it's available, AND brake = 0. It closes the moment the car
  brakes or leaves the zone.
- **Effect while open:** +6% top speed (less drag), +2 m/s² pull on the throttle, and −40% of the
  downforce part of the grip limit.
- **Why:** the old rule (anywhere within 1 s, closing only on braking) sent cars into Monza's first
  chicane at ~91 m/s on lap 2 (tow + DRS) and wiped out nearly the whole field: 49 of 52 crashes
  there were walls, and 63% of those cars had DRS open in their last second. Zones that end before
  corners fix this at the source.
- **Observation:** `drs_available`, `drs_open`, and the metres left in the current zone / 500
  (0 outside a zone).

### 5.5 ERS (NEW)

Every car carries a battery of `capacity_s` seconds of full deployment.

- **Start charge:** `start_charge × capacity` on grid starts, `U(0.2, 1.0) × capacity` on rolling
  and random spawns, so every charge level gets practised.
- **Deploy** when ERS = 1, throttle > 0 and the battery > 0.
  - Engine power is multiplied by `1 + power_boost`. This only matters above `v_switch`, where the
    engine is power-limited, and it raises top speed by ~7.7%.
  - The battery drains `throttle × control_dt` per step.
- **Harvest under braking:** `battery += harvest_rate × brake × (v / v_max) × control_dt`, capped
  at capacity. Energy recovered grows with speed and brake force. Coasting and engine braking
  don't harvest.
- **Observation:** battery / capacity, and deploying (0/1).
- **Stats:** the share of racing time deploying, and the mean battery level per car-episode.
- There's no reward for ERS or DRS use. They pay off through progress and passes.

### 5.6 Tracks and progress

- **Maps:** every `*_map.yaml` under `maps_dir` that has a matching `*_centerline.csv`. Folder and
  file names don't always match (`Mexico City/MexicoCity_map.yaml`). Fall back to the gym's
  built-in Spielberg if the folder is missing.
- **Gym env construction:**
  - `EnvConfig` with the vehicle params (and the safety `v_max` from §5.1), the map scaled by
    `map_scale`, `num_agents`, `seed`, render on/off.
  - Lidar: `lidar_beams` over the FOV, range 250 m, noise 0.01.
  - `LongitudinalActionType.ACCL`, timestep 0.05 / integrator 0.01, `DynamicModel.KS`.
  - `compute_frenet_frame=False`: the gym's Frenet tracking gets stuck at s ≈ 1.8 m on scaled maps.
  - `LoopCounterMode.TOGGLE`, `max_laps=None`.
  - Then `gym.make("f1tenth_gym:f1tenth-v0", config=cfg, render_mode="human" if render else None)`.
- **Shared lidar distance map:** the gym computes a 32 MB distance map per car per track. Patch
  `laser_models.get_dt` with a cache keyed by bitmap identity, keeping at most 4 tracks per
  process. Without it, a 22-car track used ~1 GB.
- **CenterlineTracker:** progress along the centerline polyline (closed loop, duplicate end point
  removed).
  - `reset_agent`: global projection at spawn.
  - `update(cars, xy)`: batched local search, all cars at once, in a window of segments −5..+20
    around each car's last segment.
  - `delta(s_new, s_old)`: signed progress, unwrapped across the start line.
  - `yaw_at(s)`, `pose_at(s)`.
  - `_bend(depth, ahead)`: the largest heading change within a window, on a 2 m grid, cached.
    Used by `grid_anchor(depth)`, `straight_spots(depth, ahead, max_turn)` and the DRS zones.
- **Lap timing:** a lap runs from one forward line crossing to the next, with the crossing time
  interpolated inside the step. Crossing backwards invalidates the lap in progress.
- **Sectors:** the lap is split into `lap_sectors` equal sectors, timed the same way (interpolated).
  - A sector partly driven backwards isn't timed.
  - The first time through a sector only sets its reference.

### 5.7 Spawns and starts

- **Grid** (evals, watching, and part of training):
  - Pole sits at `grid_anchor((n-1) × slot_gap)`: the point nearest the start line (either way)
    where the track stays within 10° from the last grid slot to 60 m ahead of pole. Fall back to
    the straightest spot. Without this, 7 tracks put the back of the grid in a corner.
  - Car i sits `i × 12 m` behind pole, in 2 columns ±3 m from the centerline, standing still.
  - Race offsets = −i × 12.
  - 2 s full-throttle launch.
- **Rolling** (35% of training episodes with 2+ cars):
  - The pack is dropped on a random straight spot with 150 m clear ahead, gaps U(8, 16) m, random
    order.
  - Speed U(35, 60) m/s per episode, ±5% per car; 2 columns with ±0.5 m jitter.
  - It exists so that every car starts its race with someone right in front: random spawns spread
    cars a lap/n apart, and grids spread into a procession.
- **Random:** cars evenly spread around the lap from a random base point, ±1 m lateral, ±8°
  heading, speed U(0, 15) m/s. Race offsets = i × spacing.
- **Training spawn choice:** rolling with p = 0.35 (2+ cars). Otherwise grid with p = 0.5, or 0.8
  for 12+ cars (the 22-car pack braking into the first corner only exists on a grid). Otherwise
  random.

### 5.8 What the car sees (keep; observation layout)

Every input is normalized to roughly [-1, 1], with `nan_to_num` at the end. The observation is
built for all cars at once (one batched function).

| Block | Size | Content |
|---|---|---|
| lidar distance | 50 | `sqrt(clip(d / 250, 0, 1))` for the beams picked by `obs_bands`: every beam within ±25°, every 2nd to ±60°, every 4th to ±135°. sqrt keeps close range readable. |
| lidar closing speed | 50 | `clip((prev_d - d) / (control_dt * v_max), -2, 2)` per beam: the relative speed of whatever it hits |
| car state | 7 | speed / v_max, steering angle / s_max, `clip(yaw_rate / 2, ±2)`, `clip(slip angle, ±1)`, previous steer, throttle, brake outputs |
| race | 2 | `(racing cars behind - racing cars ahead) / (n - 1)` (0 alone); `min(other cars, 22) / 22` |
| DRS | 3 | available, open, metres left in the zone / 500 |
| ERS | 2 | battery / capacity, deploying |
| mirrors | 3 | nearest racing car behind within 60 m and ±10 m sideways: distance / 60, closing speed / 20 (clip ±1), side / 5 (clip ±1, + = left); (1, 0, 0) if none |
| track preview | 6 | track heading at s + 25/50/100/150/200/300 m minus the car's heading, wrapped, / π: "knows the corners coming", also when cars block the lidar |
| alongside | 4 | per side (left, right): the nearest racing car whose body overlaps front to back (`abs(along) < length`) with a sideways gap < 6 m: gap / 6, lateral closing speed / 10 (clip ±1); (1, 0) if none |

Total: **127** inputs.
- New inputs must always be appended at the end.
- Checkpoints store a layout tag and are refused if it doesn't match.
- Old checkpoints are never upgraded.

**Relative geometry:**
- One helper `relative(xy, yaw) -> (along, side)` gives, for every pair, where car j is in car i's
  frame. Mirrors, tow and alongside all use it.
- Poses are the rear axle. Car bodies are `collision_body_center_x` ahead of it.
- Body-to-body gaps: each car is 3 circles of radius width/2 along its length, and the gap is the
  minimum circle distance minus 2r.

### 5.9 Gym patches, crashes and speed (keep)

Install these as bound methods on the gym's simulator. Remove them again in `close()`, because
they reference the env and create reference loops.

- **Racing-only scan update** (replaces `F110Simulator._update_scans`). It only handles cars still
  racing. Crashed cars are taken off the track: moved to (1e5, 1e5) with a zero state, and never
  scanned or ray-cast against. Parked wrecks turned big-grid starts into chain-reaction pile-ups:
  0% of a 22-car grid survived. For each racing car:
  1. Clean scan (walls only) → the gym's TTC check on it. If it fires, it's a **wall crash**: zero
     the speed state, set collision = 1, and set `wall_hit = True`.
  2. Keep a walls-only copy of the scan, taken before ray casting, because `ray_cast` edits its
     array in place. It's used for traffic detection and wall close calls.
  3. **Ray-cast cull:** only ray-cast opponents a beam could hit.
     - Skip one whose bounding circle (radius = half the body diagonal, centred on its collision
       pose) is farther than the clean scan's longest beam + radius + 1 m.
     - Skip one that lies entirely outside the FOV:
       `|bearing| − asin(R/d) > half_fov + 2°` with `d > R + 0.5`.
     - The gym's beams only hit forwards (`d1 ≥ 0`), but it loops over all 108 beams × 4 edges for
       a car behind. That was 11 ms of a 16 ms step on a 22-car grid. The results are identical.
  4. Add noise from the car's scan rng, clip, and store the scan as float32.
- **Car crashes:** after the scan update, the gym's `_update_agent_collisions` (LIDAR_SCAN mode)
  runs the TTC check on the scans with the cars in, for cars without a wall crash. A collision
  without `wall_hit` is a **car crash** (reason `contact`).
- **Racing-only physics step** (replaces `F110Simulator.step`): the same as the gym's step, but it
  only integrates racing cars. Cleared cars stand still, so integrating them (~85 µs per car-step)
  changed nothing. Then it calls `_update_scans`, `_update_agent_collisions`, and advances
  `sim_time`.
- **Skip the gym env's step:** call `sim.step` directly, except when rendering. The gym env's step
  only builds per-car observation dicts, which nobody reads.
- **No progress:** a car whose progress hasn't exceeded its best + 0.5 m for 6 s is out with the
  wall-crash penalty (reason `no_progress`). The timer pauses while another car blocks its path
  within 30 m. "In its path" means beams within ±(width/2 + 0.5 m) of its heading that hit a car,
  i.e. are more than 0.5 m shorter than the walls-only scan.
- **Simulation errors:** a non-finite state after a step makes every active car crash (reason
  `sim_error`), and the episode ends.
- **Early end:** the episode ends as soon as every car is out. It doesn't wait for the timer.

### 5.10 Race order, passes, positions

- **Race distance** = start offset + progress driven.
- **Position score** = race distance on grid and rolling starts; progress driven on random spawns,
  where every car starts level.
- **Two order matrices**, both updated with hysteresis: a pair flips only once the car behind is
  ahead by more than `overtake_margin` (2 m), so cars side by side don't flicker.
  - `ahead[i, j]`, by race distance. Passes are counted on it. It's updated for every pair, parked
    cars too, so driving past a wreck never counts later.
  - `place_ahead[i, j]`, by position score. Race points are counted on it.
- **Passes** (between cars both still racing):
  - A pass becomes pending and pays only once held for 1 s (`pass_hold_s`) with both cars racing.
  - It's cancelled if either car crashes, or if the place is taken straight back. A retake doesn't
    count as a pass of its own, so a defender can't let a car by and re-pass it for points.
  - Pending passes are paid at the end of the race.
  - Passes in the first 30 s of a grid start pay ×1.5.
  - Paid: +overtake to the passer and −overtaken to the passed car, from each car's own profile.
- **Finishing positions:** when the timer ends, the racing cars are ranked by position score.
- **Race points held by a car** = n − (racing cars ahead of it in `place_ahead`), or 0 once it's
  out. These equal the eval's place points.

### 5.11 Rewards (per step, each car by its own profile's weights)

| Term | Formula |
|---|---|
| progress | `ds × progress_weight / (v_max × control_dt)` |
| steering | `−steer_change × |Δ steer output|` |
| crash (terminal) | `wall_crash` (wall or no progress) or `car_crash` (contact) |
| race points | `per_place × (points now − points last step)`: a pass +1 place, being passed −1, a car ahead dropping out +1, a crash loses everything held. Over a race it sums to (finish − start places) × per_place, so P22 → P10 pays 6× P3 → P1, the grid slot is never paid, and swapping can't be farmed. Paid the moment the order changes, so the ~5 s horizon doesn't hide it. |
| passes | ±overtake / overtaken (§5.10) |
| gain | `gain_per_m ×` metres closed on the nearest racing car ahead on the centerline within 150 m, if it's the same car as last step (metres lost cost the same; dropping back and catching up again nets 0; the car ahead isn't charged) |
| close calls | `−max(wall_close_call × wall_level, car_close_call × car_level)` (below) |
| purple sector | `+fastest_sector_lap / lap_sectors` for the race's fastest time through a sector so far (not the first time through) |
| lap pace | `+sector_pace × clip(1 − (t / record − 1) / 0.05, 0, 1)` for a sector within 5% of the track's fastest time through it, plus `sector_record` for beating it (which sets the new time). The records are kept per track per worker for the whole run, shared by its training envs; evals keep their own. |

**Close-call levels:**
- **Cars:** body gap per pair. Closing speed = Δgap / dt, counted only if > 1 m/s (and finite).
  TTC = gap / closing. Level = `clip(1 − TTC / 0.4, 0, 1)`, the maximum over the other cars. Both
  cars of a pair pay, since either can make room.
- **Walls:** the nearest-wall gap = min(walls-only scan − the car's own outline distances), and its
  shrinking rate. Use the nearest wall only: TTC per beam fired on 47–85% of clean steps at 80 m/s.

**What not to add:** no finish bonus, and no end-of-race position bonus. Both were paid when the
150 s timer ran out, the network has no clock, and they only added a flat ~+4 to every state's
value.

### 5.12 Episode stats (from workers to the learner)

Per car:
- returns, progress, laps, termination reason (`timeout`, `collision`, `contact`, `no_progress`,
  `sim_error`), finishing position,
- average and top speed, passes made / suffered, passes on and by opponents' cars,
- tow fraction, close-call steps, purple sectors, best lap, lap times,
- DRS-open fraction, ERS deploy fraction and mean battery.

Per race:
- map, car count, which cars are the learner's, the driver of each car,
- duels: for each opponent, how often the learner's cars finished ahead ("ahead" = survived when
  it didn't, or the same outcome and further up the order),
- the learner's car-steps.

---

## 6. Learning (TD3) — `agent.py`, `policy.py`

### 6.1 Networks and the update block

- **Actor:** Linear(127, 384) → ReLU → Linear(384, 384) → ReLU → Linear(384, 5) → tanh.
- **Twin critic:** two copies of Linear(132, 384) → ReLU → Linear(384, 384) → ReLU → Linear(384, 1),
  in one module.
- **TD3 rules:**
  - Target policy smoothing: noise 0.2, clipped to 0.5.
  - The target is the min of the two target critics.
  - Critic loss: MSE(q1) + MSE(q2).
  - Actor loss: −q1 (with the critic's gradients frozen).
  - Policy delay 2; soft updates with τ = 0.005 (`torch._foreach_lerp_`).
- **Optimizers:**
  - Adam with `capturable=True` on CUDA.
  - Learning rates are GPU tensors, changed with `fill_`, so a captured CUDA graph sees new values.
  - `zero_grad(set_to_none=True)`.
- **One update block** = `policy_freq` critic steps + 1 actor step + soft updates. Record the
  diagnostics (critic loss, actor loss, mean Q) on the last critic step only, as GPU tensors;
  convert them with `.tolist()` only when shown.
- **CUDA graphs:** capture two blocks, "full" and "critic-only" (no actor step, critic soft update
  only), each after 3 warm-up blocks on a side stream; then replay them. Without graphs the GPU
  sat below 10% (Python overhead). Fall back to eager mode if capture fails.
- **TF32** on.

### 6.2 Replay buffer

- Lives on the GPU as two tensors: `obs` (2, capacity, 127) in float16 (state, next state), and
  `rest` (capacity, 5 + 2) in float32 (action, reward, discount).
  - A sampled batch is two gathers.
  - Sampling: `idx = (rand(batch) * size_t).long()`, where `size_t` is a device-side copy of the
    size so it works inside the graph.
- **Inserts:** stack each chunk on the host and copy it once per tensor (a pinned buffer with
  `non_blocking=True` is fine).
- **n-step transitions (n = 3):** built in the workers as (s, a, R, s', discount).
  - discount = γ^k, or 0 when the episode ended in a crash.
  - A timeout bootstraps.
  - The learner never needs γ.
- **The buffer is never saved.** That's a user requirement: only the model is stored.

### 6.3 Exploration and warm-up

- **Exploration noise:** each car gets its own σ per episode, log-uniform in [0.03, 0.4], applied
  as AR(1) noise with ρ = 0.8: `x = ρx + sqrt(1−ρ²) N(0, 1)`. A push then lasts ~0.25–0.5 s, long
  enough to try a move; independent jitter averaged out. Careful and adventurous drivers share
  every grid.
- **Fresh run:** 20k uniformly random actions before learning starts.
- **Resume:**
  1. Load the checkpoint, then reset the target actor to the actor. A `best` checkpoint's
     networks are from different moments (§7.4).
  2. Refill 1M steps with the loaded driver.
  3. Train only the critic for 150k updates. The actor stays frozen, so the critic adapts to the
     new data and rewards.
  4. Ramp the actor's lr from 0 to full over 100k updates, in 1% steps.

  The ramp exists because every resume without it collapsed (eval rounds fell from 13–17 to 1–5
  pts/car).
- **Replay ratio:** 0.25 updates per collected transition. The learner only pulls more data once
  it has done its updates, and the queue is bounded, so a slow learner makes the workers wait
  instead of silently changing the ratio.

### 6.4 Checkpoints

- **Format:** `{"model": {actor, critic, target_actor, target_critic, actor_opt, critic_opt,
  total_it}, "meta": {...}, "obs_dim", "action_dim", "layout": "racer-v1-127obs-5act"}`.
- **Loading:** refuse a checkpoint with another layout. Load each optimizer state only if present.
- **One loader** handles the learner, the opponents, the watcher and compare. It returns
  (data, model).
- **`read_actor_weights(path)`** returns the actor as float32 numpy arrays in layer order, for the
  torch-free `FrozenPolicy`. It checks the shapes.

### 6.5 Torch-free policy (`policy.py`)

- **Layer shapes:** `[(384, 127), (384,), (384, 384), (384,), (5, 384), (5,)]`. The forward pass is
  a numpy MLP: ReLU, ReLU, tanh.
- **SharedPolicy:**
  - The learner flattens the actor into a float32 `RawArray` under a lock and bumps a version
    counter every 250 updates.
  - Workers keep a private copy and re-copy only when the version changed. They check every 100
    steps and at every episode start: short episodes never refreshed otherwise.
- **FrozenPolicy:** fixed weights for opponents, from `read_actor_weights`.

---

## 7. Parallel training system — `workers.py`, `train.py`

### 7.1 Learner process

1. **Setup:** parse the CLI and resolve the settings.
   - Set `OPENBLAS_NUM_THREADS`/`OMP`/`MKL = 1` before spawning workers: numpy's BLAS threads
     fighting over cores made the workers ~500× slower.
   - Start the workers with the `spawn` context and pass the settings as arguments.
2. **Loop:** if updates are owed (budget ≥ 1 and the buffer ≥ batch), run up to 64 updates in a
   row, publishing every 250. Otherwise take a transition chunk from the queue (0.5 s timeout) and
   add `len × 0.25` to the budget.
3. **Stats:** drain the stats queue every loop.
   - Training episodes feed a rolling window of the last 200 learner car-episodes (dicts), the
     last 100 duel results, and car-steps per car count.
   - Eval rounds are handled as in §7.3.
4. **Checkpoints:** every 50k updates, save `td3_<updates>.pt` and copy it to `td3_latest.pt`.
   Delete the run's own periodic files beyond the last 5; never touch earlier runs' files, the
   best, or latest.
5. **Logging:** every `log_every_s`, write a metrics.csv row and refresh the status (dashboard
   every 2 s).
6. **Shutdown (Ctrl+C or end):**
   - Save latest.
   - Set the stop event.
   - Drain the queue for up to 5 s so blocked workers can exit.
   - Terminate the rest.
   - Print a summary.

   Workers ignore SIGINT; only the learner reacts.

### 7.2 Worker process (torch-free)

- **Its track:** one training env.
  - The worker changes map and car count after ~100k car-steps of the learner's data (±25%,
    random): a new random map other than the current one, and a car count from the `--cars` list.
  - Counting data instead of episodes gives every car count a similar share. A 22-car episode
    yields ~10× the data of a 2-car one.
  - After closing an env, call `gc.collect()`. The gym env has reference loops: without it,
    workers grew to ~2 GB and Linux ran out of memory.
- **Drivers:** each episode, `assign_drivers(n, rng, opponents, self_share)` gives the trained
  brain `max(1, round(self_share × n))` cars on random slots. The other cars go to the opponents in
  turn, in a shuffled order. A profile listed twice gets twice the cars.
- **Opponents in training:** they drive greedily, with their own profile's rewards (unused), on a
  car slowed to `opponent_speed`. Only the trained brain's cars produce transitions.
- **Actions:**
  - Warm-up: uniform random actions.
  - Otherwise: the policy plus exploration noise, clipped to [-1, 1].
  - Opponents: their FrozenPolicy.
  - Don't run the network for cars already out (that keeps the noise draws the same).
- **Transitions:** sent in chunks of 64 as numpy arrays, with a blocking put (backpressure).
- **Evaluator:** worker 0 also runs eval rounds (§7.3). On a resume it runs one immediately, as the
  baseline, before any training changes the driver.

### 7.3 Eval rounds

- **When:** every time the learner has published 40 new policy versions since the last round, or
  right after the previous round if rounds take longer.
- **The races:** for each eval map, keep one env with the most cars of the `--cars` list (built on
  first use).
  - With opponents, race each map `grids` = 3 times. The driver-to-slot assignment for map k,
    grid g uses `default_rng([k, g])`, fixed from round to round.
  - With one grid (the same slots on every map), the slot luck decided the score: the same driver
    scored 16.7 or 11.9 pts/car depending on the split, and rounds within a run spread by 1.8–2.9.
  - Without opponents every car is the same driver, so one grid is enough.
  - Greedy driving, the normal car, opponents at full speed.
  - Send the exact evaluated weights with the results.
- **Score per race:**
  - Without opponents: the mean distance of the cars. Not the median: once half the grid survives,
    the median only measures speed.
  - With opponents: place points per car of the trained brain: n − position + 1 for finishers
    (22 for P1 on 22 cars), 0 for a crash.
- **Eval bonuses** (from the profile) on top of the place points:
  - `pass_bonus` per pass on an opponent's car, for finishers.
  - `crashed_pass_bonus` per such pass, for cars that crashed later.
  - `−passed_penalty` per time passed by an opponent's car, for finishers.
  - A finisher never drops below 1 point.
  - `fastest_lap_bonus` to the brain's finishing car that sets the race's fastest lap.
  - `pace_bonus` if the brain's best lap beats the opponents' best.
  - Also return a breakdown of what the bonuses counted; the dashboard shows it.
- **Round score:** the average over maps of each map's average over its grids. Best is picked by
  points with opponents, by distance without (`--best-by` can force one).
- **Run signature:** settings that make scores comparable.
  - obs dim, layout, hidden, top speed, lidar range, episode length, `--cars`, profile, map scale,
    eval maps, score kind + eval bonuses, grid settings, launch, tow, DRS, ERS.
  - With opponents also: their identities (name@update count), the self share, the eval grids.
  - A stored best only counts if the signature matches.
- **Saving a best:**
  - The baseline round of a resume raises the bar. It's saved only if the profile has no best yet,
    or if it beats a stored best with the same signature.
  - Otherwise, every round that beats the bar is saved as `td3_best.pt`, plus a copy as
    `td3_best_<label>cars.pt` (label `6` or `mix4-8-12-16-22`). The copy keeps a car setup's best
    when a run with another `--cars` list resets the bar.
  - `best_meta.json` holds the score, the per-map scores, distance, updates, env steps and the
    signature.

### 7.4 What a best checkpoint stores

- The **evaluated** actor (from the worker), not the learner's current one.
- No actor optimizer state, because it belongs to the learner's newer actor.
- The critic and target critic are the learner's, from ~one eval round later.
- That's why loading always resets the target actor to the actor. Before that fix, a resume's
  critic-only phase learned the value of a different (newer) driver: the actor and target actor
  were 5.9% apart, against 0.1% in a normal checkpoint.

---

## 8. Console, logs, records — `console.py`, `records.py`

- **Startup header:**
  - run (steps, fresh or resumed from what), profile and folder,
  - maps (rotation, scale, variation), cars (workers, car counts, grid/rolling shares, launch),
  - rivals (opponents, self share, opponent speed), eval (maps, grids, cadence, best criterion,
    profile bonuses),
  - device, car, network layout, learning settings (lr, replay, batch, ratio, n-step, γ and its
    horizon), noise.
- **Live dashboard** (rich, when stdout is a TTY and rich is installed; otherwise printed panels):
  - **Run line:** progress bar, steps/s, elapsed, ETA. Shows the warm-up phase (refill /
    critic-only / lr ramp) and the learner line (updates/s, replay fill, episodes, Q, critic loss).
  - **Training table**, from the last 200 learner car-episodes, each value as start / best / now:
    - distances: median, mean,
    - endings: survived, crashed, into a car, stuck,
    - speed: average, top,
    - racing (multi-car only): passes per car, close calls per car, fastest sectors per car, in a
      tow, DRS open, ERS deploying,
    - return.
  - **Eval table**, with the first / best / last round: score and per-map scores, finished,
    passes per race, passes on / by opponents, fastest laps, faster than opponents, mean distance.
  - **Info rows:** rivals' win shares, traffic share per car count, maps in use now, lap records,
    RAM (warn below 1.5 GB free), last save.
  - **Recent events:** the last 6 eval rounds, new bests, lap records and warnings.
- **Printed fallback:**
  - A status panel every 60 s with the same content.
  - One summary line per eval race: points, mean and median distance, survived, overtakes, speeds,
    rivals. Then every car sorted by place, crashed cars last with "crash (wall)" / "crash (car)"
    / "stuck"; the brain's cars in bold, opponents dimmed. Wrapped at 100 columns.
  - One line per round.
  - Colours only on a TTY and without `NO_COLOR`.
- **One summary function** turns the window of car-episode dicts into every number shown or
  logged. The dashboard, the panel and metrics.csv must not each recompute them.
- **metrics.csv:** one row per `log_every_s`, appended across runs (a `run` column tells runs
  apart):
  - progress: env steps, updates, rates,
  - driving: median / mean / best distance, lap rate, survival / crash / car-crash / stuck rate,
    average / top speed,
  - racing: passes, tow, close calls, purple sectors, DRS open, ERS deploying,
  - learner: critic loss, actor loss, avg Q, RAM,
  - rivals and best: win rate vs opponents (overall + per opponent), best eval and its kind.
- **evals.csv:** one row per map per round, its grids averaged:
  - map, cars, grids, learner cars, opponents,
  - median / mean distance, survived and overtakes per race, average speed, best lap,
  - points per car, round score, new best, win rate.
- **lap_records.json:** the fastest complete lap per map (lap time, date, source train/eval, cars,
  updates), written whenever it improves. Give `records.py` a loader per profile.

---

## 9. Entry points

### 9.1 `train.py`

| Option | Default | Meaning |
|---|---|---|
| `--profile` | `middle` | Style to train; checkpoints in `saved_models/<profile>/`. |
| `--cars` | `[workers] default_cars` | One count or a list such as `4,8,12,16,22` (every worker redraws from it at each map switch; evals use the largest). |
| `--workers` | 6 | Rollout processes. |
| `--steps` | 3M | Stop after this many collected env steps. |
| `--resume [TAG]` | off | `best`, `latest`, an update count, or `PROFILE/TAG` to start a style from another profile's driver. A bare `--resume` loads `best`. |
| `--opponents` | none | `PROFILE[:TAG],...` frozen opponents; the tag defaults to best; list a profile twice for twice the cars. |
| `--lr` | 3e-4 | Actor and critic lr (1e-4 to fine-tune). |
| `--best-by` | auto | `distance` or `points` (auto: points with opponents). |
| `--opponent-speed` | 0.95 | Opponents' top speed in training. |
| `--self-share` | 1/3 | The trained brain's share of each grid. |

### 9.2 `watch.py`

- Renders races with trained checkpoints, greedily, for as many profiles as given (`--profiles
  a,b,c` are assigned to the cars in turn).
- Every car drives with its own profile's checkpoint (`--tag`, default best), loaded once per
  profile (torch actor or FrozenPolicy).
- The grid order is shuffled every race, so no style always starts on pole.
- Each car is painted in its profile's colour, in the window and the console. The gym paints a
  crashed car red; repaint before every race.
- Playback speed = renderer fps × `--speed` (0 = as fast as possible). The physics is stepped
  exactly as in training.
- **Options:** `--tag`, `--profiles`, `--cars`, `--map` (name or `random`), `--episodes`,
  `--speed`, `--random-spawn`, `--list-maps`.
- **After each race:** a table per car with place, car, profile, distance, laps, best lap,
  average / top speed, passes, times passed, tow share, return and end reason. Then per profile,
  the fastest lap here against its all-time record on this map.

### 9.3 `compare.py`

- **Head to head (default):** two drivers race on full grids (`--cars 22`), half the cars each.
  - Every grid is raced twice with the seats swapped, so neither driver profits from luckier
    slots.
  - Scoring: F1 points 25-18-15-12-10-8-6-4-2-1 for the top-10 finishers, 0 for a crash.
  - Greedy driving, the normal car, every map in turn.
  - Races run in a `spawn` process pool of torch-free workers. Torch only loads the weights in the
    main process. Each worker keeps one env and rebuilds it only when the map changes; tasks are
    sorted by map.
  - A full-length 22-car race takes ~13 s per worker.
- **Table:**
  - F1 points (total and per race), race wins, podiums,
  - races where each outscored the other, cars finished,
  - crashes wall / car, passes on the other driver's cars,
  - average finishing place, place points per race.
- **Verdict:** the F1-points difference per race, averaged over each grid's two races, with a 95%
  confidence interval (1.96 × sd / √grids). It names a winner only if the interval excludes 0;
  otherwise it says "no clear winner, run more races".
- **Options:** `--a`, `--b` (`PROFILE[:TAG]` or a `.pt` file; default b = middle), `--races 100`,
  `--cars 22` (even), `--maps all|eval|A,B`, `--seconds 150`, `--workers 4`, `--seed 0`.
- **`--leaderboard`:** prints every profile's lap record per map side by side (* marks the fastest
  profile on each map) and how many maps each profile covers.
- **Why it exists:** the eval score can mislead. aggressive v2's best scored 15.2 pts/car in evals
  (inflated by +3 per pass, also for crashed cars). Over 100 races on all 23 maps it lost clearly
  to middle: 43.3 vs 55.5 F1 points per race, 52% of its cars finishing against 72%, 268 wall
  crashes against 96.

### 9.4 `report.py`

- Reads a profile's `metrics.csv` and `evals.csv` (`--profile`) and saves `progress.png` in its
  folder. The x axis is gradient updates.
- **Six panels:**
  1. Training distance (median, mean).
  2. Eval mean distance per map + round average, with race points per car on a second axis.
  3. Endings: survived, crashed, crashed into a car, stuck, ≥1 lap.
  4. Speed (average, top).
  5. Racing: overtakes, close calls; on a second axis tow share and win rate vs opponents.
  6. Learner: avg Q, critic loss.

---

## 10. Performance (all required)

Each of these was measured in the current code and gave identical results.

- **Lidar ray-cast cull** (§5.9): 22-car grid scan update 11.1 → 2.8 ms.
- **Racing-only physics + skipping the gym env's step** (§5.9): −13% on a 22-car race with crashes.
- **Batched progress tracking and observation building:** 576 → 73 µs and 642 → 64 µs per step at
  22 cars.
- **Together:** 22-car eval races went from 8.1–9.1 to 5.5–6.1 ms/step, and an 8-car training race
  from 3.5 to 2.8 ms/step.
- **Shared distance map** per track (§5.6); BLAS threads = 1 in every process (§7.1).
- **GPU learner:** CUDA graphs, fp16 replay in two tensors, `set_to_none` gradients, diagnostics
  on the last critic step only, TF32.
- **Torch-free workers** with a numpy actor; weights through shared memory, not queues.
- **Transitions in chunks** of 64; a bounded queue.

New, not in the current code (keep results identical where possible):
- **Vectorize the per-car reward loop** in `env.step`: progress, steering, crash and timeout terms
  over all active cars at once, with a per-car branch only for cars crossing the line or a sector
  boundary (~3% of a 22-car step).
- **Compute the traffic check** (§5.9 no progress) only for cars that are stuck (~2% of a step).
- **Compute pairwise geometry once per step** (relative frames, distances, body gaps) and share it
  between mirrors, tow, alongside and close calls: 390 µs of a 1.8 ms step at 4 cars.
- **Pinned host buffer** with non-blocking copies for replay inserts; read the dashboard's
  diagnostics without blocking the GPU (~2% learner idle).

---

## 11. Lessons from the current project (do not reintroduce)

**Bugs found and fixed**
1. The gym's Frenet tracking sticks at s ≈ 1.8 m on scaled maps → own centerline tracker.
2. Workers never refreshed weights on short episodes → refresh at every episode start.
3. The resume hung when the buffer was smaller than a batch → guard training on `len ≥ batch`.
4. Resume counters were misaligned → continue from the checkpoint's `total_it`.
5. OpenBLAS thread thrash (~500× slower workers) → one BLAS thread per process.
6. Every car stored its own 32 MB distance map → shared cache.
7. `ray_cast` edits its scan in place → take the walls-only copy first.
8. Lidar-based close calls gave fake closing speeds at edge beams → body-to-body gaps for cars,
   nearest-wall gap for walls.
9. Bang-bang gym steering → smooth steering.
10. Linear-tyre dynamic model spins → KS model + grip limit.
11. Long-run OOM from closed envs kept alive by reference loops → `gc.collect()` after every
    switch; remove the patches in `close()`.
12. Grid stall cascade → 12 m slots + launch + no-progress pause while blocked.
13. Slipstream exit braking at 2 g → drag physics.
14. Best by median kept the crashier driver → mean distance / place points.
15. Grids placed in corners on 7 tracks → `grid_anchor`.
16. A resume's first eval came minutes in (a half-adapted "best") → immediate baseline round;
    opponent identities in the signature.
17. A retaken place counted as a pass → a retake cancels the pending pass.
18. F1 top-10 points as the eval score sat at the grid-order value (4 of the brain's 7 cars started
    outside the top 10 and scored 0 whatever they did) → place points for every position in evals.
    F1 points are only for head-to-head comparisons.
19. A best checkpoint paired the evaluated actor with a newer target actor → reset the target on
    load; drop the actor optimizer from the best.

**Reward and training design**
- **A crash costs more than its penalty:** it also costs everything the car would have earned
  within the horizon.
  - Bigger per-step rewards or a longer γ make a style more careful, not bolder.
  - aggressive v1 (position 1.5/step, progress ×1.3, γ 0.99375, crash −90) had avg Q ~165, so a
    crash really cost ~255. A pass was only worth ~12% risk, and it passed less than middle.
- **One crash penalty for walls and cars makes a bold style sloppy everywhere.** aggressive v2
  (−60 for all crashes) passed 2× as often as middle, but 3 of every 4 extra crashes were walls.
  Hence `wall_crash` / `car_crash`.
- **Rewards paid when the timer runs out are invisible**, because there's no clock input. Pay
  results as they happen (race points).
- **The pass hold** (1 s, cancelled by a crash or retake) makes risky moves worth less than clean
  ones: a pass that ends in a crash or is taken straight back pays nothing, a clean one pays in full.
- **Restarts are expensive:** ~7 min of warm-up plus a dip afterwards, every time. aggressive v1:
  8 runs in 11 h (longest 2 h), three restarts from the same lucky best, ~60% of the updates
  thrown away. Leave runs alone for hours; change rewards only with evidence (compare.py, crash
  types).
- **Identical cars stay in grid order** unless someone crashes. Slower opponents in training
  (0.9–0.95), pace spread, rolling starts and the gain reward exist to create passing chances.
- **The 20 Hz control** (not the gym's 100 Hz) makes γ = 0.99 look ~5 s ahead. That's enough to
  see a corner coming and brake, and costs 5× fewer policy calls and lidar casts.

---

## 12. Out of scope

- **No test files**, test frameworks or CI.
- **No backwards compatibility:** no checkpoint upgrading and no CSV header migration. Old
  checkpoints are refused by their layout tag.
- **Removed features, don't rebuild:**
  - the road-car preset and `--car`,
  - the rookie profile, speed-limit and driver-noise knobs,
  - per-profile γ,
  - finish / end-of-race bonuses,
  - the per-step position reward.
- **No saving of the replay buffer.** Rear-view lidar is not wanted: the mirrors replace it.

---

## 13. Done when

- `train.py` runs from scratch and with `--resume` and `--opponents` without errors, with the
  dashboard and with plain output.
- `watch.py` renders a mixed-profile race.
- `compare.py` gives a verdict, and `report.py` writes a plot.
- **Speed:** a 22-car grid race steps at ≤ 6 ms/step on one core. Training reaches ~950 updates/s
  on the laptop GPU at 0.25 updates per step, with the GPU as the bottleneck.
- **Behaviour checks** (by running, not by test files):
  - DRS never opens outside a zone or under braking.
  - The ERS battery stays within [0, capacity], drains only while deploying, and fills only under
    braking.
  - Every car's race-point payments add up to (finish − start points) × per_place.
