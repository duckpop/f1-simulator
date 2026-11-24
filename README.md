# F1 simulator v3

F1 Racer v3, rebuilt from scratch: the agents that race in [f1env](https://github.com/duckpop/f1env), the v3
environment (generated tracks, a JAX race simulation on the GPU). Nothing here comes from v2's code; v2 (TD3 drivers on
the F1TENTH gym) lives on the `main` branch.

This branch starts with **Phase 2: the drivers**. A **TQC** brain (Truncated Quantile Critics) learns to drive in four
profiles: **Middle**, **Aggressive**, **Smooth** and **Defensive**. Only drivers are trained. Cars get random upgrades,
and a scripted principal calls the pit stops. Phases 3 (principals) and 4 (designers) will be added here later.

| Build step (Phase 2 spec, section 6) | Status |
|---|---|
| 0. Training repo: f1env pinned, tests, a config per profile | done |
| 1. TQC agent in JAX, checked on a standard control task first | done (pendulum: -131 per episode, random -1,200) |
| 2. Replay buffer on the GPU: float16, n-step returns | done |
| 3. Training setup: random upgrades per car, the training principal, race and session sampler | done (20% qualifying sessions in stage B) |
| 4. Rewards: common and style terms, each logged | done |
| 5. Stage A: Middle alone, from scratch | ready (restart from scratch, 2026-11-24) |
| 6. Stages B-D: B Middle in a field with every rule, C Middle obeying team orders, D the personalities from Middle | ready (2026-11-24) |
| 7. Eval harness | done: alone, mixed races (stage B's pick), a 24-race season of full weekends, style metrics |
| 8. Results against v2 | `f1sim-season` writes the report and checks the done criteria; it needs trained drivers |

## Install

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/). f1env is pinned to a commit (`pyproject.toml`); it's a
private repo, so installing it takes a GitHub SSH key with access to it (on the cluster too). On a desktop whose key
lives in the keyring, the install waits for the keyring's unlock prompt.

```bash
uv venv
uv pip install -e ".[cuda,dev,live]"     # leave out cuda for CPU only; live: f1sim-watch's window
```

Working on f1env at the same time? Install your local copy over the pinned one: `uv pip install -e ../f1env`.

## Quick start

```bash
f1sim-check-tqc                                   # TQC on the pendulum swing-up: a few minutes, should pass
f1env-tracks --out tracks --train 500 --eval 10 --width 22 --no-wide   # the track pool: every track 22 m wide (once)
f1sim-train --tracks tracks                       # A: Middle alone, from scratch -> runs/middle/stage-a/best.npz
f1sim-train-b --stage b --tracks tracks           # B: Middle in a field, every rule, pit stops (from A's best)
f1sim-train-b --stage c --tracks tracks           # C: Middle obeying random team orders (from B's best)
f1sim-train-b --stage d --tracks tracks           # D: Smooth, Aggressive, Defensive, each from Middle's C best
f1sim-train-b --stage b --tracks tracks --resume  # after a stop: carry on from this stage's latest.npz
f1sim-eval --checkpoint runs/middle/stage-a/best.npz --tracks tracks --laps 50
f1sim-season --tracks tracks                      # Phase 2's evaluation and the comparison with v2
python -m f1sim.drivers.plot runs --out runs/curves.png
                                                  # the newest stage's curves (several run directories: overlaid)
```

Without a pool, `--generate N` makes N training tracks on the spot (and the eval tracks from their seeds), for small
runs: `f1sim-train --generate 4 --eval-tracks 2 --steps 20000`.

`pytest -q` runs the tests (~3 minutes); `pytest -m slow` adds the pendulum check.

## The brain: TQC

SAC's stochastic actor and automatic entropy tuning, with **5 critics x 25 quantiles = 125 atoms**. The critics' target
pools all 125 atoms of the target critics, sorts them and drops the top **d x 5**: more dropped makes the critic more
pessimistic, so the driver avoids risk (`src/f1sim/tqc.py`).

| Part | Setting |
|---|---|
| Inputs | f1env's 188 driver inputs: the lidar (50 beams x edge, barrier, each on a log scale: log(1 + d / 5 m) / log(1 + 390 / 5)) through a 1D conv along the beams, the other 88 through an MLP (128), joined. The last 4 (the blend line 2026-11-12, the stuck timer and the wait level 2026-11-16) were appended: older checkpoints load with them at zero weight. The 25 before the last (the cars around, the safety car) were added 2026-11-05: older checkpoints load with them at zero weight. The lidar's closing speeds (50) were removed 2026-11-06: an older checkpoint loads without that channel, but its lidar now reads a different scale, so it has to relearn it |
| Actor | 256 · 256 · 256, a tanh-squashed Gaussian over 5 actions; the mean action in evals |
| Critics | 512 · 512 · 512 · 512 each, the 5 batched with `vmap`, the action joined after the encoder |
| Older checkpoints | one layer fewer each (until 2026-11-05): `checkpoint.load(..., cfg)` deepens them with identity layers before each output, so they drive exactly as before and learn on with the extra depth (~4% slower training) |
| Training | Adam 3e-4 (actor, critics, alpha), batch 256, Polyak 0.005, 5-step returns, quantile Huber loss (kappa 1) |
| Actions (as v2) | steer; throttle and brake, each on above 0; DRS and ERS asked for above 0 |

Each network (the actor and every critic) has its own input branches, so the actor's and the critics' losses never
pull on shared weights.

## The four profiles

`configs/profiles/*.toml`; the shared settings and the machine presets are in `configs/drivers.toml`.

| Profile | d (atoms dropped) | Target entropy | gamma (horizon) | Starts from |
|---|---|---|---|---|
| Middle | 2 (10 of 125) | -2.0 | 0.994 (8.3 s) | scratch |
| Aggressive | 1 (5) | -1.5 | 0.990 (5 s) | Middle's stage A |
| Smooth | 3 (15) | -2.5 | 0.996 (12.5 s) | Middle's stage A |
| Defensive | 2 (10) | -2.0 | 0.992 (6.3 s) | Middle's stage A |

## The training environment

`src/f1sim/drivers/env.py`: f1env races, batched as races x cars and jit-compiled, with everything a step needs on the
GPU (the env step, the replay and the gradient steps run together in one `lax.scan`, `src/f1sim/loop.py`).

- **Track rotation** (`Rotation` in `tracks.py`): the GPU holds two sets of `tracks` (16 on the laptop, 200 on the
  cluster) and new races draw from the newer set. Every `rotate_every` steps (100k on the laptop) the older set is
  swapped for the next block of the shuffled pool, so a run works through every track before any comes back; each race
  gets at least a full interval to finish, and only races still on the swapped set restart (their last step is
  marked as cut in the replay). The track data is padded to fixed sizes, so a swap doesn't recompile the training step.
- Each race: a random track of the active set, 5 laps up to the full 305 km (half of them long enough to need a pit stop),
  a random grid slot, a standing start, f1env's weather (rain in 20% of races).
- Stage B (since 2026-11-06): each session draws its field, 6, 12, 18 or 22 cars (`field_sizes`; the others are out
  from the start, the learning seats keep their share and the stored ones race first), and half the races
  (`spread_share`) start with the cars spread around the lap at 0.6-0.9x the reference pace, lights out, in a random
  order, instead of a standing start on the grid.
- Every car gets random upgrade levels each race (each category 0-10, per car).
- Sensor noise in training (`[env] sensor_noise`, 0.05): every step each sensed distance and speed is off by up to
  5% (the lidar's edge and barrier distances, and the gaps, offsets and speeds of the cars around, in the mirrors and
  alongside, and the safety car's), so a driver can't count on exact readings; "nobody there" stays exact and the
  evals see everything exact.
- **The training principal** (`principal.py`) boxes each stint at a tyre life drawn from 8-12%, for the next-harder
  compound; wets a lap before rain, slicks a lap before the track is dry enough; pace normal, no team orders, ERS
  "deploy".
- A crash ends the episode and the car is out for the rest of its race (`[env] respawn = false` since 2026-11-01, so
  a crash isn't just a short setback; with it on, the car rejoins from the pit exit ~5 s later as a new episode). A race that's over, or past 2.5x its lap-simulation time, restarts at once on a new track, inside the
  compiled step; the cars still going are cut there (bootstrapped). In stage B only the learning cars keep a race
  going (`learners`): once they're all out it restarts, rather than waiting for the frozen cars.
- The replay stores each step once per car (inputs in float16) with whether the driver was in control and how the step
  ended; n-step returns and the next inputs are found when sampling (`src/f1sim/replay.py`).

**The training regime** (the user's, 2026-11-24; old models deleted, their evals in archive/runs-before-2026-11-24/):
only Middle learns until D, then the personalities branch off it.

- **A** (`train.py`, `f1sim-train`): Middle alone, one car per race on the pool's 22 m tracks, from scratch, until
  the eval (50 laps alone on each eval track) stops improving for `patience` evals once cars finish races.
- **B** (`stage_b.py --stage b`): Middle in fields of 6-22 cars with every rule (overtaking, contact, the safety car,
  flags, pit stops, rain), against frozen snapshots of its own earlier bests, from its A best.
- **C** (`--stage c`, `[stage_c]`): B's races plus a random principal giving Middle team orders every 30-90 s (pace
  push/save, hold position, let the team-mate past, defend, ERS deploy/save; "no order" half the time), paid by the
  order reward terms; from its B best.
- **D** (`--stage d`, `[stage_d]`): C's races with Smooth, Aggressive and Defensive learning, each from Middle's C best.

Stages B-D start from a checkpoint, so each fills part of its replay and trains the critics alone first (`[warm_start]`).
Their eval is a weekend on every eval track, a one-lap qualifying then the race (in C and D with orders for every car) against a reference field:
the stage's starting Middle, frozen, so a best means better than where the stage started. Details: `stage_b.py`'s
docstring and the comments in `configs/drivers.toml`. Speed options in the config: one encoder shared by the critics
(`[tqc] shared_encoder`), bfloat16 dense layers (`[encoder] bf16`), 64 lidar steps per beam (`[env] march_steps`).

## Watching a driver

```bash
f1sim-watch --tracks tracks                          # Middle's current best, alone, on eval track 0, live
f1sim-watch --tracks tracks --cars 22 --track 3      # the same driver in all 22 cars
f1sim-watch --tracks tracks --save                   # also save the race as runs/watch/<profile>-<track>.html
f1sim-watch --tracks tracks --no-window              # no window (on the cluster): only the HTML replay
f1sim-watch --profile smooth --tracks tracks         # any profile; or --checkpoint <file>
f1sim-watch --tracks tracks --profiles middle,aggressive,defensive --save   # 22 cars, car k drives profile k mod 3,
                                                     # each car labelled with its profile (tower, log, followed car)
```

A window shows the race as it runs, in real time: f1env's track drawing with the cars on it, following one car, and a
side panel with that car's place, lap, speed, tyres, ERS and DRS, and the timing tower. Keys: space pauses, f switches
between following a car and the whole track, n / p pick the next / previous car, + / - zoom, up / down change the speed
(`--speed` to start at another). `--save` also records the race and writes f1env's HTML replay page when the window is
closed (play, scrub, zoom, follow any car, the race log); `--no-window` only writes that.

The window needs the `live` extra (matplotlib and PyQt6: `uv pip install -e ".[live]"`). It uses Qt's generic theme:
Qt's GTK 3 theme loads the system's GTK 3, which aborts on GNOME 50 (it reads a setting gnome-settings-daemon 50
removed). It runs on the CPU by
default (`--device gpu` to change), where one race is quick (about 1 ms a step alone, 4 ms with 22 cars), so it neither
slows down nor queues behind a training run on the GPU: ~30 frames/s with 22 cars on the laptop.

The default driver is the profile's newest best checkpoint along the regime (stage D's, else C's, B's, A's, else stage
A's latest). It uses the mean action, the default car and the training principal; add `--respawn` to let crashed cars
rejoin.

## Evaluation

`f1sim-season` (`src/f1sim/drivers/season.py`) runs Phase 2's tests on the 10 fixed eval tracks with the default car
(no upgrades) and each profile's newest best checkpoint (stage D, else C, B, A; Middle's for a personality with none
yet), driving the actors' mean actions:

| Test | What's measured | v2 to beat |
|---|---|---|
| Alone, 50 laps per profile | distance completed, reward per step (the profile's weights), finish rate, crashes per lap, lap time vs the lap simulation, laps per stint, km per tyre set | finished 53%, crashes 1.39% per lap |
| A 24-race season: f1env's eval calendar and round keys, full weekends (Q1, Q2, Q3 decide the grid, 305 km races, points), 5-6 cars per profile, no respawn | points per race, finish rate, out on lap 1, crashes per lap, titles | Middle 36% finished, 28% out on lap 1 |
| Style, from the season's races | overtakes per race, attacks held (a spell with a rival within 1 s behind that ends without losing the place), km per tyre set at the profile's race speed, time in wheelspin or lock-up | |

It checks Phase 2's done criteria (every profile beats v2 Middle alone; fewer cars out on lap 1 than 28%; Aggressive
makes the most overtakes, Defensive holds the most attacks, Smooth's tyres last longest at a race speed within 2% of
the median, Middle scores the most points or within 5% of the best) and writes `runs/results/phase2-<date>.md` and
`.json`. f1env's `F1Env` runs the weekends (sessions, knockouts, the grid, points, standings); this code drives every
session in jitted chunks. A full season is ~4 million steps, mostly the 305 km races; `--seasons 4` runs four in
parallel at almost no extra cost (different weather, lights and stops). `--rounds`, `--race-km` and `--alone-laps` give
a quicker look.

## Rewards

`src/f1sim/drivers/rewards.py`. Each term is measured in its own unit and weighted by its base value
(`[rewards]` in `configs/drivers.toml`) times the profile's multiplier for style terms; the sum is multiplied by
`scale` (0.1), so the critics see a crash as -250. Every term is logged
separately (`runs/<profile>/stage-a/log.jsonl`).

| Term | Base | Middle | Aggressive | Smooth | Defensive |
|---|---|---|---|---|---|
| Progress, on the track, against the reference pace (the lap simulation's speed there, or the speed limit if lower) | +1 per step at reference pace | | | | |
| Crash (ends the episode and the car's race); every car in a contact crash pays it in full, the one hit too | -2500 | | -2000 | -2500 | -3000 |
| Black flag (the 4th warning; ends the car's race) | -2500 | | | | |
| Sector: each third of a lap driven on the track this race (once each) | +10 | | | | |
| Closer than 0.5 s behind the car ahead with the safety car called or out | -5 per s | | | | |
| Warning: the 1st, 2nd and 3rd cost 25%, 50% and 75% of a black flag, so a disqualification costs 250% of one in all (since 2026-11-06; was -50 each) | -625 / -1250 / -1875 | | | | |
| Bump (under 20 km/h) | -2 | | | | |
| Pit lane entered on a call | +5 | | | | |
| Dead tyre | -2 per s | | | | |
| Finish | +50 | | | | |
| Off the track (the car's centre); its progress doesn't count either | -2 per s | | | | |
| Standing still (under 1 m/s on the track, not queuing behind a moving car) | -20 per s | | | | |
| Overtake (held 2 s) | +20 | 1x | 2x | 0.5x | 0.5x |
| Place lost | -25 | 1x | 0.6x | 1x | 1.8x |
| Holding under attack | +1 per s | 1x | 0.5x | 0.5x | 2x |
| Inside line, under attack | +2 per s | 0 | 0 | 0 | 1x |
| Tyre wear beyond the compound's rated wear | -10 per % of life | 1x | 0.5x | 2x | 1x |
| Smoothness: input changes / wheelspin, lock-ups and sliding past the tyres' peak | -0.05 / -0.5 per unit | 1x | 0.5x | 1.5x | 1x |
| Qualifying place: 21 points for P1 down to 0 for P22 (no time: 0) | +1 per point | | | | |
| Qualifying: in the top 16 (would move on to Q2) / in the top 10 (Q3) | +3 / +5 | | | | |
| Qualifying: a new personal best lap (the first time included) | +3 | | | | |
| Launch: in the 2 s after the lights go out at a standing start, seconds of throttle with the brake fully off | +20 per s | | | | |
| Car close call: seconds at its level, 1 - time to contact / 0.4 s, from every other car's gap and closing speed (wrecks too) | -20 per s | 1x | 0.5x | 1.5x | 2x |
| Barrier close call: the same to the nearest barrier | -20 per s | 1x | 1x | 1.5x | 1.5x |

The qualifying terms are paid as the standing changes, not at the flag (minutes away, far beyond any horizon): a car
gets a place's points when its lap puts it there and gives them back when another car's lap pushes it down while it's
out on track, so over a session it collects what its final place is worth. Training's qualifying sessions are one
22-car session, so the top 16 and top 10 stand in for moving on to Q2 and Q3.

Since 2026-10-31 progress pays against the reference pace, so a step at it pays +1 in a hairpin, on a straight or
behind the safety car (before: +1 per step at the 310 km/h top speed, ~0.6 on a clean lap, ~0.4 behind the safety car:
the reward mostly said where a car was, and a safety car period was a stretch of low reward the driver couldn't
change). Tyre wear is charged only beyond the compound's rated wear for the distance, and f1env's tyres last twice as
long, so the scripted driver pays almost nothing for its tyres; the trained drivers, near the grip limit, still do.

A profile can replace base values in its own `[rewards]` table: the crash costs -40 for Aggressive (bolder) and -120
for Defensive (more careful). Smooth's crash costs -200: a crash ends the episode,
so it has to cost more than driving on at a loss. With its doubled tyre and smoothness costs a learning Smooth lost
~0.8 per step (~270 over its 333-step horizon), and at -70 it learned to crash on purpose in stage B. At -300 it
stopped doing that (+0.10 per step without its crashes, 250k steps into the next stage B run), but its crashes then
cost -0.36 per step and swamped every other term, so it went to -140, and its smoothness from 2x to 1.5x; then to -200.

A contact crash costs each car by its share of the blame (f1env's, from how fast each car moved towards the other at
contact): the car that drove into the other, or into a wreck, pays the full cost, the car it hit 30%. In a lap-1
sample 55% of the cars out had hit another car, mostly the car ahead, and the car hit can't learn much from a cost it
couldn't avoid. A red flag on laps 1-3 restarts the race from the grid (f1env), so starts get practised more.

Choices made while building (open points in the spec):

- **No farming by swapping places:** with Aggressive's weights an overtake is worth 2.7x a place lost, so two cars could
  gain by passing each other back and forth. An overtake pays only for a place better than the best one already paid
  this episode.
- **No corner cutting:** f1env has no track limits (run-off only has less grip) and progress is measured along the
  centreline, so Middle learned to cut the inside of esses across the run-off. Progress made with the car's centre off
  the track doesn't count, and each second off costs 2 more; evals and the season report show the share of time off.
- **The pit term has only its "+" side:** f1env's autopilot takes every car with a committed call into the pit lane, so
  a driver can't miss a called stop (other than by crashing first).
- **The base values are first proposals**, set against v2's progress scale as the spec asks; they're one table in the
  config.

## Speed (laptop, RTX 3050 Ti)

A step costs about the same whatever it carries, so the presets fill each step with many races; the gradient steps are
the rest of the cost.

| | Steps/s | Per second |
|---|---|---|
| The environment alone, 8 to 256 one-car races | ~370 | up to ~95,000 car-steps (v2: ~3,500) |
| One TQC gradient step (5 critics x 512 · 512 · 512, batch 256; a fourth 512 layer since 2026-11-05) | ~185 | |
| Stage A, 64 races (laptop preset), one gradient step per step | ~144 | ~9,200 transitions; 2.8 GB of GPU memory in use at the peak (2M replay, two sets of 20 tracks, rotating, an eval) |
| Stage B, 16 races x 22 cars (laptop preset), a gradient step per profile per step | ~39 | ~10,000 learning transitions (~2,500 per profile); 500k replay per profile (the peak in use not measured yet at these settings) |
| The season eval, 22 cars, four drivers | ~450 | the same for 4 parallel seasons; a full season ~2.5 hours |

With one gradient step per environment step, more races per step means each transition is reused less (about 4 times
at 64 races); the spec leaves that ratio to be tuned on the cluster (`updates` in the preset). Compiling the four
learners and the full grid takes about a minute.

The trainers let JAX use 92% of the GPU (`XLA_PYTHON_CLIENT_MEM_FRACTION`; JAX stops at 75% by default): ~3.4 GB of
the laptop's 4. Keep the peak in use well under that: with two sets of 32 tracks stage A peaked at 3.43 GB and ran out
after an hour (fragmentation, at a track rotation). `nvidia-smi` shows JAX's whole pool, not what's in use; JAX's own
`jax.devices()[0].memory_stats()["peak_bytes_in_use"]` does. Set the fraction lower for a card shared with a display.

## What's where

| Path | What |
|---|---|
| `src/f1sim/nets.py` | the networks in plain JAX (input branches, actor, critic ensemble) |
| `src/f1sim/tqc.py` | TQC: truncation, quantile Huber loss, actor and temperature updates |
| `src/f1sim/replay.py` | the GPU replay buffer and n-step sampling |
| `src/f1sim/loop.py` | collect-and-learn, jit-compiled over many steps |
| `src/f1sim/pendulum.py` | the pendulum check (`f1sim-check-tqc`) |
| `src/f1sim/checkpoint.py` | checkpoints as .npz plus a JSON record |
| `src/f1sim/drivers/` | the training environment, principal, rewards, configs, tracks, stage A (`train.py`) and stage B (`stage_b.py`) training, evals |
| `configs/` | shared settings, machine presets, the four profiles |
| `tests/` | unit tests for each part |
