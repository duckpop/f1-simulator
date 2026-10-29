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
| 5. Stage A: alone, on the track-width curriculum | runs on the laptop; the real runs are for the cluster |
| 6. Stage B: full grid, mixed profiles, frozen snapshots | runs on the laptop (smoke runs); the real runs are for the cluster |
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
f1env-tracks --out tracks --train 500 --eval 10   # the track pool, with the wide copies (once)
f1sim-train --profile middle --tracks tracks --preset cluster
f1sim-train --profile smooth --tracks tracks --preset cluster    # starts from Middle's best (and the other two)
f1sim-eval --checkpoint runs/middle/stage-a/best.npz --tracks tracks --laps 50
f1sim-train-b --tracks tracks --preset cluster    # stage B: all four together, each from its stage A best
f1sim-train-b --tracks tracks --preset cluster --resume   # after a stop: each profile from its stage B latest.npz
f1sim-season --tracks tracks                      # Phase 2's evaluation and the comparison with v2
```

Without a pool, `--generate N` makes N training tracks on the spot (and the eval tracks from their seeds), for small
runs: `f1sim-train --profile middle --generate 4 --eval-tracks 2 --widths normal --steps 20000`.

`pytest -q` runs the tests (~3 minutes); `pytest -m slow` adds the pendulum check.

## The brain: TQC

SAC's stochastic actor and automatic entropy tuning, with **5 critics x 25 quantiles = 125 atoms**. The critics' target
pools all 125 atoms of the target critics, sorts them and drops the top **d x 5**: more dropped makes the critic more
pessimistic, so the driver avoids risk (`src/f1sim/tqc.py`).

| Part | Setting |
|---|---|
| Inputs | f1env's 209 driver inputs: the lidar (50 beams x edge, closing speed, barrier) through a 1D conv along the beams, the other 59 through an MLP (128), joined |
| Actor | 256 · 256, a tanh-squashed Gaussian over 5 actions; the mean action in evals |
| Critics | 512 · 512 · 512 each, the 5 batched with `vmap`, the action joined after the encoder |
| Training | Adam 3e-4 (actor, critics, alpha), batch 256, Polyak 0.005, 5-step returns, quantile Huber loss (kappa 1) |
| Actions (as v2) | steer; throttle and brake, each on above 0; DRS and ERS asked for above 0 |

Each network (the actor and every critic) has its own input branches, so the actor's and the critics' losses never
pull on shared weights.

## The four profiles

`configs/profiles/*.toml`; the shared settings and the machine presets are in `configs/drivers.toml`.

| Profile | d (atoms dropped) | Target entropy | gamma (horizon) | Starts from |
|---|---|---|---|---|
| Middle | 2 (10 of 125) | -2.0 | 0.995 (10 s) | scratch |
| Aggressive | 1 (5) | -1.5 | 0.985 (3.3 s) | Middle's stage A |
| Smooth | 3 (15) | -2.5 | 0.997 (17 s) | Middle's stage A |
| Defensive | 2 (10) | -2.0 | 0.990 (5 s) | Middle's stage A |

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
- Every car gets random upgrade levels each race (each category 0-10, per car).
- **The training principal** (`principal.py`) boxes each stint at a tyre life drawn from 8-12%, for the next-harder
  compound; wets a lap before rain, slicks a lap before the track is dry enough; pace normal, no team orders, ERS
  "deploy".
- A crash ends the episode; the car rejoins from the pit exit ~5 s later and starts a new one (f1env's training
  respawn). A race that's over, or past 2.5x its lap-simulation time, restarts at once on a new track, inside the
  compiled step; the cars still going are cut there (bootstrapped).
- The replay stores each step once per car (inputs in float16) with whether the driver was in control and how the step
  ended; n-step returns and the next inputs are found when sampling (`src/f1sim/replay.py`).

**Stage A** (`train.py`): one car per race, on wide copies of the tracks first (~26 m, then ~20 m), then the 12-15 m
tracks, moving on when the eval (50 laps alone on each eval track, every 50k steps on the laptop) stops improving once cars finish races (until then, at the width's step
budget). Middle trains first, from scratch; the other three start
from its best stage A networks with their own settings and rewards. The eval at a wide width runs on the eval tracks'
wide copies that fit (at 26 m only 2 of the 10 do, at 20 m 5), each run more times so every width's eval has as many
races; if none fit, it runs on the normal ones.

**Stage B** (`stage_b.py`): 22 cars per race on the 12-15 m tracks, with overtaking, dirty air, slipstream, DRS,
contact, the safety car and blue flags.

- Seats are fixed by car number (the grid is random every race): 4 cars of each profile learn (16), team-mates are
  different profiles, and the other 6 cars (27%) drive frozen snapshots.
- Each profile keeps its own networks, optimisers and replay buffer. One batch of races trains all four: each car's
  steps go to its profile's buffer with its profile's reward weights, and every profile takes its gradient steps each
  environment step.
- The snapshot pool (8) starts with the four stage A bests and takes a copy of one profile's current actor, in turn,
  every 100k steps. A frozen car drives one snapshot (its mean action) for a whole session.
- 20% of sessions are 12-minute qualifying sessions: every car starts in its garage and goes out twice on new softs
  (f1env's default runs), so flying laps and the "session" input get learned. A run ends when the car parks.
- The eval is a full weekend on every eval track, as in F1: Q1, Q2 and Q3 set the grid, then a 50-lap race (`eval_laps`),
  the four current profiles against each other (5-6 cars each, the default car, no respawn): points per race, finish
  rate, out on lap 1, crashes per lap, time off the track, overtakes, places lost and attacks held. Each profile keeps
  its best checkpoint (most points per race) in `runs/<profile>/stage-b/best.npz`. It runs every `eval_every_b` steps
  (50k on the laptop).

## Watching a driver

```bash
f1sim-watch --tracks tracks                          # Middle's current best, alone, on eval track 0, live
f1sim-watch --tracks tracks --cars 22 --track 3      # the same driver in all 22 cars
f1sim-watch --tracks tracks --save                   # also save the race as runs/watch/<profile>-<track>-<width>.html
f1sim-watch --tracks tracks --no-window              # no window (on the cluster): only the HTML replay
f1sim-watch --profile smooth --tracks tracks         # any profile; or --checkpoint <file>
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

The default driver is the profile's newest best checkpoint (stage B's, else stage A's best, else the newest
`best-<width>.npz` while the curriculum is still on the wide tracks), at the width it was saved at; at 26 m only eval
tracks 3 and 7 have a wide copy, other tracks fall back to the normal width. It uses the mean action, the default car and
the training principal; add `--respawn` to let crashed cars rejoin as in training.

## Evaluation

`f1sim-season` (`src/f1sim/drivers/season.py`) runs Phase 2's tests on the 10 fixed eval tracks with the default car
(no upgrades) and each profile's best checkpoint (stage B, else stage A), driving the actors' mean actions:

| Test | What's measured | v2 to beat |
|---|---|---|
| Alone, 50 laps per profile | finish rate, crashes per lap, lap time vs the lap simulation, laps per stint | finished 53%, crashes 1.39% per lap |
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
(`[rewards]` in `configs/drivers.toml`) times the profile's multiplier for style terms. Every term is logged
separately (`runs/<profile>/stage-a/log.jsonl`).

| Term | Base | Middle | Aggressive | Smooth | Defensive |
|---|---|---|---|---|---|
| Progress, on the track | +1 per step at top speed (as v2) | | | | |
| Crash (ends the episode) | -70 | | | | |
| Warning (the 4th is a black flag) | -50 | | | | |
| Bump (under 20 km/h) | -2 | | | | |
| Pit lane entered on a call | +5 | | | | |
| Dead tyre | -2 per s | | | | |
| Finish | +50 | | | | |
| Off the track (the car's centre); its progress doesn't count either | -2 per s | | | | |
| Overtake (held 2 s) | +20 | 1x | 2x | 0.5x | 0.5x |
| Place lost | -20 | 1x | 0.5x | 1x | 2x |
| Holding under attack | +1 per s | 1x | 0.5x | 0.5x | 2x |
| Inside line, under attack | +2 per s | 0 | 0 | 0 | 1x |
| Tyre wear | -10 per % of life | 1x | 0.5x | 2x | 1x |
| Smoothness: input changes / wheelspin and lock-ups | -0.05 / -0.2 per unit | 1x | 0.5x | 2x | 1x |

For f1env's scripted driver, tyre wear costs ~5% of the progress reward and the smoothness terms almost nothing.

Choices made while building (open points in the spec):

- **No farming by swapping places:** with Aggressive's weights an overtake is worth 4x a place lost, so two cars could
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
| One TQC gradient step (5 critics x 512 · 512 · 512, batch 256) | ~185 | |
| Stage A, 64 races (laptop preset), one gradient step per step | ~130 | ~8,500 transitions; peak 2.3 GB of GPU memory with one set of tracks (two sets with rotation: ~0.3 GB more, not yet measured) |
| Stage B, 16 races x 22 cars (laptop preset), a gradient step per profile per step | ~38 | ~10,000 learning transitions (~2,500 per profile); peak 2.2 GB |
| The season eval, 22 cars, four drivers | ~450 | the same for 4 parallel seasons; a full season ~2.5 hours |

With one gradient step per environment step, more races per step means each transition is reused less (about 4 times
at 64 races); the spec leaves that ratio to be tuned on the cluster (`updates` in the preset). Compiling the four
learners and the full grid takes about a minute.

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
