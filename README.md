# F1 simulator

A multi-car Formula 1 style racing simulator built on the [F1TENTH gym](https://github.com/duckpop/f1tenth_gym), with 23 real circuits scaled up to F1 size. Every car is driven by a small neural network trained with reinforcement learning (TD3): it reads 133 numbers 20 times a second (lidar, mirrors, track preview, its own state, DRS, ERS, car design) and answers with five controls (steer, throttle, brake, DRS, ERS).

Three driver personalities share the same network shape and differ only in their rewards:

| Driver | Style | Best checkpoint (included) |
|---|---|---|
| `middle` | The baseline racer. Passes when the risk is small | 8,191 m mean distance in evals |
| `aggressive` | Bold in traffic, contact is cheaper | 17.80 race points per car vs middle |
| `defensive` | Very hard to pass, avoids contact | 4.34 race points per car vs middle and aggressive |

Cars also have designs (engine, battery, chassis, brakes, wing: levels 1–5, 17 points to spend), and a separate designer bot learns which design to give each driver over a season.

## Setup

Needs Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). Training needs an NVIDIA GPU; watching and comparing run on the CPU.

```bash
git clone --recurse-submodules git@github.com:duckpop/f1-simulator.git
cd f1-simulator
uv venv
uv pip install -r requirements.txt
```

The tracks (`maps/f1tenth_racetracks`) and the gym (`f1tenth_gym`) are git submodules. If you cloned without `--recurse-submodules`, run `git submodule update --init`.

## Use it

Watch the three drivers race (a window opens; the camera follows the leader, right-click or drag to look around, L to follow the leader again):

```bash
uv run watch.py --profiles middle,aggressive,defensive --cars 22 --map Monza --episodes 3 --seconds 300
```

Head-to-head comparison, 300 races with seats swapped:

```bash
uv run compare.py --a aggressive --b middle --races 300 --workers 8
```

Train a driver (here defensive, starting from middle, racing against middle and aggressive):

```bash
uv run train.py --profile defensive --resume middle/best --opponents middle,aggressive --cars 22 --steps 200000000
```

Check on a training run:

```bash
./status.sh defensive
```

Other tools: `season.py` races a season, `train_designer.py` trains the car designer, `report.py` summarises training logs. Every script has `--help`.

## What's where

| Path | Contents |
|---|---|
| `racer/` | The simulator: race environment, physics (traction circle, DRS, ERS, engine braking, tow), sensors, the TD3 agent, rollout workers, scoring, season and designer logic |
| `configs/settings.toml` | Every tunable value, with comments |
| `configs/profiles/` | The three driver personalities' rewards and eval scoring |
| `configs/designs.toml` | Team car designs for `--designs` |
| `saved_models/<driver>/` | Each driver's best checkpoint (`td3_best.pt`), its eval record and lap records |
| `train.py`, `watch.py`, `compare.py`, `season.py`, `train_designer.py` | The command-line tools |
| `project.md`, `project_v2.md` | The full specification |
| `f1tenth_gym/` | Fork of the F1TENTH gym (submodule) |
| `maps/f1tenth_racetracks/` | The F1TENTH race tracks (submodule) |
