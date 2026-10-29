"""Watch a trained driver race: live in a window, and optionally saved as an HTML replay (f1env's viewer).

    f1sim-watch --tracks tracks                          # Middle's current best, alone, on eval track 0, live
    f1sim-watch --tracks tracks --cars 22 --track 3      # the same driver in all 22 cars
    f1sim-watch --tracks tracks --save                   # also write runs/watch/<profile>-<track>-<width>.html
    f1sim-watch --tracks tracks --no-window              # no window (on the cluster): only the HTML replay

It runs on the CPU unless --device gpu: one race is quick there, and a training run keeps the GPU to itself.

In the window: space pauses, f switches between following a car and the whole track, n / p pick the next / previous
car, + / - zoom, up / down change the speed (1x is real time). Closing the window ends the race; with --save the replay
is written then.

The default checkpoint is the profile's newest best: stage B's best.npz, else stage A's best.npz, else the newest
best-<width>.npz (the curriculum's current width), else latest.npz. It races on the eval track at the width the
checkpoint was saved at (the normal one if that track has no copy at that width), with its mean action, the default
car (no upgrades) and the training principal calling the stops; no respawn unless --respawn.
"""

from __future__ import annotations

import argparse
import os
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from f1env.race import FINISHED, OUT, RACING, WRECK, RaceConfig
from f1env.sensors import BODY_CENTRE, BODY_LENGTH, BODY_WIDTH, make_bank
from f1env.viewer import Recorder, write_html

from .. import checkpoint, tqc
from . import env as de
from .config import PROFILES, load
from .tracks import eval_track, widened

STEP_S = 0.05
FRAME_MS = 30  # the live window aims at ~30 frames a second
FRAME_STEPS = 160  # at most this many race steps per frame (8 s of racing): beyond that the window drops the backlog
COMPOUNDS = ("softs", "mediums", "hards", "wets")
KEYS = "space: pause  f: follow / whole track  n p: next / previous car  + -: zoom  up down: speed"


def newest_best(runs, profile: str) -> Path:
    base = Path(runs) / profile
    for p in (base / "stage-b" / "best.npz", base / "stage-a" / "best.npz"):
        if p.exists():
            return p
    widths = sorted((base / "stage-a").glob("best-*.npz"), key=lambda p: p.stat().st_mtime)
    if widths:
        return widths[-1]
    if (base / "stage-a" / "latest.npz").exists():
        return base / "stage-a" / "latest.npz"
    raise FileNotFoundError(f"no checkpoint for {profile} under {base} yet (the first is saved at the first eval)")


@partial(jax.jit, static_argnames=("cfg", "ecfg", "keep"))
def _run(bank, actor, st, obs, keys, cfg, ecfg, keep):
    """len(keys) steps with the actor's mean action in every car; keep: also every step's race state and events."""

    def body(c, k):
        st, obs = c
        a = tqc.act(actor, cfg, obs.reshape(-1, de.OBS), k, mean=True).reshape(1, ecfg.cars, de.ACTIONS)
        st, obs, _, _, ev = de.step(bank, st, a, k, ecfg, events=True)
        return (st, obs), ((st.race, ev) if keep else None)

    return jax.lax.scan(body, (st, obs), keys)


class Race:
    """One race on `track`, the same actor in every car, stepped in compiled blocks (compiled once per block size and
    setup, shared between races). With a recorder, every step's state and events are copied back once per block."""

    def __init__(self, track, actor, cfg: tqc.TQCConfig, cars: int, laps: int, respawn: bool, seed: int):
        self.bank, self.actor, self.cfg = make_bank([track]), actor, cfg
        ecfg = de.EnvConfig(races=1, cars=cars, fixed_laps=laps, upgrades=False, random_grid=cars > 1, auto_reset=False, race=RaceConfig(respawn=respawn))
        self.ecfg = ecfg
        self.key = jax.random.PRNGKey(seed)
        k_reset, self.key = jax.random.split(self.key)
        self.state, self.obs = de.compiled("reset", ecfg)(self.bank, k_reset, tid=jnp.zeros(1, jnp.int32))
        self._done = de.compiled("race_done", ecfg)

    def steps(self, n: int, rec: Recorder | None = None):
        keys = jax.random.split(self.key, n + 1)
        self.key = keys[0]
        (self.state, self.obs), hist = _run(self.bank, self.actor, self.state, self.obs, keys[1:], cfg=self.cfg, ecfg=self.ecfg, keep=rec is not None)
        if rec is not None:
            races, events = jax.device_get(hist)
            for j in range(n):
                rec.add(*jax.tree.map(lambda a: a[j], (races, events)))

    @property
    def over(self) -> bool:
        return bool(np.asarray(self._done(self.bank, self.state.race, self.state.limit))[0])

    @property
    def clock(self) -> float:
        return float(self.state.race.clock[0])


def record(track, actor, cfg: tqc.TQCConfig, cars: int = 1, laps: int = 5, respawn: bool = False, seconds: float = 0.0, seed: int = 0, title: str = "", max_steps: int = 400_000):
    """No window: (Recorder, final RaceState) for one race."""
    race = Race(track, actor, cfg, cars, laps, respawn, seed)
    rec = Recorder(track, race=0, title=title or track.name)
    for _ in range(max_steps // 40):
        race.steps(40, rec)
        if (seconds and race.clock >= seconds) or race.over:
            break
    return rec, jax.device_get(race.state.race)


def _outlines(x, y, yaw):
    """(C, 4, 2) each car's body."""
    c, s = np.cos(yaw), np.sin(yaw)
    cx, cy = x + BODY_CENTRE * c, y + BODY_CENTRE * s
    hl, hw = BODY_LENGTH / 2, BODY_WIDTH / 2
    lx, ly = np.array([hl, hl, -hl, -hl]), np.array([hw, -hw, -hw, hw])
    return np.stack([cx[:, None] + lx * c[:, None] - ly * s[:, None], cy[:, None] + lx * s[:, None] + ly * c[:, None]], axis=-1)


def _clock(t: float) -> str:
    return "lights" if t < 0 else f"{int(t // 60)}:{t % 60:04.1f}"


def _state_text(race, i: int) -> str:
    status, stage = int(race.status[0, i]), int(race.pits.stage[0, i])
    if status == FINISHED:
        return f"finished P{int(race.finish_pos[0, i])}"
    if status in (WRECK, OUT):
        return "out (crashed)" if race.dnf[0, i] else "out"
    return "pit lane" if stage else ("crashed earlier, rejoined" if race.dnf[0, i] else "racing")


def live(track, actor, cfg: tqc.TQCConfig, cars: int = 1, laps: int = 5, respawn: bool = False, seconds: float = 0.0, seed: int = 0, title: str = "", speed: float = 1.0, follow: int = 0, save: bool = False, close_when_done: bool = False):
    """Race in a window, in real time (x speed). Returns (Recorder or None, final RaceState)."""
    if os.environ.get("QT_QPA_PLATFORMTHEME", "").lower() in ("", "gtk3", "gtk2", "gtk", "gnome", "xdgdesktopportal"):
        # Qt's GTK 3 theme loads the system's GTK 3, which aborts on GNOME 50 (it reads the antialiasing key that
        # gnome-settings-daemon 50 removed); Qt's own generic theme doesn't touch GTK
        os.environ["QT_QPA_PLATFORMTHEME"] = "generic"
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.wayland.textinput=false")  # harmless input-method chatter
    import matplotlib

    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    from f1env.tracks.render import render_track

    race = Race(track, actor, cfg, cars, laps, respawn, seed)
    rec = Recorder(track, race=0, title=title or track.name) if save else None
    # compile every block size now, not while the window is up (a compile freezes it for seconds); the start lights are
    # on for at least 5 s, so these first steps change nothing
    t0 = time.perf_counter()
    print("compiling the race step...", end=" ", flush=True)
    for b in (16, 4, 1):
        race.steps(b, rec)
    race.over
    print(f"{time.perf_counter() - t0:.0f} s", flush=True)

    for k in [k for k in plt.rcParams if k.startswith("keymap.") and k != "keymap.quit"]:
        plt.rcParams[k] = []  # matplotlib's own keys (f fullscreen, p pan, arrows: view history) would clash with ours
    plt.rcParams["toolbar"] = "None"
    plt.rcParams["figure.raise_window"] = False
    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.canvas.manager.set_window_title(title or track.name)
    render_track(track, ax, labels=True, title=False)
    track_labels = [(t, t.get_position()) for t in ax.texts]  # corner numbers: drawn only when in view (text is slow)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    rows, cols = track.surface.shape
    (x0, y0), res = track.origin, track.resolution
    whole = (x0, x0 + cols * res, y0, y0 + rows * res)
    colors = plt.cm.tab20(np.arange(cars) % 20) if cars > 1 else np.array([[0.88, 0.02, 0.0, 1.0]])
    bodies = PolyCollection(np.zeros((cars, 4, 2)), facecolors=colors, edgecolors="black", linewidths=0.6, zorder=6)
    ax.add_collection(bodies)
    dots = ax.scatter(np.zeros(cars), np.zeros(cars), s=36, c=colors, edgecolors="black", linewidths=0.6, zorder=7)
    labels = [ax.text(0, 0, str(i + 1), fontsize=8, ha="center", va="bottom", fontweight="bold", zorder=8) for i in range(cars)]
    # the car's readout and the timing tower live in a Qt side panel: matplotlib redraws every glyph every frame, Qt
    # only when the text changes
    from PyQt6 import QtCore, QtGui, QtWidgets

    panel = QtWidgets.QLabel()
    panel.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
    panel.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
    panel.setMargin(8)
    panel.setMinimumWidth(330)
    dock = QtWidgets.QDockWidget("Race", fig.canvas.manager.window)
    dock.setWidget(panel)
    dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
    fig.canvas.manager.window.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

    view = {"paused": False, "follow": True, "car": min(max(follow, 0), cars - 1), "half": 120.0, "speed": speed, "closed": False}

    def on_key(e):
        k = e.key
        if k == " ":
            view["paused"] = not view["paused"]
        elif k == "f":
            view["follow"] = not view["follow"]
        elif k in ("n", "right"):
            view["car"] = (view["car"] + 1) % cars
        elif k in ("p", "left"):
            view["car"] = (view["car"] - 1) % cars
        elif k in ("+", "="):
            view["half"] = max(view["half"] / 1.5, 15.0)
        elif k == "-":
            view["half"] = min(view["half"] * 1.5, 4000.0)
        elif k == "up":
            view["speed"] = min(view["speed"] * 2, 32.0)
        elif k == "down":
            view["speed"] = max(view["speed"] / 2, 0.125)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def draw(race, ended: bool):
        c = race.cars
        x, y, yaw = (np.asarray(a[0], np.float64) for a in (c.x, c.y, c.yaw))
        shown = (race.status[0] == RACING) | (race.status[0] == WRECK)
        hide = lambda a: np.where(shown[:, None, None] if a.ndim == 3 else shown[:, None], a, np.nan)
        outlines = hide(_outlines(x, y, yaw))
        centres = outlines.mean(axis=1)
        bodies.set_verts(outlines)
        dots.set_offsets(centres)
        i = view["car"]
        if view["follow"] and np.isfinite(centres[i]).all():
            w, h = fig.get_size_inches()
            hx, hy = view["half"] * w / h, view["half"]
            lim = (centres[i, 0] - hx, centres[i, 0] + hx, centres[i, 1] - hy, centres[i, 1] + hy)
        else:
            lim = whole
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        for t, (tx, ty) in track_labels:
            t.set_visible(lim[0] < tx < lim[1] and lim[2] < ty < lim[3])
        inside = shown & (centres[:, 0] > lim[0]) & (centres[:, 0] < lim[1]) & (centres[:, 1] > lim[2]) & (centres[:, 1] < lim[3])
        for j, t in enumerate(labels):  # text is the costly part of a frame: only the cars in view get their number
            t.set_visible(bool(inside[j]))
            if inside[j]:
                t.set_position((centres[j, 0], centres[j, 1] + 3.0))
        total = float(track.length)
        lap = lambda j: int(min(max(np.floor(race.line[0, j] / total) + 1, 1), race.target_laps[0]))
        kmh = float(np.hypot(c.vx[0, i], c.vy[0, i])) * 3.6
        tyre = f"{COMPOUNDS[int(race.tyres.compound[0, i])]} {100 * float(race.tyres.life[0, i]):.0f}%"
        flags = "  DRS" if race.drs_open[0, i] else ""
        top = f"Car {i + 1}  P{int(race.timing.place[0, i])}/{cars}  lap {lap(i)}/{int(race.target_laps[0])}  {kmh:3.0f} km/h"
        mode = "PAUSED" if view["paused"] else f"x{view['speed']:g}"
        end = "  race over (close the window)" if ended else ""
        text = f"{_clock(float(race.clock[0]))}  {mode}{end}\n\n{top}\n{tyre}  ERS {10 * float(race.battery[0, i]):.0f}%{flags}\n{_state_text(race, i)}"
        if cars > 1:
            order = np.argsort(race.timing.place[0])
            rows_ = [f"{'>' if j == i else ' '}P{int(race.timing.place[0, j]):>2}  car {j + 1:>2}  L{lap(j):>2}  {COMPOUNDS[int(race.tyres.compound[0, j])][0].upper()} {100 * float(race.tyres.life[0, j]):3.0f}%  {_state_text(race, j)[:8]}" for j in order]
            text += "\n\n" + "\n".join(rows_)
        text += "\n\n" + KEYS.replace("  ", "\n")
        if text != view.get("text"):
            view["text"] = text
            panel.setText(text)

    loop = {"owed": 0.0, "last": time.perf_counter(), "ended": False}

    def tick():
        """One frame: the race steps on by the time since the last frame (x speed), then the view redraws."""
        now = time.perf_counter()
        if not view["paused"] and not loop["ended"]:
            loop["owed"] += (now - loop["last"]) * view["speed"]
            n = 0
            while loop["owed"] >= STEP_S and n < FRAME_STEPS:
                left = int(loop["owed"] / STEP_S)  # compiled blocks of 16 and 4 when behind: one step at a time is slow
                b = next(b for b in (16, 4, 1) if b <= left and n + b <= FRAME_STEPS) if left else 1
                race.steps(b, rec)
                loop["owed"] -= b * STEP_S
                n += b
            if n and (race.over or (seconds and race.clock >= seconds)):
                loop["ended"] = True
            if n >= FRAME_STEPS:
                loop["owed"] = 0.0  # can't keep up at this speed: drop the backlog rather than stutter
        loop["last"] = now
        draw(jax.device_get(race.state.race), loop["ended"])
        fig.canvas.draw_idle()
        if loop["ended"] and close_when_done:
            timer.stop()
            plt.close(fig)

    # Qt's own event loop runs the window (plt.show blocks until it's closed); a timer drives the frames
    timer = fig.canvas.new_timer(interval=FRAME_MS)
    timer.add_callback(tick)
    fig.canvas.mpl_connect("close_event", lambda e: timer.stop())
    timer.start()
    plt.show()
    timer.stop()
    return rec, jax.device_get(race.state.race)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Watch a trained driver race, live, and optionally save an HTML replay.")
    ap.add_argument("--profile", default="middle", choices=PROFILES)
    ap.add_argument("--checkpoint", help="default: the profile's newest best under --runs")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tracks", help="track pool directory; default: generate the eval track")
    ap.add_argument("--track", type=int, default=0, help="eval track 0-9")
    ap.add_argument("--width", help='"normal", 26 or 20 (default: the width the checkpoint was saved at)')
    ap.add_argument("--cars", type=int, default=1, help="1 (alone) up to 22, all driven by this driver")
    ap.add_argument("--laps", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this much race time (0: the whole race)")
    ap.add_argument("--respawn", action="store_true", help="crashed cars rejoin from the pit exit (as in training)")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time (up / down arrows change it while watching)")
    ap.add_argument("--follow", type=int, default=1, help="the car to follow (n / p while watching)")
    ap.add_argument("--save", nargs="?", const="auto", help="also write the race as an HTML replay (optionally: where)")
    ap.add_argument("--no-window", action="store_true", help="no live window, only the HTML replay (e.g. on the cluster)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"], help="cpu (default) leaves the GPU to a training run")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.device == "cpu":  # one race is quick on the CPU, and doesn't queue behind (or slow down) training on the GPU
        jax.config.update("jax_platforms", "cpu")

    path = Path(a.checkpoint) if a.checkpoint else newest_best(a.runs, a.profile)
    info = checkpoint.meta(path)
    s = load(info.get("profile", a.profile), info.get("preset", "laptop"))
    agent, _ = checkpoint.load(path, tqc.init(jax.random.PRNGKey(0), s.tqc))
    width = a.width or info.get("width", "normal")
    base = eval_track(a.tracks, a.track)
    track = next(iter(widened([base], width)), None)
    if track is None:
        print(f"{base.name} has no copy at {width} m: racing on the normal track")
        track, width = base, "normal"
    title = f"{s.profile} ({info.get('env_steps', 0):,} steps, stage {info.get('stage', '?')}) on {base.name}, width {width}"
    window = not a.no_window and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not window and not a.no_window:
        print("no display: recording the race to an HTML replay instead")
    print(f"{'watching' if window else 'recording'}: {title} from {path}", flush=True)
    if window:
        rec, race = live(track, agent.actor, s.tqc, a.cars, a.laps, a.respawn, a.seconds, a.seed, title, a.speed, a.follow - 1, save=bool(a.save))
    else:
        rec, race = record(track, agent.actor, s.tqc, a.cars, a.laps, a.respawn, a.seconds, a.seed, title)

    laps = np.floor(race.line[0] / track.length).astype(int)
    finished, out = race.finish_pos[0] > 0, race.dnf[0]
    best = np.where(np.isfinite(race.best_lap[0]), race.best_lap[0], np.nan)
    print(f"race time {max(float(race.clock[0]), 0):.0f} s: {int(finished.sum())} of {a.cars} finished, {int(out.sum())} crashed; most laps {max(int(laps.max()), 0)} of {a.laps}")
    if np.isfinite(best).any():
        print(f"best lap {np.nanmin(best):.1f} s (the lap simulation's ideal: {track.lap['lap_time_s']:.1f} s)")
    if rec is not None and (a.save or not window):
        dest = Path(a.save) if a.save and a.save != "auto" else Path(a.runs) / "watch" / f"{s.profile}-{base.name}-{width}.html"
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_html(dest, rec.replay())
        print(f"replay written: {dest}  (open it in a browser: xdg-open {dest})")


if __name__ == "__main__":
    main()
