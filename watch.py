"""
Watch trained drivers race in the render window, greedily. --profiles are
given to the cars in turn; every car drives with its own profile's
checkpoint (--tag, or PROFILE:TAG per entry, e.g. aggressive:latest) and is
painted in its profile's colour - a lighter or darker shade for a second or
third checkpoint of the same profile - with its number beside it. Cars are stock unless --designs gives car pairs the teams' designs
from configs/designs.toml (two cars per team, in the file's order).
The grid order is shuffled every race (each car keeps its profile and
design), so no style always starts on pole. A panel on the left shows the
live standings: places gained (green) or lost (red) since the start, grey if
the same, or "crashed", and each car's design. The camera follows the race
leader (a pass or a crash of the leader switches to the new one) until you
pick a car yourself (right-click: next car), drag the view or show the whole
map (middle-click); L follows the leader again. A panel at the bottom right
shows what the followed car's network outputs: steering, throttle and brake
bars, DRS and ERS lights and its battery, and its gap at the DRS / ERS
detection point (the final corner's exit: under 1 s there arms DRS and the
faster recharge until the next pass).

    uv run --with torch watch.py --cars 9 --profiles middle,aggressive,defensive --map Monza
    uv run --with torch watch.py --cars 22 --profiles middle,aggressive,aggressive:latest --map Monza
    uv run --with torch watch.py --cars 22 --designs --map Spa
    uv run --with torch watch.py --list-maps
"""

import argparse
import os

import numpy as np

from racer import config, design
from racer.console import hex_color, lap_time
from racer.records import LapRecords
from racer.track import SV, available_maps


def parse_args(cfg):
    w = cfg.watch
    p = argparse.ArgumentParser(description="Watch trained drivers race.")
    p.add_argument("--tag", default=w.tag, help="checkpoint per profile: best, latest, an update count, ...")
    p.add_argument("--profiles", default=",".join(w.profiles), metavar="PROFILE[:TAG],...",
                   help=f"drivers given to the cars in turn ({', '.join(cfg.profile_names)}); PROFILE:TAG picks "
                        f"another checkpoint than --tag, so one profile can race several of its checkpoints")
    p.add_argument("--cars", type=int, default=w.cars)
    p.add_argument("--map", default=w.map, help="track name, or 'random'")
    p.add_argument("--episodes", type=int, default=w.episodes)
    p.add_argument("--seconds", type=float, default=None, help="race length (default: the training length)")
    p.add_argument("--speed", type=float, default=w.playback_speed,
                   help="playback speed: 1 = real time, 3 = 3x, 0 = as fast as possible")
    p.add_argument("--random-spawn", action="store_true", default=w.random_spawn,
                   help="spread the cars around the track instead of the grid")
    p.add_argument("--designs", nargs="?", const=cfg.design.designs_file, default=None, metavar="FILE",
                   help=f"car pairs get the teams' designs (default file {cfg.design.designs_file}); stock otherwise")
    p.add_argument("--window", default="max", metavar="max|WxH",
                   help="render window: 'max' fills the screen (default), or a size like 1920x1080 "
                        "(the gym's own window is 800x800)")
    p.add_argument("--list-maps", action="store_true", help="print the available maps and exit")
    return p.parse_args()


def rgb(hex_color_str):
    h = hex_color_str.lstrip("#")
    return tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))


def shade(hex_color_str, k):
    """The k-th checkpoint of a profile: its colour, then lighter, darker,
    much lighter, much darker (never red: that's a crashed car)."""
    f = (0.0, 0.45, -0.4, 0.7, -0.6)[k % 5]
    target = 255 if f > 0 else 0
    r, g, b = (round(c + (target - c) * abs(f)) for c in rgb(hex_color_str))
    return f"#{r:02x}{g:02x}{b:02x}"


def parse_window(spec):
    """'max' -> None (fill the screen), '1920x1080' -> (1920, 1080)."""
    if spec == "max":
        return None
    try:
        width, height = (int(x) for x in spec.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--window: expected 'max' or WIDTHxHEIGHT like 1920x1080, got {spec!r}") from None
    if width < 300 or height < 300:
        raise SystemExit("--window: at least 300x300")
    return width, height


def size_window(renderer, size):
    """The gym opens a square 800 x 800 window: maximize it (size None) or
    resize it. The camera keeps its zoom, so a wider window shows more track.
    The gym's own lap label only follows car 0 - hidden."""
    if size is None:
        renderer.window.showMaximized()
    else:
        renderer.window.resize(*size)
    if getattr(renderer, "lap_label", None) is not None:
        renderer.lap_label.hide()
    renderer.app.processEvents()


def undo_vscode_snap_env():
    """VS Code installed as a snap points its terminals' GTK / GLib at the
    snap's own copies (keeping the originals in *_VSCODE_SNAP_ORIG), and the
    render window then aborts with "GLib-GIO-ERROR ... does not contain a key
    named 'antialiasing'". Put the originals back before the window opens."""
    for key in [k for k in os.environ if k.endswith("_VSCODE_SNAP_ORIG")]:
        name, orig = key[:-len("_VSCODE_SNAP_ORIG")], os.environ.pop(key)
        if orig:
            os.environ[name] = orig
        else:
            os.environ.pop(name, None)


def paint_cars(renderer, colors):
    """Each car in its profile's colour. The gym colours cars by index and
    paints a crashed car red, so this runs before every race."""
    if renderer is None:
        return
    renderer.car_colors = list(colors)                 # cars not drawn yet pick these up
    for car, c in zip(renderer.cars or [], colors):
        car.color = c
        if hasattr(car, "rgba"):                       # OpenGL renderer: rebuilt on the next draw
            car.rgba = np.array([[v / 255 for v in c] + [car.opacity]] * 2)
            car.scale = None
        elif hasattr(car, "chassis") and car.chassis is not None:   # Qt renderer
            car.chassis.setBrush(c)


def race_order(env):
    """Cars still racing, leader first, by the env's race order (the one race
    points are paid on: a pair swaps only once one leads by overtake_margin,
    so cars side by side don't flicker); ties by position score."""
    idx = np.flatnonzero(~env.done)
    ahead = env.place_ahead[np.ix_(idx, idx)].sum(axis=0)          # racing cars ahead of each
    return idx[np.lexsort((idx, -env.position_scores()[idx], ahead))]


class Standings:
    """Live standings on the left of the render window - per car: places
    gained (green) / lost (red) since the start, grey if the same, or why it
    is out; its position; its number and profile colour - plus each car's
    number next to it on the track."""

    OUT = {"collision": "crashed", "contact": "crashed", "no_progress": "stuck", "sim_error": "error"}

    def __init__(self, renderer, cfg, num_cars):
        import pyqtgraph.opengl as gl
        from OpenGL.GL import GL_BLEND, GL_DEPTH_TEST, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA
        from PyQt6 import QtCore, QtGui, QtWidgets
        self.cfg = cfg
        self.label = QtWidgets.QLabel(renderer.view)
        self.label.setFont(QtGui.QFont("Monospace", 10))
        self.label.setStyleSheet("color: #eeeeee; background-color: rgba(20, 20, 20, 200); padding: 6px; "
                                 "border-radius: 4px;")
        self.label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.label.move(10, 40)                        # below the gym's FPS counter
        self.label.show()
        self.tags = []
        # Normal alpha blending: the default (additive) adds the text's colour
        # onto the white track, so black numbers would be invisible.
        blend = {GL_DEPTH_TEST: False, GL_BLEND: True, "glBlendFunc": (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)}
        for i in range(num_cars):
            tag = gl.GLTextItem(text=str(i), color=(0, 0, 0, 255), glOptions=blend,
                                font=QtGui.QFont("Sans", 10, QtGui.QFont.Weight.Bold))
            tag.setDepthValue(10)                      # drawn after (over) the cars
            renderer.view.addItem(tag)
            self.tags.append(tag)
        self.html = None

    def start(self, env, labels, colors, race):
        """A new race: the grid order is what places gained / lost count from.
        labels, colors: per car, its driver's name and colour."""
        self.labels, self.colors, self.race = labels, colors, race
        self.designs = [design.text(d) for d in env.designs]
        self.start_pos = {int(car): k + 1 for k, car in enumerate(race_order(env))}
        for tag in self.tags:
            tag.setVisible(True)
        self.update(env)

    def update(self, env):
        racing = ~env.done
        for i, tag in enumerate(self.tags):
            if racing[i]:
                x, y, yaw = env.std[i, 0], env.std[i, 1], env.std[i, 4]
                side = 2.5                             # m to the car's left, beside it rather than on it
                tag.setData(pos=np.array([x - side * np.sin(yaw), y + side * np.cos(yaw), 1.0]))
            elif env.termination_reason[i] != "timeout":
                tag.setVisible(False)                  # out of the race: off the track
        finished = [i for i in range(env.num_agents) if env.finish_position[i] is not None]
        if finished:
            classified = [(env.finish_position[i], i) for i in sorted(finished, key=lambda i: env.finish_position[i])]
        else:
            classified = [(k + 1, int(i)) for k, i in enumerate(race_order(env))]
        out = sorted((i for i in range(env.num_agents) if not racing[i] and env.finish_position[i] is None),
                     key=lambda i: -env.progress[i])
        rows = []
        for pos, i in classified:
            gained = self.start_pos[i] - pos
            info, color = (f"▲{gained}", "#4caf50") if gained > 0 else (f"▼{-gained}", "#ff5252") if gained < 0 \
                else ("–", "#9e9e9e")
            rows.append(self._row(info, color, f"P{pos}", i))
        for i in out:
            rows.append(self._row(self.OUT.get(env.termination_reason[i], "out"), "#ff5252", "–", i))
        secs = int(env.steps * self.cfg.sim.control_dt)
        title = f"Race {self.race} · {secs // 60}:{secs % 60:02d}" + (" · final" if env.done.all() else "")
        html = (f"<b>{title}</b><table cellspacing='0' cellpadding='1'>" + "".join(rows) + "</table>")
        if html != self.html:                          # rebuild the text only when something changed
            self.html = html
            self.label.setText(html)
            self.label.adjustSize()

    def _row(self, info, color, pos, i):
        return (f"<tr><td align='right' style='color:{color}'>{info}</td><td width='8'></td>"
                f"<td align='right'><b>{pos}</b></td><td width='8'></td>"
                f"<td><span style='color:{self.colors[i]}'>■</span> #{i}</td><td width='6'></td>"
                f"<td style='color:#cccccc'>{self.designs[i]}</td><td width='6'></td>"
                f"<td style='color:#aaaaaa'>{self.labels[i]}</td></tr>")


class Camera:
    """Follows the race leader - a pass or a crash of the leader switches to
    the new one - until you take over: right-click (the next car), drag (pan)
    or middle-click (the whole map). L follows the leader again."""

    def __init__(self, renderer):
        from PyQt6 import QtCore
        self.renderer = renderer
        self.leader_mode = True
        self.following = None                    # the car this camera set last
        key_l = QtCore.Qt.Key.Key_L
        for widget in (renderer.view, renderer.window):
            original = widget.keyPressEvent

            def key_pressed(event, original=original):
                if event.key() == key_l:
                    self.leader_mode, self.following = True, None
                else:
                    original(event)
            widget.keyPressEvent = key_pressed

    def update(self, env):
        r = self.renderer
        if self.leader_mode and self.following is not None and \
                (r.agent_to_follow != self.following or not getattr(r, "focused", True)):
            self.leader_mode = False             # right-click, middle-click or a drag: you took over
        if self.leader_mode:
            order = race_order(env)
            if len(order):
                self.following = r.agent_to_follow = int(order[0])
                r.focused = True


class Telemetry:
    """Bottom right of the render window: what the followed car is doing this
    step (right-click cycles the followed car, middle-click shows the whole
    map). The network's five outputs as the car used them - steering (the
    share of the steering available right now, left +), throttle and brake
    bars, DRS and ERS lights (white ring: the network asks for it; filled:
    DRS open / ERS deploying; amber: DRS available; dark amber: armed at the
    detection point, waiting for a zone) - plus the battery, the gap at the
    last detection pass, the chase recharge and the raw tanh outputs."""

    W, H = 310, 214

    def __init__(self, renderer, camera):
        from PyQt6 import QtCore, QtGui, QtWidgets
        self.renderer, self.camera, self.QtCore, self.QtGui = renderer, camera, QtCore, QtGui
        self.data = None
        panel = self

        class Panel(QtWidgets.QWidget):
            def paintEvent(self, event):
                painter = QtGui.QPainter(self)
                panel.paint(painter)
                painter.end()

        self.widget = Panel(renderer.view)
        self.widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.widget.resize(self.W, self.H)
        self.widget.show()

    def update(self, env, labels, colors, raw, used):
        """raw: the network's outputs this step; used: what the car used
        (env.step's info["actions"]: dead zones, launch override)."""
        i = self.renderer.agent_to_follow
        view = self.renderer.view
        h = self.H if i is not None else 54                  # the whole-map view only shows a hint
        self.widget.resize(self.W, h)
        self.widget.move(view.width() - self.W - 10, view.height() - h - 10)
        if i is None or i >= env.num_agents:
            self.data = None
        else:
            racing = not env.done[i]
            order = race_order(env)
            self.data = dict(
                car=i, label=labels[i], color=colors[i], racing=racing, reason=env.termination_reason[i],
                place=int(np.flatnonzero(order == i)[0]) + 1 if racing and i in order else env.finish_position[i],
                kmh=float(env.std[i, SV]) * 3.6, raw=np.array(raw[i]), used=np.array(used[i]),
                drs_available=bool(env.drs_available[i] > 0), drs_open=bool(env.drs_open[i] > 0),
                ers_on=bool(env.ers_on[i] > 0), battery=float(env.battery[i] / env.car.ers_capacity[i]),
                close=bool(env.ers_close[i]), drs_armed=bool(env.drs_armed[i]), detect_gap=float(env.detect_gap[i]),
                launch=env.steps < env.launch_steps)
        self.widget.update()

    def paint(self, p):
        QtGui, Qt = self.QtGui, self.QtCore.Qt
        color = QtGui.QColor
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color(20, 20, 20, 205))
        p.drawRoundedRect(0, 0, self.W, self.widget.height(), 6, 6)
        p.setFont(QtGui.QFont("Sans", 9))
        d = self.data
        if d is None:
            p.setPen(color("#9e9e9e"))
            p.drawText(12, 24, "Whole map - right-click the track to follow a car")
            p.drawText(12, 42, "(again: the next car) · L: follow the race leader")
            return

        # Header: the car, its driver, place and speed.
        p.setBrush(color(d["color"]))
        p.drawRect(12, 9, 10, 10)
        p.setPen(color("#eeeeee"))
        p.drawText(28, 19, f"#{d['car']} {d['label']}")
        right = (f"P{d['place']} · {d['kmh']:.0f} km/h" if d["racing"] else
                 Standings.OUT.get(d["reason"], "finished") if d["reason"] != "timeout" else f"finished P{d['place']}")
        p.drawText(self.W - 12 - p.fontMetrics().horizontalAdvance(right), 19, right)

        u = d["used"]
        x0, w = 72, 190
        bars = (("steer", u[0], "#e0e0e0", True), ("throttle", max(u[1], 0.0), "#4caf50", False),
                ("brake", max(u[2], 0.0), "#ef5350", False))
        for row, (name, value, fill, centred) in enumerate(bars):
            y = 30 + 22 * row
            p.setPen(color("#bdbdbd"))
            p.drawText(12, y + 11, name)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color(60, 60, 60))
            p.drawRect(x0, y, w, 13)
            p.setBrush(color(fill))
            if centred:                                  # left of centre = steering left (+)
                mid, half = x0 + w / 2, w / 2 * min(abs(value), 1.0)
                p.drawRect(int(mid - half if value > 0 else mid), y, int(half), 13)
                p.setBrush(color("#9e9e9e"))
                p.drawRect(int(mid), y - 2, 1, 17)
            else:
                p.drawRect(x0, y, int(w * min(value, 1.0)), 13)
            p.setPen(color("#eeeeee"))
            text = f"{value:+.2f}" if centred else f"{value:.0%}"
            p.drawText(x0 + w + 8, y + 11, text)

        # Lights: white ring = asked for, filled = on (DRS amber = available).
        def light(cx, name, asked, on_color, state_text):
            p.setPen(QtGui.QPen(color("#ffffff"), 2.5) if asked else QtGui.QPen(color(90, 90, 90), 1))
            p.setBrush(color(on_color))
            p.drawEllipse(cx - 13, 106, 26, 26)
            p.setPen(color("#bdbdbd"))
            p.drawText(cx - p.fontMetrics().horizontalAdvance(name) // 2, 147, name)
            p.drawText(cx - p.fontMetrics().horizontalAdvance(state_text) // 2, 161, state_text)

        drs = "#4caf50" if d["drs_open"] else "#ffb300" if d["drs_available"] else "#6d5200" if d["drs_armed"] else "#424242"
        light(34, "DRS", u[3] > 0, drs, "open" if d["drs_open"] else "available" if d["drs_available"]
              else "armed" if d["drs_armed"] else "off")
        light(96, "ERS", u[4] > 0, "#42a5f5" if d["ers_on"] else "#424242", "deploying" if d["ers_on"] else "off")
        p.setPen(color("#bdbdbd"))
        p.drawText(140, 115, "battery")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color(60, 60, 60))
        p.drawRect(140, 121, 122, 12)
        p.setBrush(color("#42a5f5"))
        p.drawRect(140, 121, int(122 * min(max(d["battery"], 0.0), 1.0)), 12)
        p.setPen(color("#eeeeee"))
        p.drawText(270, 132, f"{d['battery']:.0%}")
        if d["launch"]:
            p.setPen(color("#ffb300"))
            p.drawText(140, 153, "launch: full throttle forced")
        elif np.isfinite(d["detect_gap"]):               # the last pass of the detection point (final corner exit)
            p.setPen(color("#bdbdbd"))
            p.drawText(140, 153, f"detection: {d['detect_gap']:.2f} s behind")
        if d["close"]:                                   # chasing: the battery recharges faster until the next pass
            p.setPen(color("#66bb6a"))
            p.drawText(140, 167, "chase recharge ×1.5 this lap")

        r = d["raw"]
        p.setPen(color("#8a8a8a"))
        p.drawText(12, 180, f"network  steer {r[0]:+.2f} · throttle {r[1]:+.2f} · brake {r[2]:+.2f}")
        p.drawText(12, 194, f"              DRS {r[3]:+.2f} · ERS {r[4]:+.2f}   (ring = asked for)")
        p.setPen(color("#ffb300") if self.camera.leader_mode else color("#8a8a8a"))
        p.drawText(12, 209, "camera: race leader · right-click: pick a car" if self.camera.leader_mode
                   else "camera: your pick · L: follow the race leader")


def main():
    cfg = config.load()
    args = parse_args(cfg)
    window = parse_window(args.window)
    maps = available_maps(cfg)
    if args.list_maps:
        print("\n".join(sorted(maps)))
        return
    rng = np.random.default_rng()
    map_name = str(rng.choice(sorted(maps))) if args.map == "random" else args.map
    if map_name not in maps:
        raise SystemExit(f"unknown map {map_name!r} - available: {', '.join(sorted(maps))}")
    # Drivers: (profile, checkpoint tag); the k-th checkpoint of a profile gets the k-th shade of its colour.
    entries = []
    for item in [x.strip() for x in args.profiles.split(",") if x.strip()]:
        prof, _, tag = item.partition(":")
        try:
            cfg.profile(prof)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        entries.append((prof, tag or args.tag))
    keys = list(dict.fromkeys(entries))
    label = {k: k[0] if k[1] == args.tag else f"{k[0]}:{k[1]}" for k in keys}
    color = {k: shade(cfg.color(k[0]), [x for x in keys if x[0] == k[0]].index(k)) for k in keys}
    car_drivers = [entries[i % len(entries)] for i in range(args.cars)]
    car_designs = [design.stock(cfg)] * args.cars
    if args.designs:
        try:
            teams = design.read_designs(cfg, args.designs)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        car_designs = [teams[(i // 2) % len(teams)][1] for i in range(args.cars)]

    from racer.agent import read_actor_weights
    from racer.env import RaceEnv
    from racer.policy import FrozenPolicy
    drivers = {}
    width = max(11, *(len(v) for v in label.values()))
    for key in dict.fromkeys(car_drivers):
        prof, tag = key
        path = cfg.checkpoint_path(prof, tag)
        if not path.exists():
            raise SystemExit(f"no checkpoint at '{config.shown(path)}' - train '{prof}' first, or pick another tag")
        params, data = read_actor_weights(path, cfg)
        drivers[key] = FrozenPolicy(params)
        meta = data.get("meta", {})
        print(f"[watch] {hex_color(f'{label[key]:<{width}}', color[key])} {config.shown(path)} "
              f"({data['model'].get('total_it', '?')} updates"
              + (f", eval {meta['score']:.1f} by {meta.get('best_by')}" if "score" in meta else "") + ")")

    undo_vscode_snap_env()
    env = RaceEnv(cfg, args.cars, map_name, [k[0] for k in car_drivers], render=True)
    seconds = args.seconds if args.seconds is not None else cfg.episode_steps * cfg.sim.control_dt
    env.max_steps = int(round(seconds / cfg.sim.control_dt))
    renderer = env.unwrapped.renderer
    if renderer is not None:
        size_window(renderer, window)
        # The renderer sleeps to hold render_fps (one frame per control step =
        # real time); the physics is stepped exactly as in training.
        renderer.render_fps = float("inf") if args.speed <= 0 else args.speed / cfg.sim.control_dt
    standings = Standings(renderer, cfg, args.cars) if renderer is not None else None
    camera = Camera(renderer) if renderer is not None else None
    telemetry = Telemetry(renderer, camera) if renderer is not None else None
    spawn = "random" if args.random_spawn else "grid"
    print(f"[watch] {map_name} · {args.cars} car(s) · {spawn} start · {seconds:g} s races · playback "
          + (f"{args.speed:g}x" if args.speed > 0 else "max speed") + " · "
          + " · ".join(hex_color(f"■ {label[k]}", color[k]) for k in drivers) + " · crashed cars turn red")
    if args.designs:
        used = teams[:(args.cars + 1) // 2] if args.cars <= 2 * len(teams) else teams
        print(f"[watch] two cars each from {args.designs}: "
              + ", ".join(f"{name} {design.text(d)}" for name, d in used))
    records = {prof: LapRecords.load(cfg, prof) for prof, _ in drivers}
    dt = cfg.sim.control_dt
    for ep in range(1, args.episodes + 1):
        grid = rng.permutation(args.cars)               # a new grid order; every car keeps its driver and design
        car_drivers = [car_drivers[k] for k in grid]
        car_designs = [car_designs[k] for k in grid]
        env.set_profiles([k[0] for k in car_drivers])
        env.set_designs(np.stack(car_designs))
        groups = {key: np.array([i for i, k in enumerate(car_drivers) if k == key]) for key in drivers}
        obs = env.reset(rng, spawn=spawn)
        paint_cars(renderer, [rgb(color[k]) for k in car_drivers])
        if standings:
            standings.start(env, [label[k] for k in car_drivers], [color[k] for k in car_drivers], ep)
            camera.update(env)
        ret = np.zeros(env.num_agents)
        while not env.done.all():
            actions = np.zeros((env.num_agents, cfg.action_dim), dtype=np.float32)
            for key, idx in groups.items():
                if len(idx):
                    actions[idx] = drivers[key].act(obs[idx])
            obs, r, _, _, info = env.step(actions)
            ret += r
            if standings:
                standings.update(env)
                camera.update(env)
                telemetry.update(env, [label[k] for k in car_drivers], [color[k] for k in car_drivers], actions,
                                 info["actions"])
            env.render()

        print(f"\n[watch] race {ep} on {map_name}: {env.steps * dt:.0f} s")
        order = sorted(range(env.num_agents), key=lambda i: (env.finish_position[i] is None,
                                                              env.finish_position[i] or 0, -env.progress[i]))
        print(f"  {'pos':>4} {'car':>4} {'driver':<{width}} {'design':<10} {'distance':>9} {'laps':>5} {'best lap':>9} "
              f"{'avg':>5} {'top':>5} {'pass':>4} {'lost':>4} {'tow':>4} {'return':>7}  end")
        best_laps = env.best_laps()
        for i in order:
            pos = f"P{env.finish_position[i]}" if env.finish_position[i] else "-"
            lap = lap_time(best_laps[i], unit="") if best_laps[i] else "-"
            key = car_drivers[i]
            print(f"  {pos:>4} {i:>4} {hex_color(f'{label[key]:<{width}}', color[key])} {design.text(env.designs[i]):<10} "
                  f"{env.progress[i]:>7,.0f} m "
                  f"{env.laps()[i]:>5.2f} {lap:>9} {env.avg_speed()[i]:>5.1f} {env.max_speed[i]:>5.1f} "
                  f"{env.overtakes[i]:>4} {env.times_overtaken[i]:>4} "
                  f"{env.tow_steps[i] / max(env.racing_steps[i], 1):>4.0%} {ret[i]:>7.1f}  {env.termination_reason[i]}")
        print("  (avg / top speed in m/s · pass = passes made · lost = times passed · tow = time in a slipstream)")
        for key in drivers:
            laps = [best_laps[i] for i, k in enumerate(car_drivers) if k == key and best_laps[i]]
            rec = records[key[0]].get(map_name)
            if laps or rec:
                here = f"fastest lap here {lap_time(min(laps), unit='')}" if laps else "no complete lap here"
                best = f"{key[0]}'s record here {lap_time(rec['lap_s'], unit='')}" if rec else f"no {map_name} record yet"
                print(f"  {hex_color(label[key], color[key])}: {here} · {best}")
    env.close()


if __name__ == "__main__":
    main()
