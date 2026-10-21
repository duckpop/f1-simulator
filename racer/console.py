"""
Console output for training: number formatting, the one summary of the
training window (summarize - the dashboard, the printed panel and
metrics.csv all use it), the startup header, printed status panels and eval
lines, and the live rich dashboard (when stdout is a terminal and rich is
installed).

Colours only on a terminal without NO_COLOR, so redirected logs stay plain.
"""

import importlib.util
import os
import re
import sys
import time
from collections import Counter, deque

import numpy as np

from racer.scoring import win_rate

COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
RULE_WIDTH = 78
WRAP_WIDTH = 100
REASON_ORDER = ("timeout", "collision", "contact", "no_progress", "sim_error")


def _c(code):
    return (lambda s: f"\033[{code}m{s}\033[0m") if COLOR else (lambda s: str(s))


bold, dim = _c("1"), _c("2")
red, green, yellow, cyan = _c("31"), _c("32"), _c("33"), _c("36")


def hex_color(text, hex_rgb):
    """text in a #rrggbb colour (terminals only)."""
    if not COLOR:
        return text
    r, g, b = (int(hex_rgb.lstrip("#")[k:k + 2], 16) for k in (0, 2, 4))
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def visible_len(s):
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


# -------------------------------------------------------------------------
# Formatting
# -------------------------------------------------------------------------

def num(x, digits=2):
    """1234567 -> '1.23M', 27350 -> '27.4k', 512 -> '512'."""
    x = float(x)
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= div:
            return f"{x / div:.{digits}f}".rstrip("0").rstrip(".") + suffix
    return f"{x:.0f}"


def duration(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{int(seconds // 3600)} h {int(seconds % 3600 // 60):02d} min"


def bar(fraction, width=20):
    fraction = min(max(float(fraction), 0.0), 1.0)
    filled = int(round(fraction * width))
    return "█" * filled + dim("░" * (width - filled))


def meters(x):
    return f"{x:,.0f} m"


def pct(x):
    return f"{x:.0%}"


def score_text(x, kind="distance"):
    """An eval score: metres, or race points per car."""
    return f"{x:.1f} pts/car" if kind == "points" else meters(x)


def lap_time(seconds, unit=" s"):
    m, s = divmod(float(seconds), 60.0)
    return f"{int(m)}:{s:06.3f}" if m else f"{s:.3f}{unit}"


def end_label(reason):
    return {"collision": "crash (wall)", "contact": "crash (car)", "no_progress": "stuck", "sim_error": "sim error",
            "timeout": "survived"}.get(reason, str(reason))


# -------------------------------------------------------------------------
# The one summary of the training window
# -------------------------------------------------------------------------

def summarize(recent):
    """Every number shown or logged about the last training car-episodes
    (dicts made by train.py from the workers' episode stats)."""
    n = len(recent)
    col = lambda k: [e[k] for e in recent]
    progress, reasons = col("progress"), Counter(col("reason"))
    lapped = sum(x >= 1.0 for x in col("laps"))
    return {
        "n": n, "median": float(np.median(progress)), "mean": float(np.mean(progress)),
        "best": float(np.max(progress)), "lapped": lapped, "lap_rate": lapped / n, "reasons": reasons,
        "survived": reasons["timeout"] / n, "crashed": (reasons["collision"] + reasons["contact"]) / n,
        "car_crashed": reasons["contact"] / n, "stuck": reasons["no_progress"] / n,
        "speed": float(np.mean(col("avg_speed"))), "top_speed": float(np.max(col("max_speed"))),
        "passes": float(np.mean(col("overtakes"))), "close_calls": float(np.mean(col("close_calls"))),
        "passed": float(np.mean(col("passed"))), "passed_by_opp": float(np.mean(col("passed_by_opp"))),
        "purple": float(np.mean(col("purple"))), "tow": float(np.mean(col("tow"))),
        "drs": float(np.mean(col("drs"))), "ers": float(np.mean(col("ers"))),
        "battery": float(np.mean(col("battery"))), "return": float(np.mean(col("ret"))),
        "by_cars": {c: float(np.median([e["progress"] for e in recent if e["cars"] == c]))
                    for c in sorted(set(col("cars")))},
    }


def metrics_row(*, run, elapsed, received, updates, steps_per_s, updates_per_s, summary, best_eval, best_by,
                losses, ram, duels):
    """One metrics.csv row: the same numbers as the status panel."""
    row = dict(run=run, wall_time=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_s=elapsed, env_steps=received,
               updates=updates, steps_per_s=steps_per_s, updates_per_s=updates_per_s,
               best_eval=best_eval if best_eval > float("-inf") else None, best_by=best_by)
    if summary:
        s = summary
        row.update(median_distance_m=s["median"], mean_distance_m=s["mean"], best_distance_m=s["best"],
                   lap_rate=s["lap_rate"], survival_rate=s["survived"], crash_rate=s["crashed"],
                   car_crash_rate=s["car_crashed"], stuck_rate=s["stuck"], avg_speed_ms=s["speed"],
                   top_speed_ms=s["top_speed"], passes_per_car=s["passes"], passed_per_car=s["passed"],
                   passed_by_opp_per_car=s["passed_by_opp"], tow_frac=s["tow"],
                   close_calls_per_car=s["close_calls"], purple_sectors_per_car=s["purple"],
                   drs_open_frac=s["drs"], ers_deploy_frac=s["ers"], battery_mean=s["battery"])
    row.update(critic_loss=losses[0], actor_loss=losses[1], avg_q=losses[2])
    if ram is not None:
        row.update(ram_used_mb=ram[0], ram_free_mb=ram[1])
    overall, per = win_rate(duels)
    if overall is not None:
        row.update(win_rate_vs_opponents=overall,
                   win_rates=";".join(f"{name}={share:.3f}" for name, (share, _) in sorted(per.items())))
    return row


# -------------------------------------------------------------------------
# Reporter: printed output, or the live dashboard
# -------------------------------------------------------------------------

def _rule(title="", right=""):
    left = f"── {title} " if title else "──"
    right = f" {right} ──" if right else ""
    return cyan(left + "─" * max(3, RULE_WIDTH - len(left) - len(right)) + right)


def _row(label, text):
    return f"  {bold(f'{label:<9}')} {text}"


def _print(*lines):
    print("\n".join(lines), flush=True)


class Reporter:
    """Everything train.py shows. kind: what the eval score is ("distance"
    or "points"); eval_cars: the eval grid size."""

    def __init__(self, cfg, profile, kind, eval_cars):
        self.cfg, self.profile, self.kind, self.eval_cars = cfg, profile, kind, eval_cars
        self.live = None

    def start_dashboard(self):
        """The live dashboard on a terminal with rich installed; printed
        panels otherwise."""
        if not (self.cfg.run.dashboard and sys.stdout.isatty()):
            return
        if importlib.util.find_spec("rich") is None:
            self.info("tip: `uv run --with torch --with rich train.py ...` shows a live dashboard instead of "
                      "a status panel every minute")
            return
        import atexit
        self.live = Dashboard(self.cfg, self.profile, self.kind, self.eval_cars)
        atexit.register(self.live.stop)

    # ---- one-liners -----------------------------------------------------------
    def header(self, rows):
        _print("", cyan("━━ F1 racers · TD3 training " + "━" * (RULE_WIDTH - 28)),
               *[_row(label, text) for label, text in rows], cyan("━" * RULE_WIDTH))

    def note(self, text):
        if self.live:
            return self.live.event("yellow", f"! {text}")
        _print(f"  {yellow('!')} {text}")

    def info(self, text):
        if self.live:
            return self.live.event("dim", text)
        _print(f"  {dim('·')} {dim(text)}")

    def ok(self, text):
        if self.live:
            return self.live.event("green", f"✓ {text}")
        _print(f"  {green('✓')} {text}")

    def saved(self, paths):
        if self.live:
            return self.live.saved(paths)
        self.info("saved " + ", ".join(paths))

    def lap_records(self, new):
        """new: [(map, lap, old lap or None)] - one line per map, its best."""
        best = {}
        for m, lap, old in new:
            if m not in best or lap < best[m][0]:
                best[m] = (lap, best[m][1] if m in best else old)
        for m, (lap, old) in sorted(best.items()):
            gain = f" (−{old - lap:.3f} s)" if old is not None else " (first)"
            if self.live:
                self.live.event("green", f"⚑ lap record {m} {lap_time(lap)}{gain}")
            else:
                _print(f"  {green('⚑')} lap record {m} {lap_time(lap)}{dim(gain)}")

    def finished(self, text, detail=""):
        if self.live:
            self.live.stop()
        _print("", cyan("━" * RULE_WIDTH), f"  {green('■')} {text}", *([f"    {dim(detail)}"] if detail else []), "")

    # ---- evals ---------------------------------------------------------------------
    def eval_round(self, rnd, score, elapsed, new_best, saved, baseline):
        """rnd: scoring.score_round's result."""
        if self.live:
            return self.live.eval_round(rnd, score, elapsed, new_best, saved, baseline)
        for race in rnd["races"]:
            self._eval_race(race, elapsed)
        when = dim(f"[{duration(elapsed):>8}]")
        best = f"  {green('★ new best')}" if new_best else ""
        if baseline:
            best += "  " + cyan("baseline: the loaded driver, before any training - this run has to beat it")
        per_map = rnd["points"] if self.kind == "points" else rnd["distance"]
        maps = " · ".join(f"{m} {score_text(v, self.kind)}" for m, v in per_map.items())
        how = (f"(race points per car vs the opponents, averaged over maps: {maps}; mean distance "
               f"{meters(rnd['round_distance'])})" if self.kind == "points"
               else f"(average of the per-map mean distances: {maps})")
        _print(f"  {when} {bold('eval round')}  score {bold(score_text(score, self.kind))} {dim(how)}{best}")
        if saved:
            self.saved(saved)

    def _eval_race(self, race, elapsed):
        st = race["result"]
        learner, drivers = st["learner"], st["drivers"]
        n = len(st["progress_m"])
        when = dim(f"[{duration(elapsed):>8}]") + f" {dim(st['map'])}"
        sim_s = st["steps"] * self.cfg.sim.control_dt
        if n == 1:
            end = end_label(st["reasons"][0])
            end = green("survived") if end == "survived" else red(end) + " after"
            _print(f"  {when} {bold('eval')}  {bold(meters(race['distance'])):>9}  {st['laps'][0]:.2f} laps · "
                   f"{end} {sim_s:.0f} s · avg {st['avg_speed'][0]:.1f} / max {st['max_speed'][0]:.1f} m/s · "
                   f"return {st['returns'][0]:,.0f}")
            return
        own = [i for i in range(n) if learner[i]]
        survived = sum(st["positions"][i] is not None for i in own)
        k = sum(st["overtakes"][i] for i in own)
        whose = f"of its {len(own)} cars on a {n}-car grid" if len(own) < n else f"of {n} cars"
        rivals = ""
        if st.get("duels"):
            _, per = win_rate([st["duels"]])
            rivals = " · ahead of " + ", ".join(f"{name} {share:.0%}" for name, (share, _) in sorted(per.items()))
        pts = f"{bold(f'{race['points']:.1f} pts/car')} · " if race["points"] is not None else ""
        _print(f"  {when} {bold('eval')}  {pts}mean {bold(meters(race['distance']))} · median "
               f"{meters(np.median([st['progress_m'][i] for i in own]))} {whose} · "
               f"{survived}/{len(own)} survived {sim_s:.0f} s · {k} overtake{'' if k == 1 else 's'} · "
               f"avg {np.mean([st['avg_speed'][i] for i in own]):.1f} / max "
               f"{max(st['max_speed'][i] for i in own):.1f} m/s{rivals}")
        finishers = sorted((i for i in range(n) if st["positions"][i] is not None), key=lambda i: st["positions"][i])
        crashed = sorted((i for i in range(n) if st["positions"][i] is None), key=lambda i: -st["progress_m"][i])

        def who(i):
            return bold(f"car{i}·{drivers[i]}") if learner[i] else dim(f"car{i}·{drivers[i]}")
        cells = [green(f"P{st['positions'][i]}") + f" {who(i)} {meters(st['progress_m'][i])} "
                 + dim(f"{st['avg_speed'][i]:.0f} m/s") for i in finishers]
        cells += [red("✗") + f" {who(i)} {meters(st['progress_m'][i])} {dim(end_label(st['reasons'][i]))}"
                  for i in crashed]
        indent, line, lines, width = " " * 15, "", [], 0
        for cell in cells:
            plain = visible_len(cell)
            if line and width + plain + 3 > WRAP_WIDTH - len(indent):
                lines.append(indent + line)
                line, width = "", 0
            line += (dim(" · ") if line else "") + cell
            width += plain + (3 if width else 0)
        if line:
            lines.append(indent + line)
        _print(*lines)

    # ---- status ----------------------------------------------------------------------
    def status(self, s):
        """s: the status dict built by train.py (see Dashboard.update)."""
        if self.live:
            return self.live.update(s)
        frac = s["received"] / max(s["total_steps"], 1)
        eta = (s["total_steps"] - s["received"]) / s["steps_per_s"] if s["steps_per_s"] > 0 else float("inf")
        lines = ["", _rule(duration(s["elapsed"]), f"{frac:.0%} · ETA " + (duration(eta) if np.isfinite(eta) else "–"))]
        lines.append(_row("Steps", f"{num(s['received'])} / {num(s['total_steps'])}  {bar(frac)}  "
                                   f"{s['steps_per_s']:,.0f} steps/s"))
        phase = warmup_text(s)
        if phase:
            lines.append(_row("Warm-up", yellow(phase)))
        cl, al, q = s["losses"]
        losses = f" · Q {q:.2f} · critic loss {cl:.3f} · actor loss {al:.3f}" if cl is not None else ""
        lines.append(_row("Learner", f"{num(s['updates'])} updates ({s['updates_per_s']:,.0f}/s) · replay "
                                     f"{replay_text(s)} · {s['episodes']:,} episodes" + losses))
        sm = s["summary"]
        if sm:
            n = sm["n"]
            lines.append(_row("Driving", f"median {bold(meters(sm['median']))} · mean {meters(sm['mean'])} · best "
                                         f"{meters(sm['best'])} · ≥1 lap {sm['lap_rate']:.0%} · return "
                                         f"{sm['return']:,.1f}"))
            lines.append(_row("Speed", f"avg {bold(f'{sm['speed']:.1f} m/s')} ({sm['speed'] * 3.6:.0f} km/h) · "
                                       f"top {sm['top_speed']:.1f} m/s"))
            parts = []
            for r in REASON_ORDER + tuple(sorted(set(sm["reasons"]) - set(REASON_ORDER), key=str)):
                c = sm["reasons"].get(r, 0)
                if c:
                    text = f"{end_label(r)} {c} ({c / n:.0%})"
                    parts.append(green(text) if r == "timeout" else red(text) if r in ("collision", "contact")
                                 else yellow(text))
            lines.append(_row("Endings", f"{bar(sm['survived'], 10)} " + " · ".join(parts)))
            if self.eval_cars > 1:
                lines.append(_row("Racing", f"{sm['passes']:.2f} passes per car · passed {sm['passed']:.2f} per car · "
                                            f"close calls {sm['close_calls']:.1f} "
                                            f"steps per car · {sm['purple']:.1f} fastest sectors per car · in a tow "
                                            f"{sm['tow']:.0%} · DRS open {sm['drs']:.0%} · ERS deploying "
                                            f"{sm['ers']:.0%} (battery {sm['battery']:.0%})"))
            lines.append(_row("", dim(f"↳ last {n} training car-episodes (with exploration noise)")))
        else:
            lines.append(_row("Driving", dim("no finished episodes yet")))
        lines += [_row(label, text) for label, text in info_rows(s, plain=True)]
        if s["best_eval"] > float("-inf"):
            at = f" (at {duration(s['best_eval_at'])})" if s["best_eval_at"] is not None else " (earlier run)"
            last = f" · last {score_text(s['last_eval'], self.kind)}" if s["last_eval"] is not None else ""
            lines.append(_row("Eval", f"best {bold(green(score_text(s['best_eval'], self.kind)))}{at}{last}"))
            if s["last_eval_maps"]:
                lines.append(_row("", dim("↳ last round per map: " + " · ".join(
                    f"{m} {score_text(v, self.kind)}" for m, v in s["last_eval_maps"].items()))))
        else:
            lines.append(_row("Eval", dim("no eval yet")))
        _print(*lines)


def replay_text(s):
    """'5.2M / 6.4M', or '9.1M / 9.6M (2.7M in RAM)' once the RAM part is in use."""
    ram = s.get("buffer_ram", 0)
    return f"{num(s['buffer_len'])} / {num(s['buffer_cap'])}" + (f" ({num(ram)} in RAM)" if ram else "")


def warmup_text(s):
    if s["warmup_left"] > 0:
        who = "the loaded driver" if s["warmup_policy"] else "random driving"
        done = s["warmup_total"] - s["warmup_left"]
        return (f"filling replay memory with {who}: {num(done)} / {num(s['warmup_total'])} "
                f"- learning starts after")
    if s["critic_only_left"] > 0:
        done = s["critic_only_total"] - s["critic_only_left"]
        return f"critic catching up, driver frozen: {num(done)} / {num(s['critic_only_total'])} updates"
    if s["lr_ramp"] < 1.0:
        return f"driver easing back in: learning rate at {s['lr_ramp']:.0%}"
    return ""


def info_rows(s, plain):
    """Rivals, traffic, maps, lap records, RAM, last save - shared by the
    panel (plain ANSI) and the dashboard (rich markup)."""
    b = (lambda t: bold(t)) if plain else (lambda t: f"[bold]{t}[/]")
    d = (lambda t: dim(t)) if plain else (lambda t: f"[dim]{t}[/]")
    r = (lambda t: red(t)) if plain else (lambda t: f"[red]{t}[/]")
    rows = []
    _, per = win_rate(s.get("duels") or ())
    if per:
        rows.append(("Rivals", "finishes ahead of " + " · ".join(f"{name} {b(f'{share:.0%}')}"
                                                                  for name, (share, _) in sorted(per.items()))
                     + d("  (share of comparisons, last 100 grids)")))
    traffic = s.get("traffic_steps") or {}
    if len(traffic) > 1:
        tot = sum(traffic.values())
        by = (s.get("summary") or {}).get("by_cars", {})
        rows.append(("Traffic", " · ".join(f"{c} cars {traffic[c] / tot:.0%}"
                                           + (d(f" med {meters(by[c])}") if c in by else "") for c in sorted(traffic))
                     + d("  (share of training data · median distance)")))
    maps_now, maps = s.get("maps_now") or (), s.get("maps") or ()
    if len(maps) > 1 and maps_now:
        counts = Counter(maps_now)
        rows.append(("Maps", ", ".join(f"{m}×{c}" if c > 1 else m for m, c in sorted(counts.items()))
                     + d(f"  ({len(maps)} in rotation)")))
    records = s.get("lap_records") or {}
    if records:
        shown = sorted(records.items())
        rows.append(("Laps", f"records on {len(shown)}/{len(maps) or len(shown)} maps: "
                             + " · ".join(f"{m} {lap_time(rec['lap_s'])}" for m, rec in shown[:5])
                             + (f" · +{len(shown) - 5} more" if len(shown) > 5 else "")
                             + d("  (all: compare.py --leaderboard)")))
    ram = s.get("ram")
    if ram is not None:
        used, avail = ram
        text = f"{used / 1024:.1f} GB used by training"
        if avail is not None:
            text += f" · {avail / 1024:.1f} GB free"
            if avail < 1536:
                text = r(text + " - LOW: the system may start killing processes")
        rows.append(("RAM", text))
    return rows


# -------------------------------------------------------------------------
# Live dashboard (rich)
# -------------------------------------------------------------------------

# (label, summary key, format, better: +1 higher, -1 lower, 0 neither, multi-car only)
TRAIN_ROWS = [
    ("median distance", "median", meters, +1, False),
    ("mean distance", "mean", meters, +1, False),
    ("survived the race", "survived", pct, +1, False),
    ("crashed", "crashed", pct, -1, False),
    ("  into a car", "car_crashed", pct, -1, True),
    ("stuck", "stuck", pct, -1, False),
    ("avg speed", "speed", lambda v: f"{v:.1f} m/s", +1, False),
    ("top speed", "top_speed", lambda v: f"{v:.1f} m/s", +1, False),
    ("passes per car", "passes", lambda v: f"{v:.2f}", +1, True),
    ("passed per car", "passed", lambda v: f"{v:.2f}", -1, True),
    ("close calls per car", "close_calls", lambda v: f"{v:.1f}", -1, True),
    ("fastest sectors per car", "purple", lambda v: f"{v:.1f}", +1, True),
    ("in a tow", "tow", pct, 0, True),
    ("DRS open", "drs", pct, 0, True),
    ("ERS deploying", "ers", pct, 0, True),
    ("return", "return", lambda v: f"{v:,.0f}", +1, False),
]


def _near_best(now, best, better, share=False):
    """green at or close to the best (2 points for shares, 5% otherwise), red when clearly worse."""
    worse_by = (best - now) * better
    return "green" if worse_by <= (0.02 if share else 0.05 * abs(best)) + 1e-9 else "red"


class Dashboard:
    """One screen redrawn in place: every number as start / best / now."""

    def __init__(self, cfg, profile, kind, eval_cars):
        from rich.console import Console
        from rich.live import Live
        self.cfg, self.profile, self.kind, self.eval_cars = cfg, profile, kind, eval_cars
        self.t0 = time.time()
        self.start = None
        self.best = {}
        self.now = None
        self.status = {}
        self.rounds = []
        self.events = deque(maxlen=6)
        self.last_save = None
        self.console = Console()
        self.live = Live(self._render(), console=self.console, auto_refresh=False, transient=False,
                         redirect_stdout=True, redirect_stderr=True)
        self.live.start()

    def event(self, style, text):
        from rich.text import Text
        self.events.append((time.time() - self.t0, style, Text.from_ansi(str(text))))
        self.refresh()

    def saved(self, paths):
        self.last_save = (time.time() - self.t0, ", ".join(paths))
        self.refresh()

    def eval_round(self, rnd, score, elapsed, new_best, saved, baseline):
        races = rnd["races"]
        per_map = rnd["points"] if self.kind == "points" else rnd["distance"]
        mine = lambda r: [i for i, m in enumerate(r["learner"]) if m]
        entry = {"score": score, "maps": dict(per_map), "elapsed": elapsed, "baseline": baseline,
                 "races": len(races), "distance": rnd["round_distance"],
                 "finished": sum(sum(x["result"]["positions"][i] is not None for i in mine(x["result"])) for x in races),
                 "cars": sum(len(mine(x["result"])) for x in races),
                 "passes": float(np.mean([sum(x["result"]["overtakes"][i] for i in mine(x["result"])) for x in races])),
                 **{k: sum((x["counts"] or {}).get(k, 0) for x in races)
                    for k in ("passes_on_opp", "passed_by_opp", "fastest_lap", "faster", "top10")}}
        self.rounds.append(entry)
        text = (f"eval round {score_text(score, self.kind)} · {entry['finished']}/{entry['cars']} finished · "
                f"{entry['passes']:.1f} passes per race")
        if saved:
            self.last_save = (time.time() - self.t0, ", ".join(saved))
        if baseline:
            self.event("cyan", text + " · baseline (the loaded driver, before any training)"
                       + (" · saved as td3_best.pt" if new_best else ""))
        elif new_best:
            self.event("bold green", "★ new best · " + text + " · saved td3_best.pt")
        else:
            self.event("", text)

    def update(self, s):
        self.status = s
        sm = s["summary"]
        if sm:
            self.now = sm
            if self.start is None and sm["n"] >= 100:
                self.start = dict(sm)
            if self.start is not None:
                for _, key, _, better, _ in TRAIN_ROWS:
                    v, old = sm[key], self.best.get(key)
                    if old is None or (better >= 0 and v > old[0]) or (better < 0 and v < old[0]):
                        self.best[key] = (v, s["elapsed"])
        self.refresh()

    def refresh(self):
        self.live.update(self._render(), refresh=True)

    def stop(self):
        self.live.stop()

    # ---- rendering ---------------------------------------------------------------
    def _render(self):
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        s = self.status
        parts = [self._top(s)]
        if s:
            train, evals = self._training_table(), self._eval_table(s)
            if self.console.width >= 150:
                both = Table.grid(padding=(0, 6))
                both.add_row(train, evals)
                parts.append(both)
            else:
                parts += [train, evals]
            parts.append(self._info(s))
        parts.append(self._events())
        return Panel(Group(*parts), title=f"[bold cyan]F1 racers · TD3 · {self.profile}[/]",
                     subtitle="[dim]Ctrl+C stops and saves td3_latest.pt[/]", border_style="cyan")

    def _grid(self):
        from rich.table import Table
        g = Table.grid(padding=(0, 1))
        g.add_column(style="bold", width=9)
        g.add_column()
        return g

    def _top(self, s):
        from rich.progress_bar import ProgressBar
        from rich.table import Table
        grid = self._grid()
        if not s:
            grid.add_row("Starting", "[dim]waiting for the workers ...[/]")
            return grid
        frac = s["received"] / max(s["total_steps"], 1)
        eta = (s["total_steps"] - s["received"]) / s["steps_per_s"] if s["steps_per_s"] > 0 else float("inf")
        line = Table.grid(padding=(0, 1))
        line.add_row(ProgressBar(total=1.0, completed=frac, width=30),
                     f"{num(s['received'])} / {num(s['total_steps'])} steps · {frac:.0%} · "
                     f"{s['steps_per_s']:,.0f}/s · {duration(s['elapsed'])} in · ETA "
                     + (duration(eta) if np.isfinite(eta) else "–"))
        grid.add_row("Run", line)
        phase = warmup_text(s)
        if phase:
            grid.add_row("Warm-up", f"[yellow]{phase}[/]")
        cl, al, q = s["losses"]
        losses = f" · Q {q:.1f} · critic loss {cl:.2f}" if cl is not None else ""
        grid.add_row("Learner", f"{num(s['updates'])} updates ({s['updates_per_s']:,.0f}/s) · replay "
                                f"{replay_text(s)} · {s['episodes']:,} episodes{losses}")
        return grid

    def _table(self, title, caption, last="now"):
        from rich.table import Table
        t = Table(expand=False, show_edge=False, pad_edge=False, box=None, padding=(0, 2), caption=caption,
                  caption_justify="left")
        t.add_column(title, style="bold", header_style="bold cyan")
        t.add_column("start", justify="right", style="dim")
        t.add_column("best", justify="right")
        t.add_column(last, justify="right")
        return t

    def _training_table(self):
        t = self._table("Training", "[dim]last 200 training car-episodes, with exploration noise[/]")
        if self.now is None:
            t.add_row("[dim]no finished episodes yet[/]", "", "", "")
            return t
        for label, key, fmt, better, multi in TRAIN_ROWS:
            if multi and self.eval_cars < 2:
                continue
            start = fmt(self.start[key]) if self.start else "…"
            best, style = "", ""
            if key in self.best and better != 0:
                v, at = self.best[key]
                best = f"{fmt(v)} [dim]{duration(at)}[/]"
                style = _near_best(self.now[key], v, better, share=fmt is pct)
            now = fmt(self.now[key])
            t.add_row(label, start, best, f"[{style}]{now}[/]" if style else now)
        return t

    def _eval_table(self, s):
        what = "race points per car vs the opponents" if self.kind == "points" else "mean distance"
        to_beat = ""
        if s["best_eval"] > float("-inf"):
            at = duration(s["best_eval_at"]) if s["best_eval_at"] is not None else "an earlier run"
            to_beat = f" · score to beat: [/]{score_text(s['best_eval'], self.kind)} [dim]({at})"
        t = self._table("Eval", f"[dim]{self.eval_cars} cars from the grid, no noise · score = {what}{to_beat}[/]",
                        last="last")
        if not self.rounds:
            t.add_row("[dim]no eval round yet[/]", "", "", "")
            return t
        first, last = self.rounds[0], self.rounds[-1]
        best = max(self.rounds, key=lambda r: r["score"])
        tag = lambda r: f" [dim]{duration(r['elapsed'])}[/]"
        vs = lambda now, ref, text: f"[{_near_best(now, ref, +1)}]{text}[/]"
        t.add_row("score", score_text(first["score"], self.kind) + (" [dim]baseline[/]" if first["baseline"] else ""),
                  f"[green]{score_text(best['score'], self.kind)}[/]" + tag(best),
                  vs(last["score"], best["score"], score_text(last["score"], self.kind)) + tag(last))
        for m in last["maps"]:
            cell = lambda r: score_text(r["maps"][m], self.kind) if m in r["maps"] else "–"
            t.add_row(f"  {m}", cell(first), cell(best), cell(last))
        t.add_row("finished", f"{first['finished']}/{first['cars']}", f"{best['finished']}/{best['cars']}",
                  vs(last["finished"], best["finished"], f"{last['finished']}/{last['cars']}"))
        if self.eval_cars > 1:
            t.add_row("passes per race", f"{first['passes']:.1f}", f"{best['passes']:.1f}",
                      vs(last["passes"], best["passes"], f"{last['passes']:.1f}"))
        if self.kind == "points":
            k = last["races"]
            for label, key, fmt in (("passes on opponents", "passes_on_opp", "{}"),
                                    ("passed by opponents", "passed_by_opp", "{}"),
                                    ("fastest laps", "fastest_lap", "{}/" + str(k)),
                                    ("faster than opponents", "faster", "{}/" + str(k))):
                t.add_row(label, fmt.format(first[key]), fmt.format(best[key]), fmt.format(last[key]))
            t.add_row("top-10 finishes", *(f"{r['top10']}/{r['cars']}" for r in (first, best, last)))
        t.add_row("mean distance", meters(first["distance"]), meters(best["distance"]), meters(last["distance"]))
        return t

    def _info(self, s):
        grid = self._grid()
        for label, text in info_rows(s, plain=False):
            grid.add_row(label, text)
        if self.last_save:
            at, paths = self.last_save
            grid.add_row("Saved", f"{paths} [dim]({duration(at)})[/]")
        return grid

    def _events(self):
        from rich.console import Group
        from rich.table import Table
        from rich.text import Text
        t = Table.grid(padding=(0, 1))
        t.add_column(style="dim", justify="right", width=9)
        t.add_column()
        for at, style, text in self.events:
            text = text.copy()
            if style:
                text.stylize(style)
            t.add_row(duration(at), text)
        if not self.events:
            t.add_row("", Text("no events yet", style="dim"))
        return Group(Text("Recent", style="bold"), t)
