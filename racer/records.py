"""
Small files in each profile's checkpoint folder:

    metrics.csv       one row per log_every_s, appended across runs (`run` tells them apart)
    evals.csv         one row per map per eval round (its grids averaged)
    lap_records.json  the fastest complete lap per map
    best_meta.json    the best eval round's score and the settings it was scored under
"""

import csv
import json
import time

METRICS_FIELDS = [
    "run", "wall_time", "elapsed_s", "env_steps", "updates", "steps_per_s", "updates_per_s",
    "median_distance_m", "mean_distance_m", "best_distance_m", "lap_rate", "survival_rate", "crash_rate",
    "car_crash_rate", "stuck_rate", "avg_speed_ms", "top_speed_ms",
    "passes_per_car", "tow_frac", "close_calls_per_car", "purple_sectors_per_car", "drs_open_frac",
    "ers_deploy_frac", "battery_mean",
    "critic_loss", "actor_loss", "avg_q", "ram_used_mb", "ram_free_mb",
    "win_rate_vs_opponents", "win_rates", "best_eval", "best_by",
    "passed_per_car", "passed_by_opp_per_car",
]
# Distances / survived / overtakes / speeds count the trained brain's cars;
# survived and overtakes are per race, the rest averaged over the map's grids.
EVAL_FIELDS = [
    "run", "wall_time", "elapsed_s", "env_steps", "updates", "map", "cars", "grids", "learner_cars", "opponents",
    "median_distance_m", "mean_distance_m", "survived", "overtakes", "avg_speed_ms", "best_lap_s",
    "points_per_car", "round_distance_m", "round_points", "best_by", "new_best", "win_rate_vs_opponents",
    "passed", "passed_by_opp",
]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class CsvLog:
    """Appends rows; a file with another header is moved aside (no migration)."""

    def __init__(self, path, fields, enabled=True):
        self.path, self.fields, self.enabled = path, fields, enabled
        self._checked = False

    def write(self, row):
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self._checked and self.path.exists():
            with open(self.path, newline="") as f:
                header = next(csv.reader(f), None)
            if header != self.fields:
                self.path.rename(self.path.with_name(f"{self.path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.old.csv"))
        self._checked = True
        new = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: _fmt(row.get(k)) for k in self.fields})


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class LapRecords:
    """The fastest complete lap per map, kept as JSON."""

    def __init__(self, path, enabled=True):
        self.path, self.enabled = path, enabled
        self.records = {}
        if path.exists():
            try:
                with open(path) as f:
                    self.records = json.load(f)
            except (OSError, ValueError):
                self.records = {}

    @staticmethod
    def load(cfg, profile):
        """A profile's lap records ({} if none yet)."""
        return LapRecords(cfg.profile_dir(profile) / "lap_records.json", enabled=False).records

    def offer(self, map_name, lap_s, **info):
        """Record lap_s if it's the fastest on this map. Returns the previous
        record (None if there was none) when it's a new record, else False."""
        old = self.records.get(map_name)
        if old is not None and lap_s >= old["lap_s"]:
            return False
        self.records[map_name] = {"lap_s": round(float(lap_s), 3), "date": time.strftime("%Y-%m-%d %H:%M"), **info}
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(self.records, f, indent=1, sort_keys=True)
        return old


def load_best(path, signature):
    """(best score stored for exactly these settings or -inf, whether one
    matched, whether a best_meta.json exists at all)."""
    try:
        with open(path) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return float("-inf"), False, False
    if meta.get("signature") != signature:
        return float("-inf"), False, True
    return float(meta.get("score", float("-inf"))), True, True


def save_best(path, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=1)
    tmp.replace(path)
