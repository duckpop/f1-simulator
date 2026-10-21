"""
Draws the README's diagrams into docs/ from the simulator's real numbers:
the lidar beam angles and track-preview distances (configs/settings.toml)
and the network shape.

    uv run docs/make_diagrams.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racer import config  # noqa: E402

cfg = config.load()
DOCS = Path(__file__).resolve().parent

STYLE = """<style>
.bg { fill: #ffffff; stroke: #d6dae2 }
text { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif }
.syn { stroke: #5b3cc4; stroke-width: 0.6; opacity: 0.14 }
.leader { stroke: #586072; stroke-width: 0.8; fill: none }
.in-label { font-size: 13px; fill: #12151b }
.in-count, .col-sub, .pv-label, .legend { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace }
.in-count { font-size: 12px; fill: #586072 }
.in-dot { fill: #12151b }
.neuron { fill: #ffffff; stroke: #5b3cc4; stroke-width: 1.6 }
.out { fill: #5b3cc4 }
.out-label, .col-label { font-size: 15px; font-weight: 600; fill: #12151b }
.col-sub { font-size: 11px; fill: #586072 }
.g-lidar { fill: #c97a0a } .g-closing { fill: #e8b062 } .g-self { fill: #737b8c }
.g-race { fill: #5b3cc4 } .g-cars { fill: #0b8a6f } .g-track { fill: #2f6fd0 }
.legend { font-size: 12px; fill: #12151b }
.asphalt { fill: #e4e7ec }
.wall { stroke: #2a2f38; stroke-width: 2.5; fill: none; stroke-linejoin: round }
.centre { stroke: #586072; stroke-width: 0.8; stroke-dasharray: 3 6; fill: none }
.zone { fill: #0b8a6f; fill-opacity: 0.16; stroke: #0b8a6f; stroke-width: 1; stroke-dasharray: 3 3 }
.blind { fill: #586072; opacity: 0.1 }
.beam { stroke: #c97a0a; stroke-width: 0.9; opacity: 0.8 }
.beam-car { stroke: #d0362b; stroke-width: 1.1 }
.hit { fill: #c97a0a } .hit-car { fill: #d0362b }
.pv-tick { stroke: #2f6fd0; stroke-width: 2 }
.pv-head { stroke: #2f6fd0; stroke-width: 1.6; fill: none }
.pv-dot { fill: #2f6fd0 }
.pv-label { font-size: 11px; fill: #2f6fd0; paint-order: stroke; stroke: #e4e7ec; stroke-width: 4px; stroke-linejoin: round }
.car-other { fill: #ffffff; stroke: #12151b; stroke-width: 1.3 }
.car-self { fill: #5b3cc4 }
.badge circle { fill: #12151b }
.badge text { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; font-weight: 600; fill: #ffffff }
</style>"""


def svg(width, height, title, desc, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
            f'height="{height}" role="img"><title>{title}</title><desc>{desc}</desc>{STYLE}'
            f'<rect class="bg" x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="10"/>{body}</svg>\n')


# --------------------------------------------------------------------------
# The beams the network gets, at their real angles
# --------------------------------------------------------------------------
n_beams = int(cfg.sensors.lidar_beams)
step = cfg.sensors.lidar_fov_deg / (n_beams - 1)
ANGLES = [(i - (n_beams - 1) / 2) * step for i in cfg.obs_beams]
PREVIEW_M = list(cfg.sensors.track_preview_m)
RANGE_M = cfg.sensors.lidar_range

# --------------------------------------------------------------------------
# Car's-eye view: a straight into a right-hander, the lidar ray-cast against
# the walls and three other cars (not to scale)
# --------------------------------------------------------------------------
CX, CY = 230.0, 380.0
R = 280.0                                   # drawn lidar range
PX_PER_M = R / RANGE_M
ARC_C = (400.0, 260.0)
CAR_W, CAR_H = 12.0, 26.0


def arc(radius, n=60, t_max=90.0):
    return [(ARC_C[0] - radius * math.cos(math.radians(t)), ARC_C[1] - radius * math.sin(math.radians(t)))
            for t in [t_max * i / n for i in range(n + 1)]]


def rect_poly(cx, cy, w=CAR_W, h=CAR_H):
    return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2),
            (cx - w / 2, cy - h / 2)]


def path(points):
    return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in points)


outer = [(170.0, 540.0)] + arc(230.0) + [(470.0, 30.0)]          # the corner exits to the right edge
inner = [(290.0, 540.0)] + arc(110.0) + [(470.0, 150.0)]
cars = {"ahead": (250.0, 262.0), "side": (264.0, 382.0), "behind": (226.0, 455.0)}
obstacles = [("wall", outer), ("wall", inner)] + [("car", rect_poly(*c)) for c in cars.values()]


def cast(angle_deg):
    a = math.radians(angle_deg)
    dx, dy = math.sin(a), -math.cos(a)
    best, kind = R, None
    for k, poly in obstacles:
        for (x1, y1), (x2, y2) in zip(poly, poly[1:]):
            ex, ey = x2 - x1, y2 - y1
            den = dx * ey - dy * ex
            if abs(den) < 1e-12:
                continue
            t = ((x1 - CX) * ey - (y1 - CY) * ex) / den
            u = ((x1 - CX) * dy - (y1 - CY) * dx) / den
            if 1e-6 < t < best and 0.0 <= u <= 1.0:
                best, kind = t, k
    return CX + best * dx, CY + best * dy, kind


def centre_point(s_px):
    straight = CY - ARC_C[1]
    if s_px <= straight:
        return CX, CY - s_px, (0.0, -1.0)
    t = (s_px - straight) / 170.0
    if t > math.pi / 2:                                            # past the corner: straight to the right
        return ARC_C[0] + 170.0 * (t - math.pi / 2), ARC_C[1] - 170.0, (1.0, 0.0)
    return ARC_C[0] - 170.0 * math.cos(t), ARC_C[1] - 170.0 * math.sin(t), (math.sin(t), -math.cos(t))


def car_rect(cx, cy, cls):
    return (f'<rect class="{cls}" x="{cx - CAR_W / 2:.1f}" y="{cy - CAR_H / 2:.1f}" width="{CAR_W:.0f}" '
            f'height="{CAR_H:.0f}" rx="3"/>')


def badge(n, x, y):
    return (f'<g class="badge"><circle cx="{x}" cy="{y}" r="10"/>'
            f'<text x="{x}" y="{y + 4}" text-anchor="middle">{n}</text></g>')


def car_view():
    parts = [f'<path class="asphalt" d="{path(outer + list(reversed(inner)))} Z"/>',
             f'<path class="wall" d="{path(outer)}"/><path class="wall" d="{path(inner)}"/>',
             f'<path class="centre" d="{path([centre_point(i * 4.0)[:2] for i in range(120)])}"/>',
             f'<rect class="zone" x="{CX - 20:.0f}" y="{CY + 16:.0f}" width="40" height="76" rx="4"/>',
             f'<rect class="zone" x="{CX - 30:.0f}" y="{CY - 13:.0f}" width="18" height="26" rx="3"/>',
             f'<rect class="zone" x="{CX + 12:.0f}" y="{CY - 13:.0f}" width="18" height="26" rx="3"/>']
    half_blind = (360.0 - cfg.sensors.lidar_fov_deg) / 2
    bx, by = 120 * math.sin(math.radians(half_blind)), 120 * math.cos(math.radians(half_blind))
    parts.append(f'<path class="blind" d="M{CX:.0f} {CY:.0f} L{CX - bx:.1f} {CY + by:.1f} '
                 f'A120 120 0 0 0 {CX + bx:.1f} {CY + by:.1f} Z"/>')
    hits = []
    for ang in ANGLES:
        x, y, kind = cast(ang)
        parts.append(f'<line class="{"beam-car" if kind == "car" else "beam"}" x1="{CX:.0f}" y1="{CY:.0f}" '
                     f'x2="{x:.1f}" y2="{y:.1f}"/>')
        if kind:
            hits.append(f'<circle class="{"hit-car" if kind == "car" else "hit"}" cx="{x:.1f}" cy="{y:.1f}" r="2"/>')
    parts += hits
    parts.append('<defs><marker id="pv-arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="5" '
                 'markerHeight="5" orient="auto"><path class="pv-head" d="M1 1L7 5L1 9"/></marker></defs>')
    for d in PREVIEW_M:
        x, y, (tx, ty) = centre_point(d * PX_PER_M)
        parts.append(f'<line class="pv-tick" x1="{x - 8 * tx:.1f}" y1="{y - 8 * ty:.1f}" x2="{x + 8 * tx:.1f}" '
                     f'y2="{y + 8 * ty:.1f}" marker-end="url(#pv-arrow)"/>'
                     f'<circle class="pv-dot" cx="{x:.1f}" cy="{y:.1f}" r="3.2"/>')
        if d in (100, 200, 300, PREVIEW_M[-1]):
            lx, ly = x - 14 * ty, y + 14 * tx
            parts.append(f'<text class="pv-label" x="{lx:.1f}" y="{ly + 4:.1f}" '
                         f'text-anchor="{"end" if lx < x else "start"}">{d:g} m</text>')
    parts += [car_rect(*c, "car-other") for c in cars.values()]
    parts.append(car_rect(CX, CY, "car-self"))
    parts += [badge(1, 120, 150), badge(2, 274, 236), badge(3, 412, 34), badge(4, 186, 470), badge(5, 300, 412)]
    return svg(460, 520, "What one car senses",
               "Top-down view: 50 lidar beams over 270 degrees stop at walls or cars, seven track-preview points "
               "follow the corner ahead, a mirror zone sits behind the car and alongside zones to its sides.",
               "".join(parts))


# --------------------------------------------------------------------------
# The brain: input groups -> 384 -> 384 -> 5
# --------------------------------------------------------------------------
L = cfg.obs
GROUPS = [("Lidar distances", L.lidar, "g-lidar"), ("Lidar closing speeds", L.closing, "g-closing"),
          ("Car state", L.state, "g-self"), ("Race position", L.race, "g-race"), ("DRS", L.drs, "g-race"),
          ("ERS battery", L.ers, "g-race"), ("Mirror", L.mirrors, "g-cars"), ("Track preview", L.preview, "g-track"),
          ("Alongside", L.alongside, "g-cars"), ("Car design", L.design, "g-self")]
GROUPS = [(name, s.stop - s.start, cls) for name, s, cls in GROUPS]
OUTPUTS = ["Steer", "Throttle", "Brake", "DRS", "ERS"]
HIDDEN = cfg.td3.hidden


def brain():
    parts, labels, dots = [], [], []
    y, unit = 36.0, 2.4
    for i, (name, n, cls) in enumerate(GROUPS):
        h = n * unit
        parts.append(f'<rect class="{cls}" x="36" y="{y:.1f}" width="22" height="{h:.1f}"/>')
        mid, ly = y + h / 2, 44 + i * 36.5
        labels.append(f'<path class="leader" d="M58 {mid:.1f} C 76 {mid:.1f}, 72 {ly:.1f}, 92 {ly:.1f}"/>'
                      f'<text class="in-label" x="98" y="{ly + 4:.1f}">{name}</text>'
                      f'<text class="in-count" x="246" y="{ly + 4:.1f}" text-anchor="end">{n}</text>')
        dots.append((262.0, ly))
        y += h + 2.0
    l1, l2, ox = 368.0, 488.0, 604.0
    ny = [40 + i * 26.0 for i in range(14)]
    oy = [118 + i * 46.0 for i in range(len(OUTPUTS))]
    syn = [f'<line class="syn" x1="{x1}" y1="{y1:.1f}" x2="{l1 - 7}" y2="{y2:.1f}"/>' for x1, y1 in dots for y2 in ny]
    syn += [f'<line class="syn" x1="{l1 + 7}" y1="{a:.1f}" x2="{l2 - 7}" y2="{b:.1f}"/>' for a in ny for b in ny]
    syn += [f'<line class="syn" x1="{l2 + 7}" y1="{a:.1f}" x2="{ox - 9}" y2="{b:.1f}"/>' for a in ny for b in oy]
    nodes = [f'<circle class="in-dot" cx="{x}" cy="{yy:.1f}" r="3.5"/>' for x, yy in dots]
    nodes += [f'<circle class="neuron" cx="{l1}" cy="{yy:.1f}" r="7"/><circle class="neuron" cx="{l2}" '
              f'cy="{yy:.1f}" r="7"/>' for yy in ny]
    nodes += [f'<circle class="out" cx="{ox}" cy="{yy:.1f}" r="9"/><text class="out-label" x="{ox + 16}" '
              f'y="{yy + 5:.1f}">{name}</text>' for name, yy in zip(OUTPUTS, oy)]
    cols = (f'<text class="col-label" x="36" y="420">{cfg.obs_dim} inputs</text>'
            f'<text class="col-sub" x="36" y="438">scaled to about -1…+1</text>'
            + "".join(f'<text class="col-label" x="{x}" y="420" text-anchor="middle">{t}</text>'
                      f'<text class="col-sub" x="{x}" y="438" text-anchor="middle">{s1}</text>'
                      f'<text class="col-sub" x="{x}" y="453" text-anchor="middle">{s2}</text>'
                      for x, t, s1, s2 in ((l1, "Layer 1", f"{HIDDEN} neurons", "ReLU"),
                                           (l2, "Layer 2", f"{HIDDEN} neurons", "ReLU"),
                                           (ox, "Outputs", "5 controls", "tanh"))))
    return svg(700, 470, "The driver network",
               f"{cfg.obs_dim} inputs in ten groups feed two hidden layers of {HIDDEN} ReLU neurons, which feed "
               "five tanh outputs: steer, throttle, brake, DRS and ERS.",
               "".join(syn + parts + labels + nodes) + cols)


def inputs_strip():
    cells, x = [], 10.0
    w = 680.0 / cfg.obs_dim
    for _, n, cls in GROUPS:
        for _ in range(n):
            cells.append(f'<rect class="{cls}" x="{x + 0.4:.2f}" y="12" width="{w - 0.8:.2f}" height="30"/>')
            x += w
    count = {cls: 0 for _, _, cls in GROUPS}
    for _, n, cls in GROUPS:
        count[cls] += n
    legend = [("g-lidar", f"Lidar distances · {count['g-lidar']}"),
              ("g-closing", f"Lidar closing speeds · {count['g-closing']}"),
              ("g-self", f"Own car: state + design · {count['g-self']}"),
              ("g-race", f"Race, DRS, ERS · {count['g-race']}"),
              ("g-cars", f"Mirror + alongside · {count['g-cars']}"),
              ("g-track", f"Track preview · {count['g-track']}")]
    for i, (cls, text) in enumerate(legend):
        lx, ly = 10 + (i % 3) * 230, 64 + (i // 3) * 22
        cells.append(f'<rect class="{cls}" x="{lx}" y="{ly - 10}" width="12" height="12"/>'
                     f'<text class="legend" x="{lx + 18}" y="{ly}">{text}</text>')
    return svg(700, 104, f"The {cfg.obs_dim} inputs to scale",
               "One cell per input in network order, coloured by group.", "".join(cells))


for name, text in (("brain.svg", brain()), ("car-view.svg", car_view()), ("inputs.svg", inputs_strip())):
    (DOCS / name).write_text(text)
    print("wrote", DOCS / name)
