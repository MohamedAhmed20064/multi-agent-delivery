"""Scenarios (CSV), geometry helpers and the traffic model."""
import csv, math, os, random
from dataclasses import dataclass, asdict, field
import numpy as np

DEPOT = (50.0, 50.0)
SCENARIO_KINDS = ["uniform", "clustered", "tight", "traffic", "dynamic", "failures"]
SCENARIO_NAMES = ["1 Uniform", "2 Clustered", "3 Tight deadlines", "4 Traffic jams", "5 Live new orders", "6 Breakdowns & cancels"]


@dataclass
class Order:
    id: int
    x: float
    y: float
    due: float
    request: float = 0.0


@dataclass
class CourierSpec:
    id: int
    capacity: int
    speed: float


@dataclass
class TrafficEvent:
    time: float
    x1: float
    y1: float
    x2: float
    y2: float
    factor: float
    duration: float


class TrafficModel:
    """Rectangular congestion zones: a road segment whose midpoint is inside a zone is slower."""

    def __init__(self, events=None):
        self.events = list(events or [])

    def add(self, e):
        self.events.append(e)

    def active(self, t):
        return [e for e in self.events if e.time <= t < e.time + e.duration]

    def factor(self, P, Q, t=None):
        P = np.asarray(P, float); Q = np.asarray(Q, float)
        mid = (P + Q) / 2
        f = np.ones(mid.shape[:-1])
        for e in self.events:
            if t is not None and not (e.time <= t < e.time + e.duration):
                continue
            inside = (mid[..., 0] >= e.x1) & (mid[..., 0] <= e.x2) & (mid[..., 1] >= e.y1) & (mid[..., 1] <= e.y2)
            f = np.where(inside, np.maximum(f, e.factor), f)
        return f


@dataclass
class Scenario:
    name: str
    orders: list
    couriers: list
    traffic: list
    events: list = field(default_factory=list)   # (time, "breakdown"|"cancel", target id)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def route_length(points, closed=False):
    """Length of the path through `points` (list of (x, y)); closed=True returns to the first point."""
    pts = list(points)
    if closed and pts:
        pts.append(pts[0])
    return sum(dist(a, b) for a, b in zip(pts, pts[1:]))


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def generate_scenario(folder, kind, seed, n=20, n_couriers=8, cap=3):
    os.makedirs(folder, exist_ok=True)
    rng = random.Random(seed)
    centers = [(rng.uniform(15, 85), rng.uniform(15, 85)) for _ in range(4)]
    orders = []
    for i in range(n):
        if kind == "clustered":
            cx, cy = rng.choice(centers)
            x, y = min(98, max(2, rng.gauss(cx, 7))), min(98, max(2, rng.gauss(cy, 7)))
        else:
            x, y = rng.uniform(2, 98), rng.uniform(2, 98)
        lo, hi = (10, 30) if kind == "tight" else (25, 80)
        due = dist(DEPOT, (x, y)) / 1.5 + rng.uniform(lo, hi)
        orders.append(Order(i, round(x, 1), round(y, 1), round(due, 1), 0.0))
    if kind == "dynamic":
        for k, t in enumerate((20, 35, 55)):
            x, y = rng.uniform(2, 98), rng.uniform(2, 98)
            orders.append(Order(n + k, round(x, 1), round(y, 1), round(t + dist(DEPOT, (x, y)) / 1.5 + rng.uniform(25, 50), 1), float(t)))
    couriers = [CourierSpec(i, cap, round(rng.uniform(1.2, 1.8), 2)) for i in range(n_couriers)]
    traffic = []
    if kind == "traffic":
        traffic = [TrafficEvent(15, 25, 25, 75, 75, 2.5, 60), TrafficEvent(50, 60, 20, 95, 60, 2.0, 40)]
    events = [(10, "cancel", 3), (15, "breakdown", 2), (25, "cancel", 9), (30, "breakdown", 5)] if kind == "failures" else []
    _write(os.path.join(folder, "events.csv"), ["time", "type", "target"], events)
    _write(os.path.join(folder, "orders.csv"), ["id", "x", "y", "due", "request"], [[o.id, o.x, o.y, o.due, o.request] for o in orders])
    _write(os.path.join(folder, "couriers.csv"), ["id", "capacity", "speed"], [[c.id, c.capacity, c.speed] for c in couriers])
    _write(os.path.join(folder, "traffic.csv"), ["time", "x1", "y1", "x2", "y2", "factor", "duration"], [list(asdict(e).values()) for e in traffic])


def generate_all(root="data"):
    for k, kind in enumerate(SCENARIO_KINDS, 1):
        generate_scenario(os.path.join(root, f"scenario_{k}"), kind, seed=10 + k)


def load_scenario(folder):
    rd = lambda f: list(csv.DictReader(open(os.path.join(folder, f))))
    orders = [Order(int(r["id"]), float(r["x"]), float(r["y"]), float(r["due"]), float(r["request"])) for r in rd("orders.csv")]
    couriers = [CourierSpec(int(r["id"]), int(r["capacity"]), float(r["speed"])) for r in rd("couriers.csv")]
    traffic = [TrafficEvent(*[float(r[k]) for k in ("time", "x1", "y1", "x2", "y2", "factor", "duration")]) for r in rd("traffic.csv")]
    ev_path = os.path.join(folder, "events.csv")
    events = [(float(r["time"]), r["type"], int(r["target"])) for r in rd("events.csv")] if os.path.exists(ev_path) else []
    return Scenario(os.path.basename(folder), orders, couriers, traffic, events)


def random_scenario(n, seed=0, cap=3):
    """In-memory scenario of n customers (used for the runtime study)."""
    rng = random.Random(seed)
    orders = []
    for i in range(n):
        x, y = rng.uniform(2, 98), rng.uniform(2, 98)
        orders.append(Order(i, x, y, dist(DEPOT, (x, y)) / 1.5 + rng.uniform(25, 80)))
    couriers = [CourierSpec(i, cap, rng.uniform(1.2, 1.8)) for i in range(math.ceil(n / cap) + 2)]
    return Scenario(f"random_{n}", orders, couriers, [])
