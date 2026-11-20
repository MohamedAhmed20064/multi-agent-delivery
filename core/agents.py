"""Agents + message bus. Hierarchy:  Customer -> Supervisor -> Dispatcher -> Courier,  Traffic -> Supervisor."""
import time
from collections import deque
from dataclasses import dataclass, asdict
import numpy as np
from core.scenario import dist, DEPOT, TrafficModel, TrafficEvent
from algorithms.aco import Problem, solve_aco

NEW_ORDER, ORDER_ACK, PLAN_REQUEST, PLAN_READY = "NEW_ORDER", "ORDER_ACK", "PLAN_REQUEST", "PLAN_READY"
ROUTE_ASSIGN, STATUS_QUERY, LOCATION_UPDATE = "ROUTE_ASSIGN", "STATUS_QUERY", "LOCATION_UPDATE"
DELIVERED, RECEIPT_CONFIRM, TRAFFIC_UPDATE = "DELIVERED", "RECEIPT_CONFIRM", "TRAFFIC_UPDATE"
BREAKDOWN, CANCEL_ORDER = "BREAKDOWN", "CANCEL_ORDER"


@dataclass
class Message:
    type: str
    sender: str
    receiver: str
    payload: dict
    time: float


class MessageBus:
    def __init__(self):
        self.agents, self.queue, self.log, self.now = {}, deque(), [], 0.0

    def register(self, agent):
        self.agents[agent.name] = agent

    def send(self, type, sender, receiver, **payload):
        self.queue.append(Message(type, sender, receiver, payload, self.now))

    def drain(self):
        while self.queue:
            m = self.queue.popleft()
            self.log.append(m)
            self.agents[m.receiver].receive(m)


class Agent:
    def __init__(self, name, bus):
        self.name, self.bus = name, bus
        bus.register(self)

    def receive(self, m):
        h = getattr(self, "on_" + m.type.lower(), None)
        if h:
            h(m)


class CustomerAgent(Agent):
    def __init__(self, bus, order):
        super().__init__(f"Customer-{order.id}", bus)
        self.order, self.state, self.delivered_at = order, "new", None

    def place_order(self):
        o = self.order
        self.state = "ordered"
        self.bus.send(NEW_ORDER, self.name, "Supervisor", order=dict(id=o.id, x=o.x, y=o.y, due=o.due, request=o.request))

    def on_order_ack(self, m):
        self.state = "accepted"

    def cancel(self):
        self.state = "cancelled"
        self.bus.send(CANCEL_ORDER, self.name, "Supervisor", id=self.order.id)

    def on_delivered(self, m):
        self.state, self.delivered_at = "delivered", m.payload["time"]
        self.bus.send(RECEIPT_CONFIRM, self.name, "Supervisor", id=self.order.id, time=self.delivered_at)


class CourierAgent(Agent):
    def __init__(self, bus, spec, world_traffic, service=1.0, update_every=5.0):
        super().__init__(f"Courier-{spec.id}", bus)
        self.spec, self.world, self.service, self.update_every = spec, world_traffic, service, update_every
        self.pos = np.array(DEPOT, float)
        self.route = []            # [(order_id, x, y), ...]
        self.odometer, self.wait, self.last_update, self.broken = 0.0, 0.0, 0.0, False

    def status(self, now):
        return dict(courier=self.spec.id, pos=tuple(self.pos), route=list(self.route),
                    speed=self.spec.speed, capacity=self.spec.capacity, time=now)

    def breakdown(self):
        self.broken = True
        self.bus.send(BREAKDOWN, self.name, "Supervisor", courier=self.spec.id, orders=[r[0] for r in self.route])
        self.route = []

    def at_depot(self):
        return bool(np.allclose(self.pos, DEPOT))

    def on_route_assign(self, m):
        self.route = [tuple(r) for r in m.payload["route"]]

    def on_status_query(self, m):
        self.bus.send(LOCATION_UPDATE, self.name, "Supervisor", reply=True, **self.status(self.bus.now))

    def tick(self, now):
        if self.broken:
            return
        if now - self.last_update >= self.update_every:
            self.last_update = now
            self.bus.send(LOCATION_UPDATE, self.name, "Supervisor", reply=False, **self.status(now))

    def step(self, now, dt):
        budget = dt
        while budget > 1e-9 and not self.broken:
            if self.wait > 0:
                use = min(self.wait, budget); self.wait -= use; budget -= use; continue
            if self.route:
                tgt = np.array(self.route[0][1:3], float)
            elif not self.at_depot():
                tgt = np.array(DEPOT, float)
            else:
                break
            d = float(np.linalg.norm(tgt - self.pos))
            v = self.spec.speed / float(self.world.factor(self.pos, tgt, now))
            need = d / v
            if need <= budget:
                budget -= need; self.odometer += d; self.pos = tgt
                if self.route:
                    oid = self.route.pop(0)[0]
                    self.bus.send(DELIVERED, self.name, f"Customer-{oid}", time=now + dt - budget, courier=self.spec.id)
                    self.wait = self.service
            else:
                move = budget * v
                self.pos = self.pos + (tgt - self.pos) * (move / d)
                self.odometer += move; budget = 0


class TrafficAgent(Agent):
    def __init__(self, bus, world, events):
        super().__init__("Traffic", bus)
        self.world, self.pending = world, sorted(events, key=lambda e: e.time)

    def step(self, now):
        while self.pending and self.pending[0].time <= now + 1e-9:
            e = self.pending.pop(0)
            e.time = max(e.time, 0.0)
            self.world.add(e)
            self.bus.send(TRAFFIC_UPDATE, self.name, "Supervisor", event=asdict(e))


class DispatcherAgent(Agent):
    """Plans routes (ACO or nearest-customer). Keeps the pheromone matrix between plans (warm start)."""

    def __init__(self, bus, mode="aco", params=None, seed=0):
        super().__init__("Dispatcher", bus)
        from algorithms.aco import ACOParams
        self.mode, self.params = mode, params or ACOParams()
        self.rng = np.random.default_rng(seed)
        self.tau = np.ones((1, 1))
        self.last = None

    def _sub(self, ids):
        need = max(ids) + 2
        if need > self.tau.shape[0]:
            new = np.ones((need, need)); s = self.tau.shape[0]; new[:s, :s] = self.tau; self.tau = new
        idx = [0] + [i + 1 for i in ids]
        return self.tau[np.ix_(idx, idx)].copy(), idx

    def on_plan_request(self, m):
        p, P = m.payload, self.params
        now = p["now"]
        known = TrafficModel([TrafficEvent(**e) for e in p["traffic"]])
        starts, t0, speed, cap, frozen = [], [], [], [], {}
        for c in p["couriers"]:
            pos, sp = np.array(c["pos"]), c["speed"]
            if c["route"]:                       # the stop it is driving to stays fixed
                fz = c["route"][0]; tgt = np.array(fz[1:3])
                tt = dist(pos, tgt) * float(known.factor(pos, tgt)) / sp
                starts.append(tgt); t0.append(now + tt + P.service); cap.append(max(0, c["capacity"] - 1))
                frozen[c["courier"]] = tuple(fz)
            else:
                starts.append(pos); t0.append(now); cap.append(c["capacity"])
            speed.append(sp)
        fz_ids = {f[0] for f in frozen.values()}
        pending = [o for o in p["pending"] if o["id"] not in fz_ids]
        ids = [o["id"] for o in pending]
        tic = time.perf_counter()
        base = None
        if ids:
            prob = Problem([(o["x"], o["y"]) for o in pending], [o["due"] for o in pending],
                           starts, t0, speed, cap, DEPOT, known.factor, P)
            base = prob.nearest()
            if self.mode == "aco":
                tau, idx = self._sub(ids)
                iters = P.iterations if p["full"] else max(15, P.iterations // 2)
                if p.get("intensify"):
                    iters *= 2
                sol, tau = solve_aco(prob, P, tau, iters, rng=self.rng)
                self.tau[np.ix_(idx, idx)] = tau          # keep what was learned
            elif self.mode in ("ga", "sa"):
                from algorithms.metaheuristics import solve_ga, solve_sa
                iters = P.iterations if p["full"] else max(15, P.iterations // 2)
                sol = (solve_ga if self.mode == "ga" else solve_sa)(prob, P, self.rng, evals=P.ants * iters * (2 if p.get("intensify") else 1))
            else:
                sol = base
            routes = sol.routes
            info = dict(cost=sol.cost, dist=sol.dist, late=sol.late_cnt, history=sol.history,
                        history_dist=sol.history_dist, baseline_cost=base.cost, baseline_dist=base.dist)
        else:
            routes = [[] for _ in p["couriers"]]
            info = dict(cost=0, dist=0, late=0, history=[], history_dist=[], baseline_cost=0, baseline_dist=0)
        elapsed = time.perf_counter() - tic
        assigned = set(fz_ids)
        for k, c in enumerate(p["couriers"]):
            new = [(ids[j], pending[j]["x"], pending[j]["y"]) for j in routes[k]]
            assigned.update(o[0] for o in new)
            r = ([frozen[c["courier"]]] if c["courier"] in frozen else []) + new
            self.bus.send(ROUTE_ASSIGN, self.name, f"Courier-{c['courier']}", route=r)
        unassigned = [o["id"] for o in p["pending"] if o["id"] not in assigned]
        self.last = dict(info, n=len(ids), elapsed=elapsed, mode=self.mode, unassigned=unassigned, reason=p["reason"])
        self.bus.send(PLAN_READY, self.name, "Supervisor", **self.last)


class SupervisorAgent(Agent):
    """Manages everything: receives orders/updates/failures, decides WHEN to replan, validates plans,
    escalates effort, watches couriers for lateness risk and keeps the KPIs.

    policy (used for the ablation study):
      full                 - event-driven replanning + risk detection + escalation   (default)
      no_risk_escalation   - event-driven replanning only
      fixed_timer          - no event reaction at all: replans only every `period` minutes
    """

    def __init__(self, bus, courier_ids, cooldown=20.0, max_risk_replans=3, policy="full", period=30.0):
        super().__init__("Supervisor", bus)
        self.courier_ids, self.cooldown, self.max_risk = list(courier_ids), cooldown, max_risk_replans
        self.policy, self.period = policy, period
        self.orders, self.delivered, self.status, self.traffic = {}, {}, {}, []
        self.cancelled, self.broken = set(), set()
        self.flag, self.replanning, self.waiting, self.intensify, self.reason = None, False, set(), False, ""
        self.decisions, self.plans = [], []
        self.has_plan, self.planned_late, self.unassigned = False, 0, []
        self.last_replan, self.risk_replans, self.escalated, self.now = -1e9, 0, False, 0.0

    def log(self, text):
        self.decisions.append((self.bus.now, text))

    def request(self, reason):
        if self.policy == "fixed_timer" and reason != "timer":
            return
        if self.policy == "no_risk_escalation" and reason in ("intensify", "late-risk"):
            return
        if self.flag is None:
            self.flag = reason

    # --- inputs
    def on_new_order(self, m):
        o = m.payload["order"]
        self.orders[o["id"]] = o
        self.bus.send(ORDER_ACK, self.name, m.sender)
        self.escalated = False
        self.log(f"order #{o['id']} accepted -> replan queued")
        self.request("new-order")

    def on_cancel_order(self, m):
        oid = m.payload["id"]
        self.cancelled.add(oid)
        self.log(f"order #{oid} cancelled -> removed from plan")
        self.request("cancel")

    def on_breakdown(self, m):
        cid = m.payload["courier"]
        self.broken.add(cid); self.escalated = False
        self.log(f"Courier-{cid} BROKE DOWN -> redistributing {len(m.payload['orders'])} orders")
        self.request("breakdown")

    def on_traffic_update(self, m):
        e = m.payload["event"]
        self.traffic.append(e); self.escalated = False
        self.log(f"traffic jam approved (x{e['factor']:.1f}) -> replan queued")
        self.request("traffic")

    def on_location_update(self, m):
        st = m.payload
        self.status[st["courier"]] = st
        if st.get("reply"):
            self.waiting.discard(st["courier"])
            if not self.waiting and self.replanning:
                self._dispatch()

    def on_receipt_confirm(self, m):
        self.delivered[m.payload["id"]] = m.payload["time"]
        if self.unassigned and not self.replanning and self.bus.now - self.last_replan >= 5:
            self.request("unassigned")

    def on_plan_ready(self, m):
        p = m.payload
        self.replanning, self.has_plan, self.intensify = False, True, False
        self.planned_late, self.unassigned, self.last_replan = p["late"], p["unassigned"], self.bus.now
        self.plans.append(p)
        gain = ""
        if p["baseline_cost"] > 0 and p["mode"] == "aco":
            gain = f", {100 * (1 - p['cost'] / p['baseline_cost']):.0f}% better than nearest-first"
        self.log(f"plan ({p['reason']}): {p['n']} orders, {p['late']} predicted late{gain}")
        if p["late"] > 0 and p["mode"] != "nearest" and not self.escalated and self.policy == "full":
            self.escalated = True
            self.log("late deliveries predicted -> asking Dispatcher for a deeper search")
            self.request("intensify")

    # --- control loop
    def working(self):
        return [c for c in self.courier_ids if c not in self.broken]

    def _undelivered(self):
        return [i for i in self.orders if i not in self.delivered and i not in self.cancelled]

    def _active_traffic(self, now):
        return [e for e in self.traffic if e["time"] <= now < e["time"] + e["duration"]]

    def _projected_late(self, now):
        model = TrafficModel([TrafficEvent(**e) for e in self._active_traffic(now)])
        late = 0
        for cid in self.working():
            st = self.status.get(cid)
            if not st:
                continue
            pos, t = np.array(st["pos"]), st["time"]
            for oid, x, y in st["route"]:
                if oid in self.delivered or oid in self.cancelled:
                    continue
                tgt = np.array([x, y])
                t += dist(pos, tgt) * float(model.factor(pos, tgt)) / st["speed"]
                late += t > self.orders[oid]["due"]
                t += 1.0; pos = tgt
        return late

    def tick(self, now):
        self.now = now
        if self.policy == "fixed_timer":
            if self._undelivered() and not self.replanning and (not self.has_plan or now - self.last_replan >= self.period):
                self.request("timer")
        elif (self.has_plan and not self.replanning and self.flag is None and self.risk_replans < self.max_risk
              and now - self.last_replan >= self.cooldown and self._projected_late(now) >= self.planned_late + 2):
            self.risk_replans += 1
            self.log("courier lateness risk detected -> replan")
            self.request("late-risk")
        if self.flag and not self.replanning:
            self.reason, self.flag = self.flag, None
            self.waiting = set(self.working())
            if not self.waiting:
                return
            self.replanning, self.intensify = True, self.reason == "intensify"
            for cid in self.waiting:
                self.bus.send(STATUS_QUERY, self.name, f"Courier-{cid}")

    def _dispatch(self):
        couriers = []
        for c in self.working():
            st = dict(self.status[c])
            st["route"] = [r for r in st["route"] if r[0] not in self.cancelled and r[0] not in self.delivered]
            couriers.append(st)
        pending = [self.orders[i] for i in self._undelivered()]
        self.bus.send(PLAN_REQUEST, self.name, "Dispatcher", now=self.bus.now, reason=self.reason, couriers=couriers,
                      pending=pending, traffic=self._active_traffic(self.bus.now),
                      full=(not self.has_plan) or self.intensify, intensify=self.intensify)

    def summary(self):
        late = [t - self.orders[i]["due"] for i, t in self.delivered.items() if t > self.orders[i]["due"] + 1e-9]
        return dict(delivered=len(self.delivered), late=len(late), late_min=sum(late), replans=len(self.plans),
                    plan_seconds=sum(p["elapsed"] for p in self.plans), cancelled=len(self.cancelled))
