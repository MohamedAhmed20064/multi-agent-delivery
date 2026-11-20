"""Discrete-time simulation (1 step = 1 minute) that wires all agents together."""
from core.agents import (MessageBus, SupervisorAgent, DispatcherAgent, CourierAgent, CustomerAgent, TrafficAgent)
from core.scenario import Order, TrafficEvent, TrafficModel


class Simulation:
    def __init__(self, scn, solver="aco", params=None, seed=0, dt=1.0, policy="full"):
        self.scn, self.dt, self.now = scn, dt, 0.0
        self.bus = MessageBus()
        self.world = TrafficModel()                      # real traffic (what couriers experience)
        self.supervisor = SupervisorAgent(self.bus, [c.id for c in scn.couriers], policy=policy)
        self.dispatcher = DispatcherAgent(self.bus, solver, params, seed)
        self.couriers = [CourierAgent(self.bus, c, self.world) for c in scn.couriers]
        self.traffic = TrafficAgent(self.bus, self.world, [TrafficEvent(**vars(e)) for e in scn.traffic])
        self.orders = {o.id: o for o in scn.orders}
        self.queue = sorted(scn.orders, key=lambda o: o.request)
        self.customers = {}
        self.events = sorted(scn.events)

    def add_order(self, x, y, due_in=40.0):
        o = Order(max(self.orders) + 1, round(x, 1), round(y, 1), round(self.now + due_in, 1), self.now)
        self.orders[o.id] = o; self.queue.append(o)

    def add_traffic(self, x1, y1, x2, y2, factor=2.5, duration=40.0):
        self.traffic.pending.append(TrafficEvent(self.now, x1, y1, x2, y2, factor, duration))

    def trigger(self, kind, target):
        if kind == "breakdown":
            c = next((c for c in self.couriers if c.spec.id == target), None)
            if c and not c.broken:
                c.breakdown()
        elif kind == "cancel":
            cu = self.customers.get(target)
            if cu and cu.state not in ("delivered", "cancelled"):
                cu.cancel()

    def random_breakdown(self, rng):
        alive = [c for c in self.couriers if not c.broken]
        ok = [c for c in alive if c.route] or alive
        if len(alive) > 1 and ok:
            self.trigger("breakdown", rng.choice(ok).spec.id)

    def random_cancel(self, rng):
        ok = [i for i, c in self.customers.items() if c.state not in ("delivered", "cancelled")]
        if ok:
            self.trigger("cancel", rng.choice(ok))

    @property
    def done(self):
        return (not self.queue and not self.events
                and all(c.state in ("delivered", "cancelled") for c in self.customers.values())
                and all(c.broken or (not c.route and c.at_depot()) for c in self.couriers))

    def step(self):
        b = self.bus; b.now = self.now
        while self.queue and self.queue[0].request <= self.now + 1e-9:
            o = self.queue.pop(0)
            self.customers[o.id] = CustomerAgent(b, o)
            self.customers[o.id].place_order()
        while self.events and self.events[0][0] <= self.now + 1e-9:
            _, kind, target = self.events.pop(0)
            self.trigger(kind, target)
        self.traffic.step(self.now)
        b.drain(); self.supervisor.tick(self.now); b.drain()
        for c in self.couriers:
            c.step(self.now, self.dt)
        b.drain()
        self.now += self.dt
        b.now = self.now
        for c in self.couriers:
            c.tick(self.now)
        b.drain()

    def run(self, max_time=1000):
        while not self.done and self.now < max_time:
            self.step()
        return self.summary()

    def summary(self):
        s = self.supervisor.summary()
        s.update(distance=sum(c.odometer for c in self.couriers), makespan=self.now, total=len(self.orders) - len(self.supervisor.cancelled))
        return s
