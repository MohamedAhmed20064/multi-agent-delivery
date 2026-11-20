"""Ant Colony Optimisation for the multi-courier delivery problem (capacity + time-window penalties)."""
import time
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ACOParams:
    ants: int = 20
    iterations: int = 60
    alpha: float = 1.0          # pheromone weight
    beta: float = 3.0           # heuristic weight
    rho: float = 0.3            # evaporation rate
    q: float = 100.0
    explore: float = 0.05       # small probability of a purely random choice (avoids stagnation)
    balance_prob: float = 0.5   # share of ants that spread orders evenly across couriers
    w_late: float = 0.5         # cost per minute of lateness
    w_count: float = 15.0       # cost per late delivery
    unassigned_penalty: float = 500.0
    tau_min: float = 0.1
    tau_max: float = 5.0
    ls_every: int = 10
    ls_evals: int = 4000
    service: float = 1.0        # minutes spent at each customer


@dataclass
class Solution:
    routes: list
    dist: float
    late_min: float
    late_cnt: int
    unassigned: int
    cost: float
    history: list = field(default_factory=list)
    history_dist: list = field(default_factory=list)
    runtime: float = 0.0


class Problem:
    """n orders, m couriers (each with start point, start time, speed, free capacity)."""

    def __init__(self, pts, due, starts, t0, speed, cap, depot, factor_fn, params):
        self.p = params
        self.n, self.m = len(pts), len(starts)
        P = np.asarray(pts, float).reshape(-1, 2)
        S = np.asarray(starts, float).reshape(-1, 2)
        self.due = np.asarray(due, float)
        self.t0, self.speed = list(map(float, t0)), list(map(float, speed))
        self.cap = np.asarray(cap, int)
        self.D = np.linalg.norm(P[:, None] - P[None], axis=2)
        self.TT = self.D * factor_fn(P[:, None], P[None])
        self.DS = np.linalg.norm(S[:, None] - P[None], axis=2)
        self.TS = self.DS * factor_fn(S[:, None], P[None])
        self.DD = np.linalg.norm(P - np.asarray(depot, float), axis=1)
        self.Dl, self.TTl, self.DSl, self.TSl = self.D.tolist(), self.TT.tolist(), self.DS.tolist(), self.TS.tolist()
        self.DDl, self.duel = self.DD.tolist(), self.due.tolist()

    # ---- evaluation -------------------------------------------------
    def route_stats(self, c, route):
        if not route:
            return 0.0, 0.0, 0
        sp, svc = self.speed[c], self.p.service
        j = route[0]
        d = self.DSl[c][j]
        t = self.t0[c] + self.TSl[c][j] / sp
        late = t - self.duel[j]
        lm, lc = (late, 1) if late > 0 else (0.0, 0)
        t += svc
        for a, b in zip(route, route[1:]):
            d += self.Dl[a][b]
            t += self.TTl[a][b] / sp
            late = t - self.duel[b]
            if late > 0:
                lm += late; lc += 1
            t += svc
        return d + self.DDl[route[-1]], lm, lc

    def rcost(self, c, route):
        d, lm, lc = self.route_stats(c, route)
        return d + self.p.w_late * lm + self.p.w_count * lc

    def evaluate(self, routes):
        d = lm = 0.0; lc = assigned = 0
        for c, r in enumerate(routes):
            a, b, k = self.route_stats(c, r)
            d += a; lm += b; lc += k; assigned += len(r)
        un = self.n - assigned
        cost = d + self.p.w_late * lm + self.p.w_count * lc + self.p.unassigned_penalty * un
        return Solution([list(r) for r in routes], d, lm, lc, un, cost)

    # ---- baseline: nearest customer first ---------------------------
    def nearest(self):
        t = time.perf_counter()
        free = np.ones(self.n, bool); routes = [[] for _ in range(self.m)]
        for c in range(self.m):
            cap, prev = int(self.cap[c]), -1
            while cap > 0 and free.any():
                cand = np.flatnonzero(free)
                d = self.DS[c, cand] if prev < 0 else self.D[prev, cand]
                j = int(cand[np.argmin(d)])
                routes[c].append(j); free[j] = False; prev = j; cap -= 1
        s = self.evaluate(routes); s.runtime = time.perf_counter() - t
        return s

    # ---- one ant ----------------------------------------------------
    def construct(self, tau, p, rng):
        free = np.ones(self.n, bool)
        routes = [[] for _ in range(self.m)]
        balanced = rng.random() < p.balance_prob
        for idx, c in enumerate(rng.permutation(self.m)):
            cap = int(self.cap[c])
            if balanced:
                cap = min(cap, int(np.ceil(free.sum() / (self.m - idx))))
            t, prev, sp = self.t0[c], -1, self.speed[c]
            while cap > 0 and free.any():
                cand = np.flatnonzero(free)
                if prev < 0:
                    d, tt, row = self.DS[c, cand], self.TS[c, cand] / sp, tau[0, cand + 1]
                else:
                    d, tt, row = self.D[prev, cand], self.TT[prev, cand] / sp, tau[prev + 1, cand + 1]
                late = np.maximum(0.0, t + tt - self.due[cand])
                eta = 1.0 / (d + p.w_late * late + 1e-3)
                w = (row ** p.alpha) * (eta ** p.beta) + 1e-300
                if rng.random() < p.explore:
                    j = int(rng.choice(cand))
                else:
                    j = int(cand[rng.choice(len(cand), p=w / w.sum())])
                t += tt[list(cand).index(j)] + p.service
                routes[c].append(j); free[j] = False; prev = j; cap -= 1
        return routes

    # ---- local search (relocate + swap) -----------------------------
    def local_search(self, sol, max_evals):
        routes = [list(r) for r in sol.routes]
        rc = [self.rcost(c, r) for c, r in enumerate(routes)]
        evals = 0
        improved = True
        while improved and evals < max_evals:
            improved = False
            for a in range(self.m):
                for i in range(len(routes[a])):
                    x = routes[a][i]
                    ra = routes[a][:i] + routes[a][i + 1:]
                    ca = self.rcost(a, ra)
                    for b in range(self.m):
                        if b != a and len(routes[b]) >= self.cap[b]:
                            continue
                        base = ra if b == a else routes[b]
                        for pos in range(len(base) + 1):
                            nb = base[:pos] + [x] + base[pos:]
                            evals += 1
                            if b == a:
                                new = self.rcost(a, nb); delta = new - rc[a]
                            else:
                                new = self.rcost(b, nb); delta = ca + new - rc[a] - rc[b]
                            if delta < -1e-6:
                                if b == a:
                                    routes[a] = nb; rc[a] = new
                                else:
                                    routes[a] = ra; rc[a] = ca; routes[b] = nb; rc[b] = new
                                improved = True
                                break
                        if improved: break
                    if improved: break
                if improved: break
            if improved: continue
            for a in range(self.m):
                for b in range(a + 1, self.m):
                    for i in range(len(routes[a])):
                        for j in range(len(routes[b])):
                            ra, rb = list(routes[a]), list(routes[b])
                            ra[i], rb[j] = rb[j], ra[i]
                            evals += 1
                            na, nb = self.rcost(a, ra), self.rcost(b, rb)
                            if na + nb < rc[a] + rc[b] - 1e-6:
                                routes[a], routes[b], rc[a], rc[b] = ra, rb, na, nb
                                improved = True
                                break
                        if improved: break
                    if improved: break
                if improved: break
        out = self.evaluate(routes)
        return out if out.cost < sol.cost else sol


def solve_aco(prob, p, tau=None, iterations=None, seed=None, rng=None):
    """Returns (best Solution, final pheromone matrix). Node 0 = 'start', order j -> index j+1."""
    rng = rng or np.random.default_rng(seed)
    t_start = time.perf_counter()
    tau = np.ones((prob.n + 1, prob.n + 1)) if tau is None else np.array(tau, float)
    best, hist, hist_d = None, [], []
    for it in range(iterations or p.iterations):
        sols = [prob.evaluate(prob.construct(tau, p, rng)) for _ in range(p.ants)]
        ib = min(sols, key=lambda s: s.cost)
        if p.ls_every and it % p.ls_every == p.ls_every - 1:
            ib = prob.local_search(ib, p.ls_evals)
        if best is None or ib.cost < best.cost:
            best = ib
        tau *= 1 - p.rho
        for s, w in ((ib, 1.0), (best, 2.0)):
            for r in s.routes:
                prev = 0
                for j in r:
                    tau[prev, j + 1] += w * p.q / s.cost
                    prev = j + 1
        np.clip(tau, p.tau_min, p.tau_max, out=tau)
        hist.append(best.cost); hist_d.append(best.dist)
    best = prob.local_search(best, p.ls_evals)
    if hist:
        hist[-1], hist_d[-1] = min(hist[-1], best.cost), min(hist_d[-1], best.dist)
    best.history, best.history_dist = hist, hist_d
    best.runtime = time.perf_counter() - t_start
    return best, tau
