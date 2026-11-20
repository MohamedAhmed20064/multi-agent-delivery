"""Genetic Algorithm and Simulated Annealing on the same Problem/cost, used as comparison baselines.
Representation: permutation of orders + a 'balanced' flag; the decoder cuts it into courier routes."""
import time
import numpy as np


def _decode(prob, perm, bal):
    routes = [[] for _ in range(prob.m)]
    caps = [int(c) for c in prob.cap]
    suffix = [sum(1 for k in range(c, prob.m) if caps[k] > 0) for c in range(prob.m)]
    k, rem = 0, len(perm)
    for c in range(prob.m):
        if rem <= 0:
            break
        cap = caps[c]
        if cap <= 0:
            continue
        if bal:
            cap = min(cap, -(-rem // max(1, suffix[c])))
        take = perm[k:k + cap]
        routes[c] = list(take); k += len(take); rem -= len(take)
    return routes


def _finish(prob, p, best, hist, hist_d, t0):
    best = prob.local_search(best, p.ls_evals)
    if hist:
        hist[-1], hist_d[-1] = min(hist[-1], best.cost), min(hist_d[-1], best.dist)
    best.history, best.history_dist, best.runtime = hist, hist_d, time.perf_counter() - t0
    return best


def solve_ga(prob, p, rng, evals=None, pop=30):
    t0 = time.perf_counter()
    n = prob.n
    ev = lambda ind: prob.evaluate(_decode(prob, ind[0], ind[1]))
    popu = [(list(map(int, rng.permutation(n))), bool(rng.random() < 0.5)) for _ in range(pop)]
    sc = [ev(i) for i in popu]
    gens = max(1, (evals or p.ants * p.iterations) // pop - 1)
    best, hist, hist_d = min(sc, key=lambda s: s.cost), [], []

    def pick():
        i = min((int(x) for x in rng.integers(0, pop, 3)), key=lambda j: sc[j].cost)
        return popu[i]

    for _ in range(gens):
        order = np.argsort([s.cost for s in sc])
        new, ns = [popu[i] for i in order[:2]], [sc[i] for i in order[:2]]
        while len(new) < pop:
            a, b = pick(), pick()
            child = list(a[0])
            if n > 2:
                i, j = sorted(int(x) for x in rng.choice(n, 2, replace=False))
                mid = a[0][i:j]
                rest = [g for g in b[0] if g not in set(mid)]
                child = rest[:i] + mid + rest[i:]
                if rng.random() < 0.3:
                    x, y = (int(v) for v in rng.choice(n, 2, replace=False)); child[x], child[y] = child[y], child[x]
                if rng.random() < 0.1:
                    i, j = sorted(int(x) for x in rng.choice(n, 2, replace=False)); child[i:j] = child[i:j][::-1]
            flag = a[1] if rng.random() < 0.5 else b[1]
            if rng.random() < 0.05:
                flag = not flag
            ind = (child, flag)
            new.append(ind); ns.append(ev(ind))
        popu, sc = new, ns
        cur = min(sc, key=lambda s: s.cost)
        if cur.cost < best.cost:
            best = cur
        hist.append(best.cost); hist_d.append(best.dist)
    return _finish(prob, p, best, hist, hist_d, t0)


def solve_sa(prob, p, rng, evals=None):
    t0 = time.perf_counter()
    n, budget = prob.n, evals or p.ants * p.iterations
    ev = lambda ind: prob.evaluate(_decode(prob, ind[0], ind[1]))
    cur = (list(map(int, rng.permutation(n))), bool(rng.random() < 0.5)); cs = ev(cur); best = cs
    T0, T1 = max(1.0, 0.05 * cs.cost), 0.5
    cool = (T1 / T0) ** (1.0 / max(1, budget))
    T, hist, hist_d, step = T0, [], [], max(1, budget // 60)
    for it in range(budget):
        perm, flag = list(cur[0]), cur[1]
        r = rng.random()
        if n > 1 and r < 0.5:
            x, y = (int(v) for v in rng.choice(n, 2, replace=False)); perm[x], perm[y] = perm[y], perm[x]
        elif n > 1 and r < 0.9:
            x = int(rng.integers(n)); g = perm.pop(x); perm.insert(int(rng.integers(n)), g)
        else:
            flag = not flag
        cand = (perm, flag); s = ev(cand)
        if s.cost < cs.cost or rng.random() < np.exp(-(s.cost - cs.cost) / T):
            cur, cs = cand, s
            if cs.cost < best.cost:
                best = cs
        T *= cool
        if it % step == step - 1:
            hist.append(best.cost); hist_d.append(best.dist)
    return _finish(prob, p, best, hist, hist_d, t0)
