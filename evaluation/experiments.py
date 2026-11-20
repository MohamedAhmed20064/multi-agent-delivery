"""Evaluation.  python experiments.py [compare|ablation|tuning|scaling|all] [--runs N]
compare : Nearest vs ACO vs GA vs SA on 6 scenarios (full simulation, mean/std/min/max/CI)
ablation: Supervisor policies (full / no_risk_escalation / fixed_timer) on the dynamic scenarios
tuning  : ants / iterations / evaporation        scaling: runtime for 20/40/60 customers"""
import argparse, csv, math, os
from dataclasses import replace
import numpy as np
from algorithms.aco import ACOParams, Problem, solve_aco
from algorithms.metaheuristics import solve_ga, solve_sa
from core.scenario import generate_all, load_scenario, random_scenario, TrafficModel, DEPOT, SCENARIO_NAMES
from core.simulation import Simulation

OUT = "results"
SOLVERS = ["nearest", "aco", "ga", "sa"]
POLICIES = ["full", "no_risk_escalation", "fixed_timer"]


def run_sim(scn, solver, params, seed, policy="full"):
    return Simulation(scn, solver, params, seed, policy=policy).run()


def static_problem(scn, params):
    o = [x for x in scn.orders if x.request == 0]
    return Problem([(x.x, x.y) for x in o], [x.due for x in o], [DEPOT] * len(scn.couriers), [0.0] * len(scn.couriers),
                   [c.speed for c in scn.couriers], [c.capacity for c in scn.couriers], DEPOT, TrafficModel().factor, params)


def mean(v):
    return sum(v) / len(v)


def std(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0


def ci95(v):
    return 1.96 * std(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def write_csv(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def write_md(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    open(f"{OUT}/{name}", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def compare(runs=10, params=None):
    params = params or ACOParams()
    rows, data = [], {}
    for k in range(1, len(SCENARIO_NAMES) + 1):
        scn = load_scenario(f"data/scenario_{k}")
        for solver in SOLVERS:
            n = 1 if solver == "nearest" else runs        # nearest-customer is deterministic
            for r in range(n):
                s = run_sim(scn, solver, params, r)
                rows.append([k, solver, r, s["distance"], s["late"], s["late_min"], s["makespan"], s["plan_seconds"]])
                data.setdefault((k, solver), []).append(s)
        print(f"scenario {k} done", flush=True)
    write_csv("compare.csv", ["scenario", "solver", "run", "distance", "late", "late_min", "makespan", "plan_seconds"], rows)
    col = lambda k, sv, key: [s[key] for s in data[(k, sv)]]
    main, solv, imps = [], [], []
    for k in range(1, len(SCENARIO_NAMES) + 1):
        d = col(k, "aco", "distance"); nd = col(k, "nearest", "distance")[0]; imp = 100 * (1 - mean(d) / nd); imps.append(imp)
        main.append([SCENARIO_NAMES[k - 1], f"{nd:.0f}", f"{mean(d):.0f} ± {std(d):.0f} [{min(d):.0f}-{max(d):.0f}]", f"{imp:.1f}%",
                     f"±{ci95(d):.0f}", col(k, "nearest", "late")[0], f"{mean(col(k, 'aco', 'late')):.1f}",
                     f"{col(k, 'nearest', 'makespan')[0]:.0f}", f"{mean(col(k, 'aco', 'makespan')):.0f}"])
        solv.append([SCENARIO_NAMES[k - 1]] + [f"{mean(col(k, sv, 'distance')):.0f} / {mean(col(k, sv, 'late')):.1f}" for sv in SOLVERS])
    main.append(["**Average**", "", "", f"**{mean(imps):.1f}%**", "", "", "", "", ""])
    write_md("summary.md", ["Scenario", "Nearest dist", "ACO dist mean ± std [min-max]", "Improvement", "95% CI", "Late (nearest)", "Late (ACO)",
                            "Finish N", "Finish ACO"], main)
    best = sum(1 for k in range(1, len(SCENARIO_NAMES) + 1)
               if mean(col(k, "aco", "distance")) <= min(mean(col(k, s, "distance")) for s in ("ga", "sa")))
    write_md("solvers.md", ["Scenario (distance / late)", "Nearest", "ACO", "GA", "SA"], solv)
    open(f"{OUT}/solvers_note.txt", "w").write(f"{best}/{len(SCENARIO_NAMES)}")
    return data


def ablation(runs=10, params=None):
    params = params or ACOParams()
    rows, tab = [], []
    for k in (4, 5, 6):
        scn = load_scenario(f"data/scenario_{k}")
        for pol in POLICIES:
            res = [run_sim(scn, "aco", params, r, pol) for r in range(runs)]
            for r, s in enumerate(res):
                rows.append([k, pol, r, s["distance"], s["late"], s["late_min"], s["makespan"], s["replans"]])
            tab.append([SCENARIO_NAMES[k - 1], pol, f"{mean([s['distance'] for s in res]):.0f}",
                        f"{mean([s['late'] for s in res]):.1f} ± {std([s['late'] for s in res]):.1f}",
                        f"{mean([s['late_min'] for s in res]):.0f}", f"{mean([s['makespan'] for s in res]):.0f}",
                        f"{mean([s['replans'] for s in res]):.1f}"])
        print(f"ablation scenario {k} done", flush=True)
    write_csv("ablation.csv", ["scenario", "policy", "run", "distance", "late", "late_min", "makespan", "replans"], rows)
    write_md("ablation.md", ["Scenario", "Supervisor policy", "Distance", "Late deliveries", "Late minutes", "Finish time", "Replans"], tab)


def tuning(seeds=5):
    scn = load_scenario("data/scenario_1")
    grid = [("ants", v) for v in (10, 20, 40)] + [("iterations", v) for v in (30, 60, 100)] + [("rho", v) for v in (0.1, 0.3, 0.5)]
    rows = []
    for name, v in grid:
        p = replace(ACOParams(), **{name: v})
        sols = [solve_aco(static_problem(scn, p), p, seed=s)[0] for s in range(seeds)]
        rows.append([name, v, f"{mean([s.dist for s in sols]):.1f}", f"{std([s.dist for s in sols]):.1f}", f"{mean([s.runtime for s in sols]):.2f}"])
    write_csv("tuning.csv", ["param", "value", "avg_distance", "std", "avg_seconds"], rows)
    write_md("tuning.md", ["Parameter", "Value", "Avg distance", "Std", "Avg seconds"], rows)


def scaling(params=None):
    params = params or ACOParams()
    rows = []
    for n in (20, 40, 60):
        prob = static_problem(random_scenario(n, 1), params)
        nn = prob.nearest(); sol, _ = solve_aco(prob, params, seed=0)
        ga = solve_ga(prob, params, np.random.default_rng(0)); sa = solve_sa(prob, params, np.random.default_rng(0))
        rows.append([n, f"{nn.dist:.0f}", f"{nn.runtime:.3f}", f"{sol.dist:.0f}", f"{sol.runtime:.2f}", f"{ga.dist:.0f}", f"{ga.runtime:.2f}", f"{sa.dist:.0f}", f"{sa.runtime:.2f}"])
    h = ["Customers", "Nearest dist", "Nearest s", "ACO dist", "ACO s", "GA dist", "GA s", "SA dist", "SA s"]
    write_csv("scaling.csv", h, rows); write_md("scaling.md", h, rows)


def plots():
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return print("matplotlib not installed -> skipping plots")
    p = ACOParams(); scn = load_scenario("data/scenario_1")
    prob = static_problem(scn, p); base = prob.nearest().dist
    plt.figure(figsize=(6.4, 3.8))
    for name, fn in (("ACO", lambda s: solve_aco(prob, p, seed=s)[0]), ("GA", lambda s: solve_ga(prob, p, np.random.default_rng(s))),
                     ("SA", lambda s: solve_sa(prob, p, np.random.default_rng(s)))):
        h = np.mean([fn(s).history_dist for s in range(5)], axis=0)
        plt.plot(np.linspace(0, p.ants * p.iterations, len(h)), h, label=name)
    plt.axhline(base, ls="--", color="gray", label="nearest-customer"); plt.xlabel("evaluated solutions"); plt.ylabel("best total distance")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/convergence.png", dpi=150); plt.close()
    if os.path.exists(f"{OUT}/compare.csv"):
        rows = list(csv.DictReader(open(f"{OUT}/compare.csv")))
        ks = sorted({int(r["scenario"]) for r in rows}); plt.figure(figsize=(8, 3.8))
        for i, sv in enumerate(SOLVERS):
            plt.bar(np.arange(len(ks)) + (i - 1.5) * .2, [mean([float(r["distance"]) for r in rows if r["solver"] == sv and int(r["scenario"]) == k]) for k in ks], .2, label=sv)
        plt.xticks(range(len(ks)), [str(k) for k in ks]); plt.xlabel("scenario"); plt.ylabel("avg total distance"); plt.legend()
        plt.tight_layout(); plt.savefig(f"{OUT}/comparison.png", dpi=150); plt.close()
    if os.path.exists(f"{OUT}/ablation.csv"):
        rows = list(csv.DictReader(open(f"{OUT}/ablation.csv"))); fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
        for j, (key, title) in enumerate((("late", "late deliveries"), ("makespan", "finish time (min)"))):
            for i, pol in enumerate(POLICIES):
                ax[j].bar(np.arange(3) + (i - 1) * .27, [mean([float(r[key]) for r in rows if r["policy"] == pol and int(r["scenario"]) == k]) for k in (4, 5, 6)], .27, label=pol)
            ax[j].set_xticks(range(3)); ax[j].set_xticklabels(["4", "5", "6"]); ax[j].set_xlabel("scenario"); ax[j].set_title(title)
        ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OUT}/ablation.png", dpi=150); plt.close()
    print("plots saved in results/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what", nargs="?", default="all"); ap.add_argument("--runs", type=int, default=10)
    a = ap.parse_args()
    if not os.path.exists("data/scenario_6"):
        generate_all("data")
    if a.what in ("tuning", "all"): tuning()
    if a.what in ("scaling", "all"): scaling()
    if a.what in ("compare", "all"): compare(a.runs)
    if a.what in ("ablation", "all"): ablation(a.runs)
    plots()
    if a.what == "all":
        import evaluation.report as report; report.build()
