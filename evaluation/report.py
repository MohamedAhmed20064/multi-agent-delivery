"""Builds results/report.md and results/report.html from the experiment outputs.  python report.py"""
import base64, csv, html, os
R = "results"


def rd(n):
    p = f"{R}/{n}"
    return open(p).read().strip() if os.path.exists(p) else "_(run experiments.py first)_"


def rows(n):
    p = f"{R}/{n}"
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def avg(v):
    v = list(v)
    return sum(v) / len(v) if v else 0.0


def facts():
    out = []
    c = rows("compare.csv")
    if c:
        ks = sorted({int(r["scenario"]) for r in c})
        f = lambda sv, k, key: avg(float(r[key]) for r in c if r["solver"] == sv and int(r["scenario"]) == k)
        imps = [100 * (1 - f("aco", k, "distance") / f("nearest", k, "distance")) for k in ks]
        out.append(f"ACO reduced the total distance by {avg(imps):.1f}% on average versus nearest-customer-first "
                   f"(range {min(imps):.1f}% to {max(imps):.1f}%), and the average finish time by "
                   f"{100 * (1 - avg(f('aco', k, 'makespan') for k in ks) / avg(f('nearest', k, 'makespan') for k in ks)):.0f}%.")
        out.append(f"Average late deliveries per scenario: nearest {avg(f('nearest', k, 'late') for k in ks):.2f}, ACO {avg(f('aco', k, 'late') for k in ks):.2f}, "
                   f"GA {avg(f('ga', k, 'late') for k in ks):.2f}, SA {avg(f('sa', k, 'late') for k in ks):.2f}.")
        out.append(f"Average total distance: ACO {avg(f('aco', k, 'distance') for k in ks):.0f}, GA {avg(f('ga', k, 'distance') for k in ks):.0f}, "
                   f"SA {avg(f('sa', k, 'distance') for k in ks):.0f}, nearest {avg(f('nearest', k, 'distance') for k in ks):.0f}.")
    a = rows("ablation.csv")
    if a:
        g = lambda pol, key: avg(float(r[key]) for r in a if r["policy"] == pol)
        out.append(f"Supervisor ablation (dynamic scenarios 4-6): late minutes {g('full', 'late_min'):.0f} with the full Supervisor, "
                   f"{g('no_risk_escalation', 'late_min'):.0f} without risk detection/escalation, {g('fixed_timer', 'late_min'):.0f} with a fixed 30-minute timer; "
                   f"finish time {g('full', 'makespan'):.0f} / {g('no_risk_escalation', 'makespan'):.0f} / {g('fixed_timer', 'makespan'):.0f} minutes.")
    return out


def blocks():
    B = [("h1", "Multi-Agent Delivery with a Supervisor Agent and Ant Colony Optimisation"),
         ("h2", "1. Problem"),
         ("p", "Customers order deliveries (location + requested delivery time). Couriers have a capacity, a speed and a position. The goal is to assign orders and "
               "plan routes that minimise total distance and late deliveries, and to re-plan when something changes during execution."),
         ("h2", "2. System architecture"),
         ("p", "Five kinds of agents communicate only through messages on a bus: Customer, Courier (x8), Dispatcher, Traffic and the Supervisor, which sits on top of the hierarchy."),
         ("table", "| Agent | Responsibility |\n|---|---|\n"
                   "| Supervisor | Receives orders, cancellations, breakdowns and traffic reports; decides when to re-plan; polls courier positions before each plan; "
                   "validates plans, escalates search effort when lateness is predicted, detects lateness risk, keeps KPIs |\n"
                   "| Dispatcher | Builds routes (ACO or baseline) and sends them to couriers; keeps the pheromone matrix between plans |\n"
                   "| Courier | Executes its route, sends location updates, may break down |\n"
                   "| Customer | Places / cancels an order and confirms receipt |\n"
                   "| Traffic | Slows roads inside congestion zones |"),
         ("p", "Messages: NEW_ORDER, ORDER_ACK, CANCEL_ORDER, BREAKDOWN, TRAFFIC_UPDATE, STATUS_QUERY, LOCATION_UPDATE, PLAN_REQUEST, PLAN_READY, ROUTE_ASSIGN, DELIVERED, RECEIPT_CONFIRM."),
         ("h2", "3. Algorithm"),
         ("p", "Each ant builds courier routes one after another respecting capacity; the next customer is chosen with probability proportional to pheromone^alpha x (1 / (distance + expected lateness penalty))^beta, "
               "with a 5% purely random choice. Cost = distance + 0.5 x late minutes + 15 x late deliveries. Pheromone evaporates (rho = 0.3) and is reinforced by the iteration-best and global-best solutions. "
               "A relocate/swap local search polishes elite solutions. On re-planning the courier's current target is frozen, the remaining orders are re-distributed with half the iterations, "
               "and the existing pheromone is reused instead of starting from scratch."),
         ("h2", "4. Experimental setup"),
         ("p", "Six fixed scenarios (CSV files): uniform, clustered, tight deadlines, traffic jams, live new orders, and courier breakdowns + cancellations; 20 customers and 8 couriers with capacity 3. "
               "Every stochastic solver is run 10 times per scenario in the full simulation (1 step = 1 minute). Nearest-customer is deterministic and run once. "
               "GA and SA get the same evaluation budget as ACO (ants x iterations) and the same final local search; ACO additionally uses periodic local search."),
         ("h2", "5. Results"),
         ("h3", "5.1 ACO vs nearest-customer-first"), ("table", rd("summary.md")), ("img", "comparison.png"),
         ("h3", "5.2 ACO vs Genetic Algorithm and Simulated Annealing (distance / late deliveries)"), ("table", rd("solvers.md")), ("img", "convergence.png"),
         ("h3", "5.3 Does the Supervisor help? (ablation)"), ("table", rd("ablation.md")), ("img", "ablation.png"),
         ("h3", "5.4 Parameter study (scenario 1)"), ("table", rd("tuning.md")),
         ("h3", "5.5 Runtime for 20 / 40 / 60 customers"), ("table", rd("scaling.md")),
         ("h2", "6. Discussion")]
    B += [("p", f) for f in facts()]
    B += [("p", "Nearest-customer-first is a weak baseline, so the comparison with GA and SA is the more informative one. Results come from simulation with synthetic data; "
                "differences of a few percent between ACO and GA should be read together with the standard deviations in the tables."),
          ("h2", "7. Limitations"),
          ("p", "One order per customer; capacity means the maximum number of orders assigned to a courier at once; travel time is straight-line distance scaled by a congestion factor; "
                "synthetic scenarios; SA/GA parameters were not tuned as extensively as ACO's."),
          ("h2", "8. Reproduce"),
          ("p", "pip install numpy matplotlib; python experiments.py all --runs 10; python report.py; python gui.py")]
    return B


def tbl_html(md):
    lines = [l for l in md.splitlines() if l.strip().startswith("|")]
    if len(lines) < 2:
        return f"<p>{html.escape(md)}</p>"
    cell = lambda l: [c.strip().replace("**", "") for c in l.strip().strip("|").split("|")]
    h = "<table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cell(lines[0])) + "</tr>"
    for l in lines[2:]:
        h += "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cell(l)) + "</tr>"
    return h + "</table>"


def build():
    md, hh = [], []
    for kind, v in blocks():
        if kind in ("h1", "h2", "h3"):
            md.append("#" * int(kind[1]) + " " + v + "\n"); hh.append(f"<{kind}>{html.escape(v)}</{kind}>")
        elif kind == "p":
            md.append(v + "\n"); hh.append(f"<p>{html.escape(v)}</p>")
        elif kind == "table":
            md.append(v + "\n"); hh.append(tbl_html(v))
        elif kind == "img" and os.path.exists(f"{R}/{v}"):
            md.append(f"![{v}]({v})\n")
            hh.append(f'<img src="data:image/png;base64,{base64.b64encode(open(f"{R}/{v}", "rb").read()).decode()}" style="max-width:100%">')
    open(f"{R}/report.md", "w").write("\n".join(md))
    css = "body{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:30px auto;line-height:1.5}table{border-collapse:collapse;margin:10px 0;font-size:13px}td,th{border:1px solid #999;padding:4px 8px}th{background:#eee}"
    open(f"{R}/report.html", "w", encoding="utf-8").write(f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{''.join(hh)}</body></html>")
    print("report written: results/report.md, results/report.html (open the .html in Word to save as .docx)")


if __name__ == "__main__":
    build()
