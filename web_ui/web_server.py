"""Web Dashboard Server for Multi-Agent Delivery.
Run: python web_server.py
"""
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from core.scenario import DEPOT, SCENARIO_NAMES, generate_all, load_scenario
from core.simulation import Simulation

if not os.path.exists("data/scenario_6"):
    generate_all("data")

SCNS = [load_scenario(f"data/scenario_{k}") for k in range(1, len(SCENARIO_NAMES) + 1)]
SHOW_EVENTS = {"NEW_ORDER", "TRAFFIC_UPDATE", "PLAN_REQUEST", "RECEIPT_CONFIRM", "BREAKDOWN", "CANCEL_ORDER"}

STATE_LOCK = threading.Lock()
STATE = {
    "scn_idx": 0,
    "solver": "aco",
    "policy": "full",
    "sim": None,
    "compare_mode": False,
    "sim_b": None,
    "solver_b": "nearest",
}


def serialize_sim(s):
    if s is None:
        return {}
    traffic = [
        {"x1": e.x1, "y1": e.y1, "x2": e.x2, "y2": e.y2, "factor": e.factor}
        for e in s.world.active(s.now)
    ]

    owner = {}
    couriers = []
    for k, cr in enumerate(s.couriers):
        pts = [list(cr.pos)] + [[x, y] for _, x, y in cr.route]
        if cr.route or not cr.at_depot():
            pts.append(list(DEPOT))
        for oid, _, _ in cr.route:
            owner[oid] = k
        couriers.append({
            "id": cr.spec.id,
            "idx": k,
            "x": cr.pos[0],
            "y": cr.pos[1],
            "broken": cr.broken,
            "route": pts,
            "orders_count": len(cr.route),
        })

    orders = []
    for oid, o in s.orders.items():
        cu = s.customers.get(oid)
        if cu is None:
            continue
        late = (cu.delivered_at > o.due + 1e-9) if cu.state == "delivered" else (s.now > o.due)
        orders.append({
            "id": oid,
            "x": o.x,
            "y": o.y,
            "due": o.due,
            "state": cu.state,
            "late": late,
            "owner": owner.get(oid, -1),
        })

    feed = []
    for m in s.bus.log[-35:]:
        if m.type in SHOW_EVENTS:
            p = m.payload
            txt_map = {
                "NEW_ORDER": f"Order #{p.get('order', {}).get('id')} placed",
                "TRAFFIC_UPDATE": "Traffic congestion detected on grid",
                "PLAN_REQUEST": f"Dispatcher replan triggered ({p.get('reason')})",
                "RECEIPT_CONFIRM": f"Order #{p.get('id')} delivered & confirmed",
                "BREAKDOWN": f"ALERT: Breakdown! {len(p.get('orders', []))} orders stranded",
                "CANCEL_ORDER": f"Order #{p.get('id')} cancelled by customer",
            }
            feed.append({
                "time": round(m.time, 1),
                "sender": m.sender,
                "receiver": m.receiver,
                "type": m.type,
                "text": txt_map.get(m.type, m.type),
                "is_sup": False,
            })

    for t, text in s.supervisor.decisions[-15:]:
        feed.append({
            "time": round(t, 1),
            "sender": "SUPERVISOR",
            "receiver": "FLEET",
            "type": "SUPERVISOR",
            "text": text,
            "is_sup": True,
        })
    feed.sort(key=lambda x: x["time"])

    last = s.dispatcher.last or {}
    return {
        "now": round(s.now, 1),
        "done": s.done,
        "depot": list(DEPOT),
        "traffic": traffic,
        "couriers": couriers,
        "orders": orders,
        "summary": s.summary(),
        "solver": s.dispatcher.mode.upper(),
        "feed": feed[-25:],
        "convergence": {
            "history": last.get("history", []),
            "baseline": last.get("baseline_cost", 0),
        },
    }


def reset_simulations():
    scn = SCNS[STATE["scn_idx"]]
    seed = random.randint(0, 9999)
    STATE["sim"] = Simulation(scn, STATE["solver"], seed=seed, policy=STATE["policy"])
    if STATE["compare_mode"]:
        STATE["sim_b"] = Simulation(scn, STATE["solver_b"], seed=seed, policy=STATE["policy"])
    else:
        STATE["sim_b"] = None


reset_simulations()


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
            with open(html_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif parsed.path == "/api/state":
            with STATE_LOCK:
                self._send_json({
                    "scenarios": SCENARIO_NAMES,
                    "scn_idx": STATE["scn_idx"],
                    "solver": STATE["solver"],
                    "solver_b": STATE["solver_b"],
                    "policy": STATE["policy"],
                    "compare_mode": STATE["compare_mode"],
                    "primary": serialize_sim(STATE["sim"]),
                    "secondary": serialize_sim(STATE["sim_b"]) if STATE["compare_mode"] else None,
                })
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        path = urlparse(self.path).path

        with STATE_LOCK:
            sims = [STATE["sim"]] + ([STATE["sim_b"]] if STATE["compare_mode"] and STATE["sim_b"] else [])

            if path == "/api/reset":
                STATE["scn_idx"] = int(payload.get("scn_idx", STATE["scn_idx"]))
                STATE["solver"] = payload.get("solver", STATE["solver"])
                STATE["solver_b"] = payload.get("solver_b", STATE["solver_b"])
                STATE["policy"] = payload.get("policy", STATE["policy"])
                STATE["compare_mode"] = bool(payload.get("compare_mode", STATE["compare_mode"]))
                reset_simulations()

            elif path == "/api/step":
                for s in sims:
                    if s and not s.done:
                        s.step()

            elif path == "/api/order":
                x = float(payload.get("x", random.uniform(8, 92)))
                y = float(payload.get("y", random.uniform(8, 92)))
                for s in sims:
                    if s:
                        s.add_order(x, y, 40.0)

            elif path == "/api/jam":
                x, y = random.uniform(10, 60), random.uniform(10, 60)
                for s in sims:
                    if s:
                        s.add_traffic(x, y, x + 35, y + 35, 2.5, 40.0)

            elif path == "/api/breakdown":
                ids = [c.spec.id for c in STATE["sim"].couriers if not c.broken]
                if len(ids) > 1:
                    target = random.choice(ids)
                    for s in sims:
                        if s:
                            s.trigger("breakdown", target)

            elif path == "/api/cancel":
                ids = [i for i, c in STATE["sim"].customers.items() if c.state not in ("delivered", "cancelled")]
                if ids:
                    target = random.choice(ids)
                    for s in sims:
                        if s:
                            s.trigger("cancel", target)

            self._send_json({
                "primary": serialize_sim(STATE["sim"]),
                "secondary": serialize_sim(STATE["sim_b"]) if STATE["compare_mode"] else None,
            })


if __name__ == "__main__":
    port = 8050
    server = ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)
    url = f"http://127.0.0.1:{port}"
    print(f"Multi-Agent Delivery Dashboard running at {url}")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    server.serve_forever()