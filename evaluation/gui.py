"""Tkinter dashboard.  python gui.py
Main window: one simulation with live events.  'Live compare' opens two simulations side by side on the same scenario."""
import os, random
import tkinter as tk
from tkinter import ttk, messagebox
from core.scenario import generate_all, load_scenario, DEPOT, SCENARIO_NAMES
from core.simulation import Simulation

BG, CARD, FG, MUT, GOLD, RED, GREEN = "#0E1621", "#16222F", "#E8EEF5", "#A9B7C6", "#F5B83D", "#F87171", "#4ADE80"
COLORS = ["#4FD1C5", "#F6A05A", "#B79CFF", "#6CB6FF", "#F472B6", "#A3E635", "#FB7185", "#38BDF8"]
SOLVER_LABELS = ["ACO", "Nearest customer", "Genetic Algorithm", "Simulated Annealing"]
SOLVER_KEYS = ["aco", "nearest", "ga", "sa"]
POLICIES = ["full", "no_risk_escalation", "fixed_timer"]
SHOW = {"NEW_ORDER", "TRAFFIC_UPDATE", "PLAN_REQUEST", "RECEIPT_CONFIRM", "BREAKDOWN", "CANCEL_ORDER"}


def draw_sim(c, s, size):
    pad = 18
    sc = (size - 2 * pad) / 100
    px = lambda x, y: (pad + x * sc, pad + y * sc)
    r = max(3, size / 110)
    c.delete("all")
    for g in range(0, 101, 20):
        c.create_line(*px(g, 0), *px(g, 100), fill="#1C2B3B"); c.create_line(*px(0, g), *px(100, g), fill="#1C2B3B")
    for e in s.world.active(s.now):
        c.create_rectangle(*px(e.x1, e.y1), *px(e.x2, e.y2), fill="#4B2530", outline=RED, stipple="gray50")
        c.create_text(*px(e.x1 + 2, e.y1 + 3), text=f"traffic x{e.factor:.1f}", fill=RED, anchor="w", font=("Segoe UI", 8))
    owner = {}
    for k, cr in enumerate(s.couriers):
        if cr.broken:
            continue
        pts = [tuple(cr.pos)] + [(x, y) for _, x, y in cr.route] + ([DEPOT] if cr.route or not cr.at_depot() else [])
        if len(pts) > 1:
            c.create_line(*[v for p in pts for v in px(*p)], fill=COLORS[k % 8], width=2)
        for oid, _, _ in cr.route:
            owner[oid] = k
    for oid, o in s.orders.items():
        cu = s.customers.get(oid)
        if cu is None:
            continue
        x, y = px(o.x, o.y)
        if cu.state == "cancelled":
            c.create_line(x - r, y - r, x + r, y + r, fill=MUT); c.create_line(x - r, y + r, x + r, y - r, fill=MUT)
        elif cu.state == "delivered":
            late = cu.delivered_at > o.due + 1e-9
            c.create_oval(x - r * .7, y - r * .7, x + r * .7, y + r * .7, fill=RED if late else GREEN, outline="")
        else:
            col = COLORS[owner[oid] % 8] if oid in owner else "#6E8196"
            c.create_oval(x - r, y - r, x + r, y + r, fill=FG, outline=col, width=3)
            c.create_text(x + r + 5, y - r - 4, text=str(oid), fill=MUT, font=("Segoe UI", 8))
    dx, dy = px(*DEPOT)
    c.create_rectangle(dx - 10, dy - 10, dx + 10, dy + 10, fill=GOLD, outline=""); c.create_text(dx, dy, text="D", font=("Segoe UI", 10, "bold"))
    for k, cr in enumerate(s.couriers):
        x, y = px(*cr.pos)
        c.create_rectangle(x - 8, y - 8, x + 8, y + 8, fill=RED if cr.broken else COLORS[k % 8], outline=BG, width=2)
        c.create_text(x, y, text="X" if cr.broken else str(cr.spec.id), font=("Segoe UI", 8, "bold"), fill="#0E1621")


def kpi_text(s):
    sm = s.summary()
    return (f"time {s.now:5.0f} min   orders {sm['delivered']}/{sm['total']}   cancelled {sm['cancelled']}\n"
            f"late {sm['late']}   distance {sm['distance']:.0f}   replans {sm['replans']}\n"
            f"solver {s.dispatcher.mode.upper()}   planning {sm['plan_seconds']:.1f}s" + ("   [DONE]" if s.done else ""))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multi-Agent Delivery - Supervisor Dashboard")
        self.configure(bg=BG); self.geometry("1220x780")
        if not os.path.exists("data/scenario_6"):
            generate_all("data")
        self.scns = [load_scenario(f"data/scenario_{k}") for k in range(1, len(SCENARIO_NAMES) + 1)]
        self.sim, self.running, self.add_mode, self.feed_pos, self.dec_pos = None, False, False, 0, 0
        self.size = 660
        self._ui(); self.reset()

    def _ui(self):
        self.cv = tk.Canvas(self, width=self.size, height=self.size, bg="#101B27", highlightthickness=0)
        self.cv.grid(row=0, column=0, padx=12, pady=12, sticky="n"); self.cv.bind("<Button-1>", self.on_click)
        side = tk.Frame(self, bg=BG); side.grid(row=0, column=1, sticky="ns", pady=12, padx=(0, 12))
        tk.Label(side, text="Multi-Agent Delivery", bg=BG, fg=FG, font=("Segoe UI", 18, "bold")).pack(anchor="w")
        tk.Label(side, text="Supervisor-led agents + ACO", bg=BG, fg=MUT).pack(anchor="w", pady=(0, 8))
        row = tk.Frame(side, bg=BG); row.pack(fill="x")
        self.scn_box = ttk.Combobox(row, values=SCENARIO_NAMES, state="readonly", width=22); self.scn_box.current(0); self.scn_box.pack(side="left")
        self.solver_box = ttk.Combobox(row, values=SOLVER_LABELS, state="readonly", width=18); self.solver_box.current(0); self.solver_box.pack(side="left", padx=6)
        row2 = tk.Frame(side, bg=BG); row2.pack(fill="x", pady=(4, 0))
        tk.Label(row2, text="Supervisor policy", bg=BG, fg=MUT).pack(side="left")
        self.pol_box = ttk.Combobox(row2, values=POLICIES, state="readonly", width=20); self.pol_box.current(0); self.pol_box.pack(side="left", padx=6)
        b = tk.Frame(side, bg=BG); b.pack(fill="x", pady=8)
        self.btns = {}
        for i, (t, f) in enumerate([("Start / Restart", self.restart), ("Pause / Resume", self.toggle), ("Step", self.step_once),
                                     ("+ New order", self.toggle_add), ("Traffic jam", self.jam), ("Breakdown", self.breakdown),
                                     ("Cancel order", self.cancel), ("Live compare", self.live_compare), ("Results table", self.compare)]):
            btn = tk.Button(b, text=t, command=f, bg=GOLD if i == 0 else "#1D2D3E", fg="#1A1204" if i == 0 else FG, relief="flat", padx=6, pady=6)
            btn.grid(row=i // 3, column=i % 3, padx=3, pady=3, sticky="ew"); self.btns[t] = btn
        tk.Label(side, text="Speed", bg=BG, fg=MUT).pack(anchor="w")
        self.speed = tk.Scale(side, from_=1, to=30, orient="horizontal", bg=BG, fg=FG, highlightthickness=0, troughcolor=CARD); self.speed.set(8); self.speed.pack(fill="x")
        self.kpi = tk.StringVar()
        tk.Label(side, textvariable=self.kpi, bg=CARD, fg=FG, font=("Consolas", 10), justify="left", anchor="w", padx=10, pady=8).pack(fill="x", pady=6)
        tk.Label(side, text="Supervisor feed", bg=BG, fg=GOLD).pack(anchor="w")
        self.feed = tk.Text(side, width=56, height=10, bg=CARD, fg=FG, font=("Consolas", 9), relief="flat", wrap="word")
        self.feed.pack(fill="x"); self.feed.tag_config("sup", foreground=GOLD)
        tk.Label(side, text="Solver convergence (last plan)", bg=BG, fg=MUT).pack(anchor="w", pady=(6, 0))
        self.conv = tk.Canvas(side, width=440, height=120, bg=CARD, highlightthickness=0); self.conv.pack()

    # ---- control
    def reset(self):
        self.sim = Simulation(self.scns[self.scn_box.current()], SOLVER_KEYS[self.solver_box.current()],
                              seed=random.randint(0, 9999), policy=self.pol_box.get())
        self.feed.delete("1.0", "end"); self.feed_pos = self.dec_pos = 0; self.running = False
        self.draw()

    def restart(self):
        self.reset(); self.running = True; self.loop()

    def toggle(self):
        self.running = not self.running
        if self.running: self.loop()

    def step_once(self):
        self.running = False; self.tick()

    def toggle_add(self):
        self.add_mode = not self.add_mode
        self.btns["+ New order"].config(text="Click on map..." if self.add_mode else "+ New order")

    def on_click(self, e):
        if self.add_mode:
            sc = (self.size - 36) / 100
            self.sim.add_order((e.x - 18) / sc, (e.y - 18) / sc, 40)
            self.toggle_add()
            if not self.running: self.running = True; self.loop()

    def _resume(self):
        if not self.running and not self.sim.done:
            self.running = True; self.loop()

    def jam(self):
        x, y = random.uniform(10, 60), random.uniform(10, 60)
        self.sim.add_traffic(x, y, x + 35, y + 35, 2.5, 40); self._resume()

    def breakdown(self):
        self.sim.random_breakdown(random); self._resume()

    def cancel(self):
        self.sim.random_cancel(random); self._resume()

    def live_compare(self):
        CompareWindow(self, self.scns[self.scn_box.current()], self.speed.get())

    def compare(self):
        from evaluation.experiments import run_sim
        from algorithms.aco import ACOParams
        self.title("Running comparison...")
        self.update_idletasks()
        scn, P = self.scns[self.scn_box.current()], ACOParams()
        n = run_sim(scn, "nearest", P, 0)
        a = [run_sim(scn, "aco", P, s) for s in range(3)]
        d = sum(x["distance"] for x in a) / 3
        self.title("Multi-Agent Delivery - Supervisor Dashboard")
        messagebox.showinfo("Nearest vs ACO", f"{SCENARIO_NAMES[self.scn_box.current()]}\n\nNearest: distance {n['distance']:.0f}, late {n['late']}\n"
                            f"ACO (avg of 3): distance {d:.0f}, late {sum(x['late'] for x in a) / 3:.1f}\nImprovement: {100 * (1 - d / n['distance']):.1f}%")

    def loop(self):
        if not self.running: return
        self.tick()
        if self.running: self.after(int(1000 / self.speed.get()), self.loop)

    def tick(self):
        self.sim.step()
        if self.sim.done: self.running = False
        self.draw()

    def draw(self):
        draw_sim(self.cv, self.sim, self.size)
        self.kpi.set(kpi_text(self.sim)); self.update_feed(); self.draw_conv()

    def update_feed(self):
        s = self.sim
        for m in s.bus.log[self.feed_pos:]:
            if m.type in SHOW:
                p = m.payload
                txt = {"NEW_ORDER": f"order #{p.get('order', {}).get('id')} placed", "TRAFFIC_UPDATE": "traffic report",
                       "PLAN_REQUEST": f"asks Dispatcher to plan ({p.get('reason')})", "RECEIPT_CONFIRM": f"order #{p.get('id')} receipt confirmed",
                       "BREAKDOWN": f"breakdown, {len(p.get('orders', []))} orders stranded", "CANCEL_ORDER": f"cancels order #{p.get('id')}"}[m.type]
                self.feed.insert("end", f"[{m.time:4.0f}] {m.sender} > {m.receiver}: {txt}\n")
        self.feed_pos = len(s.bus.log)
        for t, text in s.supervisor.decisions[self.dec_pos:]:
            self.feed.insert("end", f"[{t:4.0f}] SUPERVISOR: {text}\n", "sup")
        self.dec_pos = len(s.supervisor.decisions); self.feed.see("end")

    def draw_conv(self):
        c, last = self.conv, self.sim.dispatcher.last
        c.delete("all")
        if not last or len(last["history"]) < 2: return
        h, base = last["history"], last["baseline_cost"]
        lo, hi = min(min(h), base) * 0.95, max(max(h), base) * 1.05
        X = lambda i: 10 + i * 420 / (len(h) - 1); Y = lambda v: 110 - (v - lo) / (hi - lo + 1e-9) * 95
        c.create_line(10, Y(base), 430, Y(base), fill="#6CB6FF", dash=(4, 3)); c.create_text(430, Y(base) - 8, text="nearest", fill="#6CB6FF", anchor="e", font=("Segoe UI", 8))
        c.create_line(*[v for i, y in enumerate(h) for v in (X(i), Y(y))], fill=GOLD, width=2)


class CompareWindow(tk.Toplevel):
    """Two simulations of the same scenario side by side; events are applied to both."""
    SIZE = 470

    def __init__(self, master, scn, speed):
        super().__init__(master)
        self.title("Live compare"); self.configure(bg=BG)
        self.scn, self.running, self.sims = scn, False, []
        top = tk.Frame(self, bg=BG); top.grid(row=0, column=0, columnspan=2, pady=8)
        for t, f in (("Start / Restart", self.restart), ("Pause / Resume", self.toggle), ("+ Random order", self.order), ("Traffic jam", self.jam),
                     ("Breakdown", self.breakdown), ("Cancel order", self.cancel)):
            tk.Button(top, text=t, command=f, bg=GOLD if t.startswith("Start") else "#1D2D3E", fg="#1A1204" if t.startswith("Start") else FG, relief="flat", padx=8, pady=5).pack(side="left", padx=3)
        self.speed = tk.Scale(top, from_=1, to=30, orient="horizontal", bg=BG, fg=FG, highlightthickness=0, troughcolor=CARD, length=120); self.speed.set(speed); self.speed.pack(side="left", padx=8)
        self.boxes, self.cvs, self.vars = [], [], []
        for i in range(2):
            f = tk.Frame(self, bg=BG); f.grid(row=1, column=i, padx=8, pady=4)
            box = ttk.Combobox(f, values=SOLVER_LABELS, state="readonly", width=22); box.current(i); box.pack(pady=4)
            cv = tk.Canvas(f, width=self.SIZE, height=self.SIZE, bg="#101B27", highlightthickness=0); cv.pack()
            v = tk.StringVar(); tk.Label(f, textvariable=v, bg=CARD, fg=FG, font=("Consolas", 10), justify="left", anchor="w", padx=8, pady=6, width=62).pack(fill="x", pady=4)
            self.boxes.append(box); self.cvs.append(cv); self.vars.append(v)
        self.restart()

    def restart(self):
        self.sims = [Simulation(self.scn, SOLVER_KEYS[b.current()], seed=7) for b in self.boxes]
        self.running = True; self.draw(); self.loop()

    def toggle(self):
        self.running = not self.running
        if self.running: self.loop()

    def each(self, fn):
        for s in self.sims: fn(s)
        if not self.running: self.running = True; self.loop()

    def order(self):
        x, y = random.uniform(5, 95), random.uniform(5, 95); self.each(lambda s: s.add_order(x, y, 40))

    def jam(self):
        x, y = random.uniform(10, 60), random.uniform(10, 60); self.each(lambda s: s.add_traffic(x, y, x + 35, y + 35, 2.5, 40))

    def breakdown(self):
        ids = [c.spec.id for c in self.sims[0].couriers if not c.broken]
        if len(ids) > 1:
            t = random.choice(ids); self.each(lambda s: s.trigger("breakdown", t))

    def cancel(self):
        ids = [i for i, c in self.sims[0].customers.items() if c.state not in ("delivered", "cancelled")]
        if ids:
            t = random.choice(ids); self.each(lambda s: s.trigger("cancel", t))

    def draw(self):
        for s, cv, v in zip(self.sims, self.cvs, self.vars):
            draw_sim(cv, s, self.SIZE); v.set(kpi_text(s))

    def loop(self):
        if not self.running or not self.winfo_exists(): return
        for s in self.sims:
            if not s.done: s.step()
        self.draw()
        if all(s.done for s in self.sims): self.running = False; return
        self.after(int(1000 / self.speed.get()), self.loop)


if __name__ == "__main__":
    App().mainloop()
