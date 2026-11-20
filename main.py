"""python main.py gui | demo | web | experiments"""
import sys
cmd = sys.argv[1] if len(sys.argv) > 1 else "web"

if cmd == "gui":
    from evaluation.gui import App; App().mainloop()
elif cmd == "demo":
    from core.scenario import generate_all, load_scenario
    from core.simulation import Simulation
    generate_all("data")
    for solver in ("nearest", "aco"):
        print(solver, Simulation(load_scenario("data/scenario_5"), solver).run())
elif cmd == "web":
    import runpy
    runpy.run_path("web_ui/web_server.py", run_name="__main__")
else:
    sys.argv = ["evaluation/experiments.py"] + sys.argv[2:]
    import runpy; runpy.run_path("evaluation/experiments.py", run_name="__main__")