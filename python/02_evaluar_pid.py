"""
Sesión 1 · Paso 3: controlador PID de referencia (la "maestra" que luego imita la red).

Evalúa el PID en escalón, senoidal, triangular y aleatoria limitada:
error de posición (RMSE, máximo), sobreimpulso, tiempo de establecimiento, esfuerzo de control.
--tune hace una búsqueda en rejilla de kp y kd (ki = 0.2*kp) y guarda data/pid_gains.json.

Uso:   python 02_evaluar_pid.py [--mock] [--tune]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from controllers import PID
from coppelia_robot import make_robot
from nn_data import save_json
from scenarios import nominal_scenarios
from sim_loop import run_episode, step_metrics, tracking_metrics


def evaluate(robot, gains=None, seed=5000, verbose=False):
    pid = PID(robot.dt, robot.u_max, robot.n, gains)
    rows, eps = [], {}
    for sc in nominal_scenarios(robot, seed):
        ep = run_episode(robot, pid, sc["traj"], sc["T"], sc["q0"], seed=sc["seed"])
        m = tracking_metrics(ep, robot.dt)
        if sc["t_step"] is not None:
            m.update(step_metrics(ep, robot.dt, sc["t_step"]))
        m["escenario"] = sc["name"]
        rows.append(m)
        eps[sc["name"]] = ep
    return rows, eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--tune", action="store_true")
    a = ap.parse_args()
    robot = make_robot(a.mock)

    if a.tune:
        best = None
        for kp in (2.0, 4.0, 6.0, 8.0, 12.0):
            for kd in (0.0, 0.15, 0.4):
                g = dict(kp=kp, ki=0.2 * kp, kd=kd)
                rows, _ = evaluate(robot, g)
                score = np.mean([r["rmse_ss"] for r in rows if r["escenario"] != "step_single"])
                print(f"kp={kp:4.1f} kd={kd:4.2f} ki={g['ki']:.2f} -> RMSE (régimen) medio {score:.4f} rad")
                if best is None or score < best[0]:
                    best = (score, g)
        cfg_g = dict(cfg.PID_DEFAULT)
        cfg_g.update(best[1])
        save_json(cfg.DATA / "pid_gains.json", cfg_g)
        print(f"\nMejor: {best[1]}  (guardado en data/pid_gains.json)\n")

    rows, eps = evaluate(robot)
    print(f"{'escenario':12s} {'RMSE[rad]':>10s} {'RMSE_ss':>9s} {'emax[rad]':>10s} {'u_rms':>8s} {'OS[%]':>7s} {'ts[s]':>7s}")
    for r in rows:
        print(f"{r['escenario']:12s} {r['rmse']:10.4f} {r['rmse_ss']:9.4f} {r['e_max']:10.4f} {r['u_rms']:8.3f} "
              f"{r.get('overshoot_pct', np.nan):7.1f} {r.get('settle_s', np.nan):7.2f}")
    save_json(cfg.DATA / "pid_metrics.json", rows)

    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    for k, nm in enumerate(("step_single", "sine")):
        ep = eps[nm]
        ax[0, k].plot(ep["t"], ep["qr"][:, 0], "--", label="qr J1")
        ax[0, k].plot(ep["t"], ep["q"][:, 0], label="q J1")
        ax[0, k].set_title(f"PID · {nm}"); ax[0, k].legend(); ax[0, k].grid(alpha=.3)
        ax[1, k].plot(ep["t"], ep["u_app"][:, 0]); ax[1, k].set_ylabel("u J1 [rad/s]"); ax[1, k].grid(alpha=.3)
        ax[1, k].set_xlabel("t [s]")
    plt.tight_layout()
    plt.savefig(cfg.FIGS / "02_pid.png", dpi=130)
    robot.close()


if __name__ == "__main__":
    main()
