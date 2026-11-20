"""
Sesión 2/3 · Pasos 9-10: simulación y comparación PID vs controlador neuronal (todo simulado en Coppelia).

Mismas condiciones para todos los controladores (misma pose inicial, trayectoria, ruido, perturbación).
  A) Nominal: RMSE, error máximo, sobreimpulso, tiempo de establecimiento, esfuerzo de control.
  B) Perturbaciones: carga extra, motor -25 %, empujón, ruido -> degradación = RMSE_perturbado / RMSE_nominal.
  C) Generalización: trayectorias/poses/velocidades/cargas con semillas NUNCA usadas en el dataset.
  D) Costo embebido (si hay micro): tiempo de inferencia, RAM, Flash, CPU, y fidelidad micro vs NumPy.

Controladores:  PID  ·  NN (reemplaza al PID, objetivo directo)  ·  PID+NN (residual)  ·  *-micro (inferencia en el S32K).

Uso:   python 06_comparar_pid_vs_nn.py --objetivo directo [--mock] [--modelo pc|micro] [--micro COM5@921600 [--push-weights]]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
from controllers import PID, MicroController, NNController, ResidualController, safety
from coppelia_robot import make_robot
from micro_link import Micro
from nn_data import Norm, build_features, save_json
from nn_numpy import MLP
from scenarios import nominal_scenarios, perturbations
from sim_loop import Perturbation, run_episode, step_metrics, tracking_metrics
from trajectories import make_trajectory, random_pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--objetivo", choices=["directo", "residual"], default="directo")
    ap.add_argument("--modelo", choices=["pc", "micro"], default="pc", help="de qué entrenamiento salen los pesos")
    ap.add_argument("--micro", default=None, help="URL del micro para procesador-en-el-lazo (PIL)")
    ap.add_argument("--push-weights", action="store_true", help="copia los pesos del modelo al micro (firmware con entrenamiento)")
    ap.add_argument("--n-gen", type=int, default=8, help="episodios de generalización por tipo")
    a = ap.parse_args()
    residual = a.objetivo == "residual"

    robot = make_robot(a.mock)
    n = robot.n
    norm = Norm.load(np.load(cfg.DATA / f"norm_{a.objetivo}.npz"))
    model = np.load(cfg.DATA / f"model_{a.modelo}_{a.objetivo}.npz")
    mlp = MLP(int(model["n_in"]), int(model["h1"]), int(model["h2"]), int(model["n_out"]))
    mlp.set_params(model["params"])

    pid = PID(robot.dt, robot.u_max, n)
    nn = NNController(mlp, norm)
    ctrls = {"PID": pid}
    if residual:
        ctrls["PID+NN"] = ResidualController(PID(robot.dt, robot.u_max, n), nn)
    else:
        ctrls["NN"] = nn

    mic, info = None, None
    if a.micro:
        mic = Micro(a.micro)
        mic.ping()
        info = mic.info()
        if a.push_weights:
            mic.set_weights(model["params"])
        elif not np.array_equal(mic.get_weights(), model["params"].astype(np.float32)):
            print("AVISO: los pesos del micro difieren de los del modelo usado en PC "
                  "(usa --push-weights, o --modelo micro si el modelo viene de 07_entrenar_en_micro.py).")
        name = "PID+NN-micro" if residual else "NN-micro"
        ctrls[name] = MicroController(mic, PID(robot.dt, robot.u_max, n) if residual else None)

    rows = []

    def run(label_group, sc, pert, **extra):
        for cname, c in ctrls.items():
            ep = run_episode(robot, c, sc["traj"], sc["T"], sc["q0"], pert, seed=sc["seed"])
            m = tracking_metrics(ep, robot.dt)
            if sc.get("t_step") is not None:
                m.update(step_metrics(ep, robot.dt, sc["t_step"]))
            rows.append(dict(grupo=label_group, escenario=sc["name"], perturbacion=pert.label, controlador=cname, **m, **extra))
            yield cname, ep

    # ---------------- A) nominal
    eps_nom = {}
    for sc in nominal_scenarios(robot):
        for cname, ep in run("A_nominal", sc, Perturbation()):
            eps_nom[(sc["name"], cname)] = ep

    # ---------------- B) perturbaciones (sobre senoidal y aleatoria)
    base = {s["name"]: s for s in nominal_scenarios(robot)}
    for sname in ("sine", "random"):
        for pert in perturbations(robot):
            for _ in run("B_perturb", base[sname], pert):
                pass

    # ---------------- C) generalización: semillas nuevas, más velocidad, cargas distintas
    rng = np.random.default_rng(9000)
    for kind in cfg.KINDS:
        for k in range(a.n_gen):
            q0 = random_pose(rng, robot.q_min, robot.q_max)
            traj = make_trajectory(kind, rng, robot.q_min, robot.q_max, 0.85 * float(robot.u_max[0]), cfg.EP_T, q0)
            pert = Perturbation(payload_kg=float(rng.uniform(0, cfg.PAYLOAD_MAX_KG)), label="gen")
            sc = dict(name=kind, traj=traj, T=cfg.EP_T, q0=q0, t_step=None, seed=9100 + k)
            for _ in run("C_generaliza", sc, pert):
                pass

    df = pd.DataFrame(rows)
    df.to_csv(cfg.DATA / f"comparacion_{a.objetivo}_{a.modelo}.csv", index=False)
    pd.set_option("display.width", 200)

    print("\n=== A) NOMINAL ===")
    A = df[df.grupo == "A_nominal"]
    print(A[["escenario", "controlador", "rmse", "rmse_ss", "e_max", "u_rms", "u_rough", "overshoot_pct", "settle_s"]]
          .round(4).to_string(index=False))

    print("\n=== B) PERTURBACIONES (RMSE [rad] y degradación respecto al nominal) ===")
    B = df[df.grupo == "B_perturb"].groupby(["perturbacion", "controlador"], sort=False).rmse.mean().unstack()
    nom = B.loc["nominal"]
    out = B.copy()
    for c in B.columns:
        out[c + " (x nominal)"] = B[c] / nom[c]
    print(out.round(3).to_string())

    print("\n=== C) GENERALIZACIÓN (trayectorias no vistas, velocidades mayores, cargas) ===")
    C = df[df.grupo == "C_generaliza"].groupby(["escenario", "controlador"], sort=False)[["rmse", "rmse_ss", "e_max", "u_rms"]].mean()
    print(C.unstack().round(4).to_string())

    # ---------------- D) costo embebido
    print("\n=== COMPARACIÓN PID vs NN (tabla de la práctica) ===")
    tab = {}
    for cname in ctrls:
        s = df[(df.grupo != "B_perturb") & (df.controlador == cname) & (df.escenario != "step_single")]
        st = df[(df.controlador == cname) & (df.escenario == "step_single")].iloc[0]
        tab[cname] = {
            "RMSE [rad]": s.rmse.mean(), "RMSE régimen [rad]": s.rmse_ss.mean(), "Error máx [rad]": s.e_max.mean(),
            "Sobreimpulso [%]": st.overshoot_pct, "T. establecimiento [s]": st.settle_s,
            "Esfuerzo u_rms [rad/s]": s.u_rms.mean(), "Suavidad du/dt [rad/s²]": s.u_rough.mean()}
    for cname, c in ctrls.items():
        if hasattr(c, "times_us") and len(c.times_us):
            tab[cname]["T. inferencia PC [µs]"] = float(np.mean(c.times_us))
        if isinstance(c, MicroController) and c.cycles:
            us = 1e6 * np.mean(c.cycles) / info["core_hz"]
            tab[cname].update({"T. inferencia micro [µs]": us, "CPU @ lazo de control [%]": 100 * us / (robot.dt * 1e6),
                               "RAM red [B]": info["ram_bytes"], "Flash constantes [B]": info["flash_bytes"]})
    T = pd.DataFrame(tab)
    print(T.round(4).to_string())
    T.to_csv(cfg.DATA / f"tabla_comparacion_{a.objetivo}_{a.modelo}.csv")
    if mic is not None and info["core_hz"] == 120000000 and a.micro.startswith("exec:"):
        print("(los tiempos del micro virtual son tiempos de la PC, NO del S32K)")

    # ---------------- fidelidad micro vs NumPy
    if mic is not None:
        ep = eps_nom[("sine", "PID+NN" if residual else "NN")]
        qm, dqm, qr, dqr = ep["qm"][:300], ep["dqm"][:300], ep["qr"][:300], ep["dqr"][:300]
        worst = 0.0
        for k in range(len(qm)):
            u_np = nn(qm[k], dqm[k], qr[k], dqr[k])
            if not residual:
                u_np = safety(qm[k], u_np, robot.u_max, robot.q_min, robot.q_max)
            u_mc, _ = mic.infer(qm[k], dqm[k], qr[k], dqr[k])
            worst = max(worst, float(np.max(np.abs(u_np - u_mc))))
        print(f"\nFidelidad micro vs NumPy (300 pasos, mismos pesos): |u_micro - u_numpy|max = {worst:.2e} rad/s  "
              f"{'OK' if worst < 1e-3 else 'REVISAR (¿mismos pesos / nn_config.h?)'}")
        mic.close()

    # ---------------- figuras
    fig, ax = plt.subplots(2, 3, figsize=(15, 6.5))
    for k, sname in enumerate(("step_single", "sine", "random")):
        for cname in ctrls:
            ep = eps_nom[(sname, cname)]
            ax[0, k].plot(ep["t"], ep["q"][:, 0], label=cname)
            ax[1, k].plot(ep["t"], ep["u_app"][:, 0], label=cname)
        ax[0, k].plot(ep["t"], ep["qr"][:, 0], "k--", lw=.8, label="ref")
        ax[0, k].set_title(f"{sname} · J1"); ax[0, k].set_ylabel("q [rad]"); ax[0, k].grid(alpha=.3)
        ax[1, k].set_ylabel("u [rad/s]"); ax[1, k].set_xlabel("t [s]"); ax[1, k].grid(alpha=.3)
    ax[0, 0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(cfg.FIGS / f"06_comparacion_{a.objetivo}_{a.modelo}.png", dpi=130)

    plt.figure(figsize=(8, 4))
    w = 0.8 / len(B.columns)
    for i, c in enumerate(B.columns):
        plt.bar(np.arange(len(B)) + i * w, B[c], w, label=c)
    plt.xticks(np.arange(len(B)) + 0.4 - w / 2, B.index, rotation=20, fontsize=8)
    plt.ylabel("RMSE [rad]"); plt.title("Robustez ante perturbaciones"); plt.legend(); plt.grid(alpha=.3, axis="y")
    plt.tight_layout()
    plt.savefig(cfg.FIGS / f"06_perturbaciones_{a.objetivo}_{a.modelo}.png", dpi=130)
    save_json(cfg.DATA / f"resumen_{a.objetivo}_{a.modelo}.json", T.to_dict())
    robot.close()


if __name__ == "__main__":
    main()
