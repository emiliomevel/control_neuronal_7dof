"""
Sesión 1 · Paso 4: generación del dataset operando el robot con el PID.

Cada episodio: pose inicial aleatoria, trayectoria aleatoria (escalón/senoidal/triangular/aleatoria
limitada), a veces carga extra, variación de ganancia del motor, empujones y ruido pequeño.
Se registra cada DECIM pasos:  q, dq, qr, dqr, e, de  ->  u_PID  (y u_teacher para el objetivo residual).

Uso:   python 03_generar_dataset.py [--mock] [--episodes 80] [--seed 1]
Salida: data/dataset_raw.npz  y  data/dataset_raw.csv
"""
import argparse
import time

import numpy as np
import pandas as pd

import config as cfg
from controllers import PID
from coppelia_robot import make_robot
from sim_loop import Perturbation, run_episode
from trajectories import sample_free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--episodes", type=int, default=cfg.N_EPISODES)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    a = ap.parse_args()

    robot = make_robot(a.mock)
    n = robot.n
    pid = PID(robot.dt, robot.u_max, n)
    rng = np.random.default_rng(a.seed)
    v_ref = 0.6 * float(robot.u_max[0])
    cols = {k: [] for k in ("q", "dq", "qr", "dqr", "u", "u_teacher")}
    meta = {k: [] for k in ("ep", "kind", "t", "payload", "gain")}
    t_start = time.time()
    skipped = 0

    for ep_i in range(a.episodes):
        kind_i = ep_i % len(cfg.KINDS)                       # reparto uniforme de tipos
        kind = cfg.KINDS[kind_i]
        q0, traj, free, tries = sample_free(robot, kind, rng, v_ref, cfg.EP_T, tries=200)
        if not free:                                         # no se registra nada que pueda chocar
            skipped += 1
            print(f"  episodio {ep_i + 1}: sin trayectoria libre de colisión tras 200 intentos ({kind}) -> omitido")
            continue
        pert = Perturbation(
            payload_kg=float(rng.uniform(0, cfg.PAYLOAD_MAX_KG)) if rng.random() < 0.5 else 0.0,
            motor_gain=float(rng.uniform(0.9, 1.1)) if rng.random() < 0.5 else 1.0,
            noise_q=0.001 if rng.random() < 0.5 else 0.0, noise_dq=0.01 if rng.random() < 0.5 else 0.0)
        if rng.random() < 0.3:                               # empujón pequeño
            pert.pulse_t0 = float(rng.uniform(1.0, cfg.EP_T - 1.5))
            pert.pulse_amp = rng.uniform(-0.3, 0.3, n)
        ep = run_episode(robot, pid, traj, cfg.EP_T, q0, pert, seed=int(rng.integers(1 << 30)))
        idx = np.arange(0, len(ep["t"]), cfg.DECIM)
        # el controlador ve la medición (qm, dqm) -> eso es lo que se registra como estado
        u_pid = ep["u_cmd"][idx]
        u_teacher = np.clip(u_pid + cfg.KFF * ep["dqr"][idx], -robot.u_max, robot.u_max)
        for key, src in (("q", "qm"), ("dq", "dqm"), ("qr", "qr"), ("dqr", "dqr")):
            cols[key].append(ep[src][idx])
        cols["u"].append(u_pid)
        cols["u_teacher"].append(u_teacher)
        meta["ep"].append(np.full(len(idx), ep_i))
        meta["kind"].append(np.full(len(idx), kind_i))
        meta["t"].append(ep["t"][idx])
        meta["payload"].append(np.full(len(idx), pert.payload_kg))
        meta["gain"].append(np.full(len(idx), pert.motor_gain))
        if (ep_i + 1) % 5 == 0 or ep_i == a.episodes - 1:
            el = time.time() - t_start
            print(f"  episodio {ep_i + 1}/{a.episodes}  ({el:.0f} s, ETA {el / (ep_i + 1) * (a.episodes - ep_i - 1):.0f} s)")

    if skipped:
        print(f"\nAVISO: {skipped} episodios omitidos por colisión; corre 01b_detectar_colisiones.py para ampliar el espacio libre")
    D = {k: np.concatenate(v) for k, v in {**cols, **meta}.items()}
    D.update(u_max=float(robot.u_max[0]), q_min=robot.q_min, q_max=robot.q_max, dt=robot.dt * cfg.DECIM,
             kinds=np.array(cfg.KINDS))
    np.savez(cfg.DATA / "dataset_raw.npz", **D)

    # CSV legible (una fila por instante, columnas por articulación)
    df = pd.DataFrame({"ep": D["ep"], "kind": [cfg.KINDS[i] for i in D["kind"]], "t": D["t"]})
    for name in ("q", "dq", "qr", "dqr", "u"):
        for j in range(n):
            df[f"{name}{j + 1}"] = D[name][:, j]
    for j in range(n):
        df[f"e{j + 1}"] = D["qr"][:, j] - D["q"][:, j]
        df[f"de{j + 1}"] = D["dqr"][:, j] - D["dq"][:, j]
    df["payload_kg"], df["motor_gain"] = D["payload"], D["gain"]
    df.to_csv(cfg.DATA / "dataset_raw.csv", index=False, float_format="%.6g")
    T = len(D["t"])
    print(f"\nDataset: {T} instantes × {n} articulaciones = {T * n} muestras articulares "
          f"({a.episodes} episodios).  Guardado en data/dataset_raw.npz / .csv")
    robot.close()


if __name__ == "__main__":
    main()
