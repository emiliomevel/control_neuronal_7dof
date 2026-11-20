"""
Sesión 1 · Paso 2: caracterizar el manipulador (lee parámetros de Coppelia y mide lo que no se puede leer).

Obtiene por articulación:  límites q_min/q_max, velocidad máxima, aceleración máxima,
relación comando->movimiento (ganancia y constante de tiempo, análoga a PWM->velocidad),
frecuencia de muestreo, masa del último eslabón.  Escribe data/robot_params.json,
que luego usan PID, trayectorias, dataset y firmware (u_max, límites).

Uso:   python 01_caracterizar_robot.py            # CoppeliaSim abierto con la escena
       python 01_caracterizar_robot.py --mock     # sin simulador (depuración)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from coppelia_robot import make_robot
from nn_data import save_json


def run_joint(robot, j, u, T, q0):
    """Comanda velocidad constante u solo a la articulación j (hacia el lado con más espacio)."""
    sign = 1.0 if (robot.q_max[j] - q0[j]) >= (q0[j] - robot.q_min[j]) else -1.0
    room = (robot.q_max[j] - q0[j]) if sign > 0 else (q0[j] - robot.q_min[j])
    steps = max(int(min(T, 0.7 * room / max(abs(u), 1e-6)) / robot.dt), 10)
    robot.reset(q0)
    cmd = np.zeros(robot.n)
    cmd[j] = sign * u
    dqs = []
    for _ in range(steps):
        robot.apply(cmd)
        robot.step()
        _, dq = robot.read()
        dqs.append(sign * dq[j])
    return np.array(dqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()

    robot = make_robot(a.mock)
    n, dt = robot.n, robot.dt
    q_mid = 0.5 * (robot.q_min + robot.q_max)
    print(f"Robot: {robot.info.get('kind')}  n={n}  dt={dt}")
    for j, nm in enumerate(robot.names):
        print(f"  J{j + 1}: {nm}   q∈[{robot.q_min[j]:+.3f}, {robot.q_max[j]:+.3f}] rad")

    res, fig_resp = [], []
    for j in range(n):
        # 1) respuesta al escalón de comando (ganancia estática y constante de tiempo)
        gains, taus = [], []
        for u in (0.25, 0.5, 1.0):
            dq = run_joint(robot, j, u, 1.5, q_mid)
            ss = float(np.mean(dq[-max(len(dq) // 5, 3):]))
            gains.append(ss / u)
            idx = np.where(dq >= 0.632 * ss)[0]
            taus.append(float(idx[0] * dt) if len(idx) else np.nan)
            if u == 0.5:
                fig_resp.append(dq)
        # 2) velocidad y aceleración máximas (comando muy grande)
        dq = run_joint(robot, j, 3.0, 0.8, q_mid)
        sm = np.convolve(dq, np.ones(3) / 3, mode="valid")
        vmax = float(np.max(np.abs(sm)))
        amax = float(np.max(np.abs(np.diff(dq[:30]) / dt)))
        res.append(dict(joint=j + 1, name=robot.names[j], q_min=float(robot.q_min[j]), q_max=float(robot.q_max[j]),
                        gain=float(np.mean(gains)), tau_s=float(np.nanmean(taus)), v_max=vmax, a_max=amax))
        print(f"  J{j + 1}: ganancia={res[-1]['gain']:.3f}  tau={res[-1]['tau_s'] * 1000:.0f} ms  "
              f"vmax={vmax:.2f} rad/s  amax={amax:.1f} rad/s²")

    u_max = float(min(cfg.U_MAX_CFG, 0.9 * min(r["v_max"] for r in res)))
    mass = robot.mass_info()
    old = cfg.load_robot_params() or {}
    keep = {k: old[k] for k in ("q_min_safe", "q_max_safe", "range_frac", "collision_report") if k in old}
    out = dict(robot=robot.info.get("kind"), n_joints=n, dt=dt, fs_hz=1.0 / dt,
               q_min=robot.q_min_hw, q_max=robot.q_max_hw, u_max=u_max, **keep, joints=res,
               encoder_resolution="simulado (float64): sin cuantización", payload_max_kg=cfg.PAYLOAD_MAX_KG, **mass)
    save_json(cfg.DATA / "robot_params.json", out)
    print(f"\n|u|max adoptado = {u_max:.3f} rad/s   (mín(U_MAX_CFG, 0.9*v_max medido))")
    print(f"Escrito {cfg.DATA / 'robot_params.json'}")

    t = np.arange(len(fig_resp[0])) * dt
    plt.figure(figsize=(7, 4))
    for j, dq in enumerate(fig_resp):
        plt.plot(np.arange(len(dq)) * dt, dq, label=f"J{j + 1}")
    plt.axhline(0.5, ls=":", c="k")
    plt.xlabel("t [s]"); plt.ylabel("velocidad [rad/s]"); plt.title("Respuesta al comando u=0.5 rad/s")
    plt.legend(ncol=4, fontsize=7); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(cfg.FIGS / "01_respuesta_escalon.png", dpi=130)
    robot.close()


if __name__ == "__main__":
    main()
