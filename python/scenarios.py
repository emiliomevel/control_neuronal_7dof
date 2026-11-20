"""Escenarios de evaluación reproducibles (mismas condiciones para PID, NN, residual y micro)."""
import numpy as np

import config as cfg
from sim_loop import Perturbation
from trajectories import find_free_pose, make_step_single, sample_free


def nominal_scenarios(robot, seed=5000, T=8.0):
    rng = np.random.default_rng(seed)
    v_ref = 0.6 * float(robot.u_max[0])
    out = []
    # escalón único (para sobreimpulso y tiempo de establecimiento) desde la pose central
    q_mid = 0.5 * (robot.q_min + robot.q_max)
    amp = 0.4 * np.where(np.arange(robot.n) % 2 == 0, 1.0, -1.0)
    q_mid, amp, ok = find_free_pose(robot, rng, q_mid, amp)
    if not ok:
        print("AVISO: no se encontró un escalón libre de colisión; corre 01b_detectar_colisiones.py")
    out.append(dict(name="step_single", traj=make_step_single(q_mid, amp, 0.5), T=6.0, q0=q_mid, t_step=0.5, seed=seed))
    for i, kind in enumerate(cfg.KINDS):
        q0, traj, ok, _ = sample_free(robot, kind, rng, v_ref, T)
        if not ok:
            print(f"AVISO: trayectoria '{kind}' sin garantía de libre de colisión (corre 01b_detectar_colisiones.py)")
        out.append(dict(name=kind, traj=traj, T=T, q0=q0, t_step=None, seed=seed + i))
    return out


def perturbations(robot, seed=7000):
    rng = np.random.default_rng(seed)
    n = robot.n
    return [
        Perturbation(label="nominal"),
        Perturbation(payload_kg=cfg.PAYLOAD_MAX_KG, label="carga +%.1f kg" % cfg.PAYLOAD_MAX_KG),
        Perturbation(motor_gain=0.75, label="motor -25% ganancia"),
        Perturbation(pulse_t0=2.0, pulse_dur=0.4, pulse_amp=rng.choice([-1.0, 1.0], n) * 0.5, label="empujon 0.5 rad/s"),
        Perturbation(noise_q=0.002, noise_dq=0.02, label="ruido de sensor"),
    ]
