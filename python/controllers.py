"""
Controladores. Todos tienen la misma firma:  u = ctrl(q, dq, qr, dqr)   (arrays de n articulaciones)
y un método reset(). La capa de seguridad `safety` se aplica SIEMPRE después (ver sim_loop.py),
sea quien sea el controlador: la red no tiene autoridad ilimitada sobre el actuador.
"""
import time

import numpy as np

import config as cfg
from nn_data import build_features


def safety(q, u, u_max, q_min, q_max, margin=cfg.JOINT_MARGIN):
    """Saturación |u|<=u_max, descarte de NaN/inf y bloqueo de movimiento hacia fuera de los límites articulares.
    (Misma lógica que nn_safety() en el firmware.)"""
    u = np.nan_to_num(np.asarray(u, float), nan=0.0, posinf=0.0, neginf=0.0)
    u = np.clip(u, -u_max, u_max)
    u = np.where((q <= q_min + margin) & (u < 0), 0.0, u)
    u = np.where((q >= q_max - margin) & (u > 0), 0.0, u)
    return u


class PID:
    """PID de posición articular que entrega velocidad comandada.
        e = qr - q ;  de = dqr - dq ;  u = kp*e + ki*∫e + kd*de      (saturado a ±u_max)
    La integral tiene anti-windup (integración condicional + tope de |ki*I|)."""
    name = "PID"

    def __init__(self, dt, u_max, n=cfg.N_JOINTS, gains=None):
        g = dict(cfg.load_pid())
        if gains:
            g.update(gains)
        self.dt = dt
        self.u_max = np.broadcast_to(np.asarray(u_max, float), (n,)).copy()
        self.kp, self.ki, self.kd, self.i_lim = (np.broadcast_to(np.asarray(g[k], float), (n,)).copy()
                                                 for k in ("kp", "ki", "kd", "i_lim"))
        self.reset()
        self.last_unsat = np.zeros(n)

    def reset(self):
        self.I = np.zeros_like(self.kp)

    def __call__(self, q, dq, qr, dqr):
        e = qr - q
        de = dqr - dq
        I_new = self.I + e * self.dt
        I_new = np.clip(I_new, -self.i_lim / np.maximum(self.ki, 1e-12), self.i_lim / np.maximum(self.ki, 1e-12))
        u_unsat = self.kp * e + self.ki * I_new + self.kd * de
        u = np.clip(u_unsat, -self.u_max, self.u_max)
        saturado_y_empuja = (np.abs(u_unsat) > self.u_max) & (np.sign(e) == np.sign(u_unsat))
        self.I = np.where(saturado_y_empuja, self.I, I_new)
        self.last_unsat = u_unsat
        return u


class NNController:
    """Red neuronal (NumPy) que sustituye al PID (objetivo 'directo')."""
    name = "NN"

    def __init__(self, mlp, norm):
        self.mlp, self.norm = mlp, norm
        self.times_us = []

    def reset(self):
        pass

    def __call__(self, q, dq, qr, dqr):
        t0 = time.perf_counter()
        X = self.norm.x(build_features(q, dq, qr, dqr, self.norm.mode))
        y = np.clip(self.mlp.forward(X).reshape(-1), -1.0, 1.0) * self.norm.y_scale
        self.times_us.append((time.perf_counter() - t0) * 1e6)
        return y.astype(float)


class ResidualController:
    """u = clip(u_PID + u_NN): la red solo aporta una corrección acotada a ±y_scale."""
    name = "PID+NN"

    def __init__(self, pid, nn):
        self.pid, self.nn = pid, nn
        self.times_us = nn.times_us

    def reset(self):
        self.pid.reset()
        self.nn.reset()

    def __call__(self, q, dq, qr, dqr):
        u = self.pid(q, dq, qr, dqr) + self.nn(q, dq, qr, dqr)
        return np.clip(u, -self.pid.u_max, self.pid.u_max)


class MicroController:
    """Procesador-en-el-lazo: el estado se manda al micro por UART y éste devuelve u_NN (inferencia en el firmware)."""
    name = "NN-micro"

    def __init__(self, micro, pid=None):
        self.micro, self.pid = micro, pid
        self.cycles = []
        if pid is not None:
            self.name = "PID+NN-micro"

    def reset(self):
        if self.pid is not None:
            self.pid.reset()

    def __call__(self, q, dq, qr, dqr):
        u, cyc = self.micro.infer(q, dq, qr, dqr)
        self.cycles.append(cyc)
        if self.pid is not None:
            u = np.clip(self.pid(q, dq, qr, dqr) + u, -self.pid.u_max, self.pid.u_max)
        return u
