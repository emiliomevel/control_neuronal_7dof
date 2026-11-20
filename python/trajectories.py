"""
Generadores de trayectorias articulares (qr(t), dqr(t)) para 7 articulaciones.
Tipos: escalón, senoidal, triangular y aleatoria limitada. Todas respetan los límites
articulares (con margen) y una velocidad máxima de referencia.
"""
import numpy as np


class Trajectory:
    kind = "base"

    def eval(self, t):
        raise NotImplementedError


class StepTraj(Trajectory):
    """Escalones: lista de (t_i, nivel_i). Antes del primer escalón se mantiene q_start."""
    kind = "step"

    def __init__(self, q_start, steps):
        self.q_start = np.asarray(q_start, float)
        self.steps = [(float(t), np.asarray(l, float)) for t, l in steps]

    def eval(self, t):
        q = self.q_start
        for ti, li in self.steps:
            if t >= ti:
                q = li
        return q.copy(), np.zeros_like(q)


class SumSineTraj(Trajectory):
    """qr = c + sum_k A_k sin(w_k t + phi_k).  K=1 -> senoidal; K=3 -> aleatoria suave."""

    def __init__(self, c, A, w, phi, kind="sine"):
        self.c, self.A, self.w, self.phi, self.kind = c, A, w, phi, kind

    def eval(self, t):
        arg = self.w * t + self.phi
        return self.c + (self.A * np.sin(arg)).sum(0), (self.A * self.w * np.cos(arg)).sum(0)


class TriangleTraj(Trajectory):
    kind = "triangle"

    def __init__(self, c, A, period, phase):
        self.c, self.A, self.P, self.ph = c, A, period, phase

    def eval(self, t):
        ph = (t / self.P + self.ph) % 1.0
        tri = 4.0 * np.abs(ph - 0.5) - 1.0
        return self.c + self.A * tri, self.A * 4.0 * np.sign(ph - 0.5) / self.P


def random_pose(rng, q_min, q_max, frac=0.15):
    """Pose inicial aleatoria dentro de los límites (con margen frac del rango)."""
    r = q_max - q_min
    return rng.uniform(q_min + frac * r, q_max - frac * r)


def _amp_center(rng, q_min, q_max, A):
    r = q_max - q_min
    m = 0.1 * r
    A = np.minimum(A, np.maximum(0.5 * r - m, 0.02))
    lo, hi = q_min + m + A, q_max - m - A
    c = np.where(hi > lo, rng.uniform(lo, np.maximum(hi, lo + 1e-9)), 0.5 * (q_min + q_max))
    return c, A


def make_trajectory(kind, rng, q_min, q_max, v_ref, T, q_start):
    """v_ref: velocidad máxima de referencia [rad/s] (p. ej. 0.6*u_max)."""
    n = len(q_min)
    r = q_max - q_min
    m = 0.1 * r
    if kind == "step":
        n_lv = int(rng.integers(1, 4))
        times = np.sort(rng.uniform(0.5, 0.75 * T, n_lv))
        cur = np.asarray(q_start, float)
        steps = []
        for t in times:
            d = rng.uniform(-0.8, 0.8, n)
            cur = np.clip(cur + d, q_min + m, q_max - m)
            steps.append((t, cur.copy()))
        return StepTraj(q_start, steps)
    if kind == "sine":
        f = rng.uniform(0.1, 0.5, n)
        w = 2 * np.pi * f
        A = rng.uniform(0.15, 0.7, n)
        A = np.minimum(A, v_ref / w)
        c, A = _amp_center(rng, q_min, q_max, A)
        phi = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0, 2 * np.pi, n))
        return SumSineTraj(c, A[None], w[None], phi[None], "sine")
    if kind == "triangle":
        P = rng.uniform(3.0, 8.0, n)
        A = rng.uniform(0.2, 0.8, n)
        A = np.minimum(A, v_ref * P / 4.0)
        c, A = _amp_center(rng, q_min, q_max, A)
        return TriangleTraj(c, A, P, rng.random(n))
    if kind == "random":
        K = 3
        f = rng.uniform(0.05, 0.6, (K, n))
        w = 2 * np.pi * f
        A = rng.uniform(0.05, 0.4, (K, n))
        scale = np.minimum(1.0, v_ref / (A * w).sum(0))        # limita la velocidad pico
        A = A * scale
        c, _ = _amp_center(rng, q_min, q_max, A.sum(0))
        return SumSineTraj(c, A, w, rng.uniform(0, 2 * np.pi, (K, n)), "random")
    raise ValueError(kind)


def make_step_single(q_start, amp, t0=0.5):
    """Escalón único de amplitud amp[j] (con signo) en todas las articulaciones a t0."""
    q_start = np.asarray(q_start, float)
    return StepTraj(q_start, [(t0, q_start + np.asarray(amp, float))])


def sample_free(robot, kind, rng, v_ref, T, tries=60, q0=None):
    """Pose inicial + trayectoria SIN colisiones (verificadas geométricamente con robot.collisions).
    Devuelve (q0, traj, ok, intentos). ok=False si no se encontró ninguna en `tries` intentos."""
    fixed = q0 is not None
    for k in range(1, tries + 1):
        p0 = np.asarray(q0, float) if fixed else random_pose(rng, robot.q_min, robot.q_max)
        if not robot.config_free(p0):
            continue
        traj = make_trajectory(kind, rng, robot.q_min, robot.q_max, v_ref, T, p0)
        if robot.path_free(p0, traj, T)[0]:
            return p0, traj, True, k
    return p0, traj, False, tries


def find_free_pose(robot, rng, base, amp, tries=200):
    """Pose base y amplitud de escalón sin colisión (prueba la pedida y luego poses aleatorias, con amp reducida)."""
    for k in range(tries):
        p = base if k == 0 else random_pose(rng, robot.q_min, robot.q_max)
        a = amp * (1.0 if k < tries // 2 else 0.5)
        if robot.config_free(p) and robot.config_free(np.clip(p + a, robot.q_min, robot.q_max)):
            return p, a, True
    return base, amp, False
