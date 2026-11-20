"""
Lazo de simulación común (dataset, PID, NN, residual, micro) + perturbaciones + métricas.
Todos los controladores se evalúan con exactamente el mismo lazo y las mismas condiciones.
"""
from dataclasses import dataclass, field

import numpy as np

import config as cfg
from controllers import safety


@dataclass
class Perturbation:
    payload_kg: float = 0.0          # masa extra en el efector final
    motor_gain: float = 1.0          # variación de parámetros del motor: u_aplicada = gain * u
    pulse_t0: float = None           # perturbación tipo "empujón": suma a la velocidad comandada
    pulse_dur: float = 0.3
    pulse_amp: np.ndarray = None     # [rad/s] por articulación
    noise_q: float = 0.0             # ruido de medición (std) de q [rad]
    noise_dq: float = 0.0            # ruido de medición (std) de dq [rad/s]
    label: str = "nominal"

    def pulse(self, t, n):
        if self.pulse_t0 is not None and self.pulse_t0 <= t < self.pulse_t0 + self.pulse_dur:
            return self.pulse_amp if self.pulse_amp is not None else np.zeros(n)
        return np.zeros(n)


def run_episode(robot, ctrl, traj, T, q0, pert=None, seed=0):
    """Devuelve dict de arrays (steps, n):
       q, dq  : estado verdadero         qm, dqm: estado medido (lo que ve el controlador)
       qr, dqr: referencia               u_cmd  : salida del controlador (antes de seguridad)
       u_app  : lo realmente aplicado (después de seguridad, ganancia del motor y empujón)"""
    pert = pert or Perturbation()
    rng = np.random.default_rng(seed)
    n, dt = robot.n, robot.dt
    robot.reset(q0, payload_kg=pert.payload_kg)
    ctrl.reset()
    steps = int(round(T / dt))
    keys = ("t", "q", "dq", "qm", "dqm", "qr", "dqr", "u_cmd", "u_app")
    L = {k: [] for k in keys}
    for k in range(steps):
        t = k * dt
        q, dq = robot.read()
        qm = q + pert.noise_q * rng.standard_normal(n)
        dqm = dq + pert.noise_dq * rng.standard_normal(n)
        qr, dqr = traj.eval(t)
        u_cmd = np.asarray(ctrl(qm, dqm, qr, dqr), float)
        u_safe = safety(qm, u_cmd, robot.u_max, robot.q_min, robot.q_max)
        u_app = pert.motor_gain * u_safe + pert.pulse(t, n)
        robot.apply(u_app)
        robot.step()
        for key, val in zip(keys, (t, q, dq, qm, dqm, qr, dqr, u_cmd, u_app)):
            L[key].append(val)
    return {k: np.array(v) for k, v in L.items()}


# --------------------------------------------------------------------------- métricas
def tracking_metrics(ep, dt):
    e = ep["qr"] - ep["q"]
    u = ep["u_app"]
    du = np.diff(u, axis=0) / dt
    k_ss = int(round(2.0 / dt))                                 # régimen: después del transitorio inicial (2 s)
    return dict(
        rmse=float(np.sqrt(np.mean(e ** 2))),
        rmse_ss=float(np.sqrt(np.mean(e[k_ss:] ** 2))) if len(e) > k_ss + 5 else float("nan"),
        e_max=float(np.max(np.abs(e))),
        u_rms=float(np.sqrt(np.mean(u ** 2))),                  # esfuerzo de control (RMS)
        u_int=float(np.sum(np.abs(u)) * dt / u.shape[1]),       # ∫|u| dt promedio por articulación
        u_rough=float(np.sqrt(np.mean(du ** 2))),               # suavidad: RMS de du/dt [rad/s²]
    )


def step_metrics(ep, dt, t_step, band_frac=0.02, band_min=0.005):
    """Sobreimpulso [%] y tiempo de establecimiento [s] para un escalón único (media sobre articulaciones)."""
    k0 = int(round(t_step / dt))
    q, qr = ep["q"], ep["qr"]
    target = qr[-1]
    amp = target - qr[0]
    os_, ts_ = [], []
    for j in range(q.shape[1]):
        a = amp[j]
        if abs(a) < 1e-6:
            continue
        y = q[k0:, j]
        os_.append(100.0 * max(0.0, np.max((y - target[j]) * np.sign(a))) / abs(a))
        band = max(band_frac * abs(a), band_min)
        out = np.where(np.abs(y - target[j]) > band)[0]
        ts_.append(np.nan if len(out) and out[-1] >= len(y) - 1 else (0.0 if len(out) == 0 else (out[-1] + 1) * dt))
    return dict(overshoot_pct=float(np.nanmean(os_)), settle_s=float(np.nanmean(ts_)) if not np.all(np.isnan(ts_)) else np.nan,
                settle_fail=int(np.sum(np.isnan(ts_))))
