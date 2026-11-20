"""
Utilidades compartidas: construcción de entradas, normalización, orden de épocas, lr, métricas.
Orden de las 6 características por articulación (igual que el firmware):  [q, dq, qr, dqr, e, de]
con e = qr - q  y  de = dqr - dq.
"""
import json
from dataclasses import dataclass

import numpy as np

import config as cfg


def build_features(q, dq, qr, dqr, mode):
    """Arrays (T,n) o (n,)  ->  'shared': (T*n, 6) [fila = t*n + j]   |   'full': (T, 6n) [bloques por articulación]."""
    q, dq, qr, dqr = (np.atleast_2d(np.asarray(a, float)) for a in (q, dq, qr, dqr))
    F = np.stack([q, dq, qr, dqr, qr - q, dqr - dq], axis=-1)          # (T, n, 6)
    return F.reshape(-1, 6) if mode == "shared" else F.reshape(F.shape[0], -1)


def make_targets(u, mode):
    u = np.atleast_2d(np.asarray(u, float))
    return u.reshape(-1, 1) if mode == "shared" else u


@dataclass
class Norm:
    mean: np.ndarray
    std: np.ndarray
    y_scale: float          # u [rad/s] = y_norm * y_scale
    clip: float
    mode: str
    residual: bool
    n_joints: int

    def x(self, X):
        return np.clip((np.asarray(X) - self.mean) / self.std, -self.clip, self.clip).astype(np.float32)

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std, y_scale=self.y_scale, clip=self.clip,
                 mode=self.mode, residual=self.residual, n_joints=self.n_joints)

    @staticmethod
    def load(d):
        return Norm(np.asarray(d["mean"]), np.asarray(d["std"]), float(d["y_scale"]), float(d["clip"]),
                    str(d["mode"]), bool(d["residual"]), int(d["n_joints"]))


def net_dims(mode, n_joints):
    return (6, 1) if mode == "shared" else (6 * n_joints, n_joints)


def epoch_order(n, seed, epoch):
    """Mismo orden de muestras en PC y micro (para poder comparar el entrenamiento)."""
    return np.random.RandomState(seed * 1000 + epoch).permutation(n)


def lr_at(epoch, epochs, lr0=cfg.LR, min_frac=cfg.LR_MIN_FRAC):
    c = 0.5 * (1 + np.cos(np.pi * epoch / max(epochs - 1, 1)))
    return float(lr0 * (min_frac + (1 - min_frac) * c))


def metrics_phys(se_sum, ae_sum, n, y_scale):
    """De sumas normalizadas a pérdida L, MAE y RMSE en rad/s."""
    return dict(loss=se_sum / n, mae=y_scale * ae_sum / n, rmse=y_scale * np.sqrt(se_sum / n))


def evaluate(mlp, X, T, y_scale, chunk=4096):
    se = ae = 0.0
    for i in range(0, len(X), chunk):
        a, b = mlp.eval_sums(X[i:i + chunk], T[i:i + chunk])
        se += a
        ae += b
    return metrics_phys(se, ae, len(X), y_scale)


def save_json(path, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)
