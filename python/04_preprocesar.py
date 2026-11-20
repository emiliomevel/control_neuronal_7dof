"""
Sesión 1 · Paso 5: preprocesamiento.

 1. Eliminación de datos inválidos (NaN/inf, |u| fuera de límite).
 2. Saturación física de variables (dq, dqr, e, de, q dentro de límites).
 3. Eliminación de muestras anómalas (z-score > Z_OUT en alguna característica).
 4. División POR EPISODIOS (estratificada por tipo de trayectoria), así las trayectorias de prueba
    y validación nunca aparecen en entrenamiento:
        test = TEST_FRAC del total (15 %)  ·  val = VAL_FRAC del total (15 %) tomado del 85 % restante
        train = el resto (≈ 70 % del total)
 5. Normalización (media/desviación SOLO del conjunto de entrenamiento) + saturación ±X_CLIP.

Escribe data/proc_<objetivo>.npz y firmware/src/nn_config.h (arquitectura + normalización + límites).

  --objetivo directo   : la red reemplaza al PID          (y = u_PID)
  --objetivo residual  : la red aprende una corrección    (y = u_teacher - u_PID), u = u_PID + u_NN
Uso:   python 04_preprocesar.py [--objetivo directo|residual]
"""
import argparse

import numpy as np

import config as cfg
from export_c_header import write_config
from nn_data import Norm, build_features, make_targets, net_dims, save_json


def split_episodes(ep_ids, kinds, rng):
    """Devuelve (train_eps, val_eps, test_eps) estratificado por tipo de trayectoria."""
    tr, va, te = [], [], []
    for k in np.unique(kinds):
        eps = np.unique(ep_ids[kinds == k])
        rng.shuffle(eps)
        n = len(eps)
        n_te = max(1, int(round(cfg.TEST_FRAC * n))) if n >= 3 else 0
        n_va = max(1, int(round(cfg.VAL_FRAC * n))) if n >= 3 else 0
        te += list(eps[:n_te])
        va += list(eps[n_te:n_te + n_va])
        tr += list(eps[n_te + n_va:])
    return np.array(tr), np.array(va), np.array(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objetivo", choices=["directo", "residual"], default="directo")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    a = ap.parse_args()
    residual = a.objetivo == "residual"
    mode = cfg.NET_MODE

    R = np.load(cfg.DATA / "dataset_raw.npz", allow_pickle=True)
    q, dq, qr, dqr = (R[k].astype(float) for k in ("q", "dq", "qr", "dqr"))
    u = R["u_teacher"] - R["u"] if residual else R["u"]
    u_max = float(R["u_max"])
    q_min, q_max = R["q_min"], R["q_max"]
    ep, kind = R["ep"], R["kind"]
    n_joints = q.shape[1]
    y_scale = cfg.RESIDUAL_MAX if residual else u_max
    T0 = len(q)
    print(f"Dataset crudo: {T0} instantes × {n_joints} articulaciones   objetivo={a.objetivo}  modo={mode}")

    # 1) inválidos
    ok = np.isfinite(np.concatenate([q, dq, qr, dqr, u], axis=1)).all(1)
    ok &= (np.abs(R["u"]) <= u_max * 1.001).all(1)
    print(f"  1) inválidos descartados: {int((~ok).sum())}")

    # 2) saturación física
    q = np.clip(q, q_min, q_max)
    dq = np.clip(dq, -2 * u_max, 2 * u_max)
    dqr = np.clip(dqr, -2 * u_max, 2 * u_max)
    qr = np.clip(qr, q - cfg.E_CLIP, q + cfg.E_CLIP)             # => |e| <= E_CLIP
    u = np.clip(u, -y_scale, y_scale)

    # 3) anómalas (z-score global de cada característica, compartida entre articulaciones)
    F = build_features(q, dq, qr, dqr, "full").reshape(T0, n_joints, 6)
    mu, sd = F.reshape(-1, 6).mean(0), F.reshape(-1, 6).std(0) + 1e-9
    z_bad = (np.abs((F - mu) / sd) > cfg.Z_OUT).any(axis=(1, 2))
    print(f"  3) anómalas (|z|>{cfg.Z_OUT}) descartadas: {int((z_bad & ok).sum())}")
    ok &= ~z_bad
    keep = np.where(ok)[0]
    print(f"     instantes finales: {len(keep)}  ({100 * len(keep) / T0:.1f} %)")

    # 4) split por episodios
    rng = np.random.default_rng(a.seed)
    tr_e, va_e, te_e = split_episodes(ep[keep], kind[keep], rng)
    sel = {nm: keep[np.isin(ep[keep], e)] for nm, e in (("train", tr_e), ("val", va_e), ("test", te_e))}
    tot = sum(len(v) for v in sel.values())
    for nm, v in sel.items():
        print(f"  4) {nm:5s}: {len(v):6d} instantes ({100 * len(v) / tot:4.1f} %)  episodios={len(np.unique(ep[v]))}")

    # 5) normalización con estadísticos de entrenamiento
    def XY(idx):
        return build_features(q[idx], dq[idx], qr[idx], dqr[idx], mode), make_targets(u[idx], mode)

    Xtr, Ytr = XY(sel["train"])
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6
    norm = Norm(mean.astype(np.float32), std.astype(np.float32), y_scale, cfg.X_CLIP, mode, residual, n_joints)
    out = dict(mode=mode, objetivo=a.objetivo, n_joints=n_joints, u_max=u_max, y_scale=y_scale)
    for nm, idx in sel.items():
        X, Y = XY(idx)
        out[f"X_{nm}"] = norm.x(X)
        out[f"Y_{nm}"] = (Y / y_scale).astype(np.float32)
        out[f"ep_{nm}"] = np.repeat(ep[idx], n_joints) if mode == "shared" else ep[idx]
        out[f"kind_{nm}"] = np.repeat(kind[idx], n_joints) if mode == "shared" else kind[idx]
    norm.save(cfg.DATA / f"norm_{a.objetivo}.npz")
    np.savez(cfg.DATA / f"proc_{a.objetivo}.npz", **out, kinds=R["kinds"])
    print(f"\n  muestras (entrada de la red): train={len(out['X_train'])}  val={len(out['X_val'])}  test={len(out['X_test'])}")

    n_in, n_out = net_dims(mode, n_joints)
    print(f"  red: {n_in} -> {cfg.H1} -> {cfg.H2} -> {n_out}")
    write_config(cfg.FW_SRC / "nn_config.h", norm, cfg.H1, cfg.H2, q_min, q_max, np.full(n_joints, u_max))
    save_json(cfg.DATA / f"split_{a.objetivo}.json", dict(train=tr_e, val=va_e, test=te_e))


if __name__ == "__main__":
    main()
