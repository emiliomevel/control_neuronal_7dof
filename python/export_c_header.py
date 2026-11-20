"""
Genera los headers del firmware:
  firmware/src/nn_config.h   -> arquitectura, normalización, límites (lo escribe 04_preprocesar.py)
  firmware/src/nn_weights.h  -> pesos fijos (los escribe 05/07 con --export, o este script)

Uso:  python export_c_header.py data/model_micro_directo.npz     # re-exporta pesos de un modelo guardado
"""
import argparse
import sys
from datetime import datetime

import numpy as np

import config as cfg


def _f(x):
    s = f"{float(x):.9g}"
    if not any(c in s for c in ".en"):      # "5" -> "5.0" (literal float válido en C)
        s += ".0"
    return s + "f"


def _arr(v, per_line=6, one_line=True):
    """one_line=True para macros #define (no pueden partirse sin '\\'); False para arreglos grandes."""
    v = list(v)
    if one_line:
        return "{ " + ", ".join(_f(x) for x in v) + " }"
    rows = [", ".join(_f(x) for x in v[i:i + per_line]) for i in range(0, len(v), per_line)]
    return "{\n    " + ",\n    ".join(rows) + "\n}"


def write_config(path, norm, h1, h2, q_min, q_max, u_max, margin=cfg.JOINT_MARGIN):
    n_in = len(norm.mean)
    n_out = 1 if norm.mode == "shared" else norm.n_joints
    txt = f"""/* GENERADO por 04_preprocesar.py ({datetime.now():%Y-%m-%d %H:%M}) -- NO editar a mano.
 * Modo: {norm.mode}   objetivo: {"residual" if norm.residual else "directo"} */
#ifndef NN_CONFIG_H
#define NN_CONFIG_H

#define NN_N_JOINTS   {norm.n_joints}
#define NN_SHARED     {1 if norm.mode == "shared" else 0}   /* 1: misma red por articulacion | 0: red completa */
#define NN_IS_RESIDUAL {1 if norm.residual else 0}   /* 1: la red entrega solo una correccion (u = u_PID + u_NN) */
#define NN_IN         {n_in}
#define NN_H1         {h1}
#define NN_H2         {h2}
#define NN_OUT        {n_out}
#define NN_X_CLIP     {_f(norm.clip)}
#define NN_Y_SCALE    {_f(norm.y_scale)}   /* u [rad/s] = y_norm * NN_Y_SCALE */
#define NN_JOINT_MARGIN {_f(margin)}

#define NN_X_MEAN_INIT   {_arr(norm.mean)}
#define NN_X_STD_INIT    {_arr(norm.std)}
#define NN_U_MAX_INIT    {_arr(np.broadcast_to(u_max, (norm.n_joints,)))}
#define NN_Q_MIN_INIT    {_arr(q_min)}
#define NN_Q_MAX_INIT    {_arr(q_max)}

#endif /* NN_CONFIG_H */
"""
    path.write_text(txt)
    print(f"[export] {path}")


def write_weights(path, params, arch, valid=True, note=""):
    params = np.asarray(params, np.float32)
    txt = f"""/* GENERADO por export_c_header.py ({datetime.now():%Y-%m-%d %H:%M}) -- arquitectura {arch}
 * {note}
 * Para usar estos pesos fijos: en nn_user_cfg.h pon NN_TRAIN_ON_BOARD = 0 (o NN_INIT_FROM_HEADER = 1). */
#ifndef NN_WEIGHTS_H
#define NN_WEIGHTS_H

#define NN_WEIGHTS_VALID {1 if valid else 0}
#define NN_WEIGHTS_COUNT {len(params)}u

static const float NN_WEIGHTS[{len(params)}] = {_arr(params, 8, one_line=False)};

#endif /* NN_WEIGHTS_H */
"""
    path.write_text(txt)
    print(f"[export] {path}  ({len(params)} parámetros)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="archivo .npz con 'params' y 'arch'")
    a = ap.parse_args()
    d = np.load(a.model, allow_pickle=True)
    write_weights(cfg.FW_SRC / "nn_weights.h", d["params"], str(d["arch"]), note=f"origen: {a.model}")


if __name__ == "__main__":
    sys.exit(main())
