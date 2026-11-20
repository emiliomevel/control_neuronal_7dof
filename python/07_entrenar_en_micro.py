"""
Sesión 2/3 · Entrenamiento DENTRO del micro (S32K312 mini / S32K344).

La PC manda el dataset por UART en tramas de BATCH muestras (con BATCH=1 es muestra por muestra);
el micro hace forward + backprop de cada muestra, acumula el gradiente, da UN paso del optimizador
por trama y descarta las muestras: NO se guarda el dataset en el micro.
Después de cada época la PC manda el conjunto de validación (solo forward) y el micro devuelve
las sumas de error; la PC calcula loss / MAE / RMSE. Al final la PC lee los pesos (GET_W) y,
con --export, escribe firmware/src/nn_weights.h para fijarlos (NN_TRAIN_ON_BOARD = 0).

Uso:   python 07_entrenar_en_micro.py [--url COM5@921600 | exec:../firmware/host_sim/micro_sim]
                                      [--objetivo directo|residual] [--epochs 10] [--batch 16]
                                      [--max-train 20000] [--init-from-pc] [--export]
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
from export_c_header import write_weights
from micro_link import Micro
from nn_data import epoch_order, lr_at, metrics_phys, net_dims, save_json


def stream_eval(mic, X, Y, frame):
    se = ae = 0.0
    cyc = 0
    for i in range(0, len(X), frame):
        s, a, c = mic.eval_batch(X[i:i + frame], Y[i:i + frame])
        se += s; ae += a; cyc += c
    return se, ae, cyc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=cfg.MICRO_URL)
    ap.add_argument("--objetivo", choices=["directo", "residual"], default="directo")
    ap.add_argument("--epochs", type=int, default=cfg.EPOCHS_MICRO)
    ap.add_argument("--batch", type=int, default=cfg.BATCH)
    ap.add_argument("--lr", type=float, default=cfg.LR)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--max-train", type=int, default=0, help="muestras de entrenamiento por época (0 = todas)")
    ap.add_argument("--max-val", type=int, default=0, help="muestras de validación por época (0 = todas)")
    ap.add_argument("--init-from-pc", action="store_true", help="arrancar con los pesos entrenados en PC")
    ap.add_argument("--keep-best", action="store_true", default=True)
    ap.add_argument("--export", action="store_true", help="escribe firmware/src/nn_weights.h con los pesos finales")
    a = ap.parse_args()

    D = np.load(cfg.DATA / f"proc_{a.objetivo}.npz", allow_pickle=True)
    mode, y_scale = str(D["mode"]), float(D["y_scale"])
    Xtr, Ytr, Xva, Yva, Xte, Yte = (D[k] for k in ("X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test"))
    n_in, n_out = net_dims(mode, int(D["n_joints"]))

    mic = Micro(a.url)
    mic.ping()
    info = mic.info()
    print("Micro:", info)
    if not info["train_on_board"]:
        raise SystemExit("Este firmware está compilado con pesos fijos (NN_TRAIN_ON_BOARD=0): no puede entrenar.")
    assert (info["n_in"], info["h1"], info["h2"], info["n_out"]) == (n_in, cfg.H1, cfg.H2, n_out), \
        "La arquitectura del firmware (nn_config.h) no coincide con la del dataset: corre 04_preprocesar.py y recompila."
    assert info["shared"] == (mode == "shared") and info["residual"] == (a.objetivo == "residual"), \
        "nn_config.h no corresponde a este modo/objetivo: corre 04_preprocesar.py --objetivo ... y recompila."
    assert info["adam"] == (cfg.OPTIMIZER == "adam"), "NN_USE_ADAM del firmware ≠ OPTIMIZER de config.py"
    frame = min(a.batch, info["max_samples"])
    eval_frame = min(info["max_samples"], 64)      # tramas de evaluación moderadas (a 115200 una de 146 muestras tarda 0.36 s)

    if a.init_from_pc:
        mic.set_weights(np.load(cfg.DATA / f"model_pc_{a.objetivo}.npz")["params"])
        print("Pesos iniciales: los del entrenamiento en PC (ajuste fino en el micro)")
    else:
        mic.init(a.seed)
    print(f"Entrenando en el micro: {len(Xtr)} muestras/época (máx {a.max_train or 'todas'}), trama={frame}, "
          f"{a.epochs} épocas, red {n_in}-{cfg.H1}-{cfg.H2}-{n_out} ({info['n_params']} parámetros)\n")

    Xva_u = Xva[:a.max_val] if a.max_val else Xva
    Yva_u = Yva[:a.max_val] if a.max_val else Yva
    hist, best = [], (np.inf, None, 0)
    t_wall0 = time.time()
    micro_time_s = 0.0
    for ep in range(a.epochs):
        lr = lr_at(ep, a.epochs, a.lr)
        mic.set_lr(lr)
        order = epoch_order(len(Xtr), a.seed, ep)
        if a.max_train:
            order = order[:a.max_train]
        nb = len(order) // frame
        t0 = time.time()
        loss_sum, cyc = 0.0, 0
        for b in range(nb):
            idx = order[b * frame:(b + 1) * frame]
            l, c = mic.train_batch(Xtr[idx], Ytr[idx])
            loss_sum += l; cyc += c
        t_ep = time.time() - t0
        micro_time_s += cyc / info["core_hz"]
        se, ae, cv = stream_eval(mic, Xva_u, Yva_u, eval_frame)
        mv = metrics_phys(se, ae, len(Xva_u), y_scale)
        row = dict(epoch=ep + 1, lr=lr, train_loss=loss_sum / (nb * frame), val_loss=mv["loss"], val_mae=mv["mae"],
                   val_rmse=mv["rmse"], epoch_wall_s=t_ep, epoch_micro_compute_s=cyc / info["core_hz"],
                   samples_per_s=nb * frame / t_ep)
        hist.append(row)
        if mv["loss"] < best[0]:
            best = (mv["loss"], mic.get_weights(), ep + 1)
        done = ep + 1
        eta = (time.time() - t_wall0) / done * (a.epochs - done)
        print(f"  ép {done:3d}/{a.epochs}  lr={lr:.2e}  loss_tr={row['train_loss']:.5f}  loss_val={mv['loss']:.5f}  "
              f"MAE={mv['mae']:.4f}  RMSE={mv['rmse']:.4f} rad/s  | {t_ep:.1f}s ({row['samples_per_s']:.0f} muestras/s, "
              f"cómputo en micro {row['epoch_micro_compute_s']:.2f}s)  ETA {eta:.0f}s")

    if a.keep_best and best[1] is not None:
        mic.set_weights(best[1])                    # el micro se queda con los mejores pesos
        print(f"\nPesos de la mejor época ({best[2]}) cargados en el micro")
    w = mic.get_weights()
    res = {}
    for nm, X, Y in (("val", Xva, Yva), ("test", Xte, Yte)):
        se, ae, _ = stream_eval(mic, X, Y, eval_frame)
        res[nm] = metrics_phys(se, ae, len(X), y_scale)
    print(f"\n{'':6s}{'loss':>10s}{'MAE[rad/s]':>12s}{'RMSE[rad/s]':>13s}")
    for nm in ("val", "test"):
        m = res[nm]
        print(f"{nm:6s}{m['loss']:10.5f}{m['mae']:12.4f}{m['rmse']:13.4f}")

    arch = f"{n_in}-{cfg.H1}-{cfg.H2}-{n_out}"
    wall = time.time() - t_wall0
    np.savez(cfg.DATA / f"model_micro_{a.objetivo}.npz", params=w, arch=arch, mode=mode, objetivo=a.objetivo,
             n_in=n_in, h1=cfg.H1, h2=cfg.H2, n_out=n_out, seed=a.seed)
    pd.DataFrame(hist).to_csv(cfg.DATA / f"history_micro_{a.objetivo}.csv", index=False)
    save_json(cfg.DATA / f"metrics_micro_{a.objetivo}.json", dict(
        n_params=info["n_params"], ram_bytes_nn=info["ram_bytes"], flash_bytes_nn=info["flash_bytes"],
        core_hz=info["core_hz"], infer_cycles=info["infer_cycles"],
        infer_us=1e6 * info["infer_cycles"] / info["core_hz"], train_compute_s_micro=micro_time_s,
        train_wall_s=wall, best_epoch=best[2], val=res["val"], test=res["test"], batch=frame,
        bytes_tx=mic.bytes_tx, bytes_rx=mic.bytes_rx, arch=arch))
    print(f"\nTiempo de entrenamiento: {wall:.0f} s de pared (UART incluida) · {micro_time_s:.1f} s de cómputo puro en el micro")
    print(f"Inferencia (7 articulaciones): {info['infer_cycles']} ciclos = {1e6 * info['infer_cycles'] / info['core_hz']:.1f} µs")
    print(f"RAM de la red: {info['ram_bytes']} B   ·   Flash de constantes: {info['flash_bytes']} B   "
          "(la RAM/Flash TOTAL del firmware se lee del .map / Print Size de S32DS)")

    h = pd.DataFrame(hist)
    plt.figure(figsize=(6.5, 4))
    plt.semilogy(h.epoch, h.train_loss, label="train (micro)")
    plt.semilogy(h.epoch, h.val_loss, label="val (micro)")
    f_pc = cfg.DATA / f"history_pc_{a.objetivo}.csv"
    if f_pc.exists():
        hp = pd.read_csv(f_pc)
        plt.semilogy(hp.epoch * (a.epochs / max(len(hp), 1)), hp.val_loss, "--", alpha=.6, label="val (PC)")
    plt.xlabel("época"); plt.ylabel("L"); plt.grid(alpha=.3); plt.legend(); plt.title("Entrenamiento en el micro")
    plt.tight_layout(); plt.savefig(cfg.FIGS / f"07_entrenamiento_micro_{a.objetivo}.png", dpi=130)

    if a.export:
        write_weights(cfg.FW_SRC / "nn_weights.h", w, arch, note=f"entrenado EN EL MICRO, objetivo {a.objetivo}")
        print("Ahora: nn_user_cfg.h -> NN_TRAIN_ON_BOARD 0, recompila y graba: la red queda fija sin reentrenar.")
    mic.close()


if __name__ == "__main__":
    main()