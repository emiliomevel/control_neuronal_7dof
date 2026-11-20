"""
Sesión 2 · Paso 7: entrenamiento en PC (prototipo del algoritmo que luego corre en el micro).

Usa nn_numpy.py, que replica el firmware: misma inicialización (xorshift32), misma pérdida
  L = (1/N) Σ |u_i - û_i|²,  mismo backprop, mismo Adam y una actualización por trama de BATCH muestras.
Registra: loss train/val, MAE, RMSE (rad/s), tiempo de entrenamiento y número de parámetros.
Luego prueba generalización en trayectorias NO vistas (test) y desglosa el error por tipo de trayectoria.

Uso:   python 05_entrenar_pc.py [--objetivo directo|residual] [--epochs 40] [--export]
Salida: data/model_pc_<objetivo>.npz, data/history_pc_<objetivo>.csv, figuras en data/figs/
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
from nn_data import epoch_order, evaluate, lr_at, net_dims, save_json
from nn_numpy import MLP, Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objetivo", choices=["directo", "residual"], default="directo")
    ap.add_argument("--epochs", type=int, default=cfg.EPOCHS_PC)
    ap.add_argument("--batch", type=int, default=cfg.BATCH)
    ap.add_argument("--lr", type=float, default=cfg.LR)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--export", action="store_true", help="escribe firmware/src/nn_weights.h con los pesos finales")
    a = ap.parse_args()

    D = np.load(cfg.DATA / f"proc_{a.objetivo}.npz", allow_pickle=True)
    mode, y_scale = str(D["mode"]), float(D["y_scale"])
    Xtr, Ytr, Xva, Yva, Xte, Yte = (D[k] for k in ("X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test"))
    n_in, n_out = net_dims(mode, int(D["n_joints"]))
    mlp = MLP(n_in, cfg.H1, cfg.H2, n_out)
    mlp.randomize(a.seed)
    tr = Trainer(mlp, cfg.OPTIMIZER, b1=cfg.ADAM_B1, b2=cfg.ADAM_B2, eps=cfg.ADAM_EPS, mu=cfg.SGD_MU)
    print(f"Red {n_in}-{cfg.H1}-{cfg.H2}-{n_out}  parámetros={mlp.n_params}  optimizador={cfg.OPTIMIZER}  "
          f"batch={a.batch}  train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}")

    hist, t_train = [], 0.0
    best = (np.inf, None, 0)
    for ep in range(a.epochs):
        lr = lr_at(ep, a.epochs, a.lr)
        order = epoch_order(len(Xtr), a.seed, ep)
        nb = len(order) // a.batch                      # se descarta la trama incompleta (igual que el micro)
        t0 = time.perf_counter()
        loss_sum = 0.0
        for b in range(nb):
            idx = order[b * a.batch:(b + 1) * a.batch]
            loss_sum += tr.train_batch(Xtr[idx], Ytr[idx], lr)
        dt = time.perf_counter() - t0
        t_train += dt
        mv = evaluate(mlp, Xva, Yva, y_scale)
        row = dict(epoch=ep + 1, lr=lr, train_loss=loss_sum / (nb * a.batch), val_loss=mv["loss"],
                   val_mae=mv["mae"], val_rmse=mv["rmse"], epoch_time_s=dt)
        hist.append(row)
        if mv["loss"] < best[0]:
            best = (mv["loss"], mlp.p.copy(), ep + 1)
        print(f"  ép {ep + 1:3d}/{a.epochs}  lr={lr:.2e}  loss_tr={row['train_loss']:.5f}  loss_val={mv['loss']:.5f}  "
              f"MAE={mv['mae']:.4f}  RMSE={mv['rmse']:.4f} rad/s  ({dt:.1f}s)")

    mlp.set_params(best[1])
    print(f"\nMejores pesos: época {best[2]}  (val_loss={best[0]:.5f}).  Tiempo total de entrenamiento: {t_train:.1f} s")
    m_tr, m_va, m_te = (evaluate(mlp, X, Y, y_scale) for X, Y in ((Xtr, Ytr), (Xva, Yva), (Xte, Yte)))
    print(f"{'':7s}{'loss':>10s}{'MAE[rad/s]':>12s}{'RMSE[rad/s]':>13s}")
    for nm, m in (("train", m_tr), ("val", m_va), ("test", m_te)):
        print(f"{nm:7s}{m['loss']:10.5f}{m['mae']:12.4f}{m['rmse']:13.4f}")

    # ---- ¿generaliza o memorizó? error por tipo de trayectoria en test (episodios no vistos)
    kinds = [str(k) for k in D["kinds"]]
    print("\nError en TEST por tipo de trayectoria (episodios no vistos en entrenamiento):")
    per_kind = {}
    for ki, kn in enumerate(kinds):
        sel = D["kind_test"] == ki
        if sel.any():
            per_kind[kn] = evaluate(mlp, Xte[sel], Yte[sel], y_scale)
            print(f"  {kn:9s} RMSE={per_kind[kn]['rmse']:.4f} rad/s  MAE={per_kind[kn]['mae']:.4f}  (n={int(sel.sum())})")
    gap = m_te["rmse"] / max(m_tr["rmse"], 1e-9)
    print(f"  RMSE_test / RMSE_train = {gap:.2f}   (≈1: generaliza · ≫1: memorizó)")

    arch = f"{n_in}-{cfg.H1}-{cfg.H2}-{n_out}"
    np.savez(cfg.DATA / f"model_pc_{a.objetivo}.npz", params=mlp.p.copy(), arch=arch, mode=mode, objetivo=a.objetivo,
             n_in=n_in, h1=cfg.H1, h2=cfg.H2, n_out=n_out, seed=a.seed)
    pd.DataFrame(hist).to_csv(cfg.DATA / f"history_pc_{a.objetivo}.csv", index=False)
    save_json(cfg.DATA / f"metrics_pc_{a.objetivo}.json", dict(
        n_params=mlp.n_params, train_time_s=t_train, best_epoch=best[2], train=m_tr, val=m_va, test=m_te,
        per_kind=per_kind, gen_gap=gap, arch=arch, optimizer=cfg.OPTIMIZER, batch=a.batch))

    # ---- figuras
    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].semilogy(h.epoch, h.train_loss, label="train"); ax[0].semilogy(h.epoch, h.val_loss, label="val")
    ax[0].set_title("Pérdida L (normalizada)"); ax[0].set_xlabel("época"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(h.epoch, h.val_rmse, label="RMSE"); ax[1].plot(h.epoch, h.val_mae, label="MAE")
    ax[1].set_title("Validación [rad/s]"); ax[1].set_xlabel("época"); ax[1].legend(); ax[1].grid(alpha=.3)
    n_plot = min(2000, len(Xte))
    yp = mlp.forward(Xte[:n_plot]).ravel() * y_scale
    yt = Yte[:n_plot].ravel() * y_scale
    ax[2].scatter(yt, yp, s=3, alpha=.4)
    lim = np.abs(yt).max() * 1.05
    ax[2].plot([-lim, lim], [-lim, lim], "k--", lw=.8)
    ax[2].set_xlabel("u_PID [rad/s]" if a.objetivo == "directo" else "objetivo residual [rad/s]")
    ax[2].set_ylabel("u_NN [rad/s]"); ax[2].set_title("Test: PID vs NN"); ax[2].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(cfg.FIGS / f"05_entrenamiento_pc_{a.objetivo}.png", dpi=130)

    if a.export:
        write_weights(cfg.FW_SRC / "nn_weights.h", mlp.p, arch, note=f"entrenado en PC, objetivo {a.objetivo}")


if __name__ == "__main__":
    main()
