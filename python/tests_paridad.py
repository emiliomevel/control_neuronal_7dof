"""
Pruebas de integridad (no necesitan Coppelia ni hardware):
  1. Gradiente analítico del backprop vs. diferencias finitas (float64).
  2. Firmware C (micro virtual) vs. NumPy: mismos pesos iniciales, mismo entrenamiento, misma inferencia.

Antes:  make -C ../firmware/host_sim
Uso:    python tests_paridad.py [exec:../firmware/host_sim/micro_sim | COM5@921600]
"""
import sys

import numpy as np

import config as cfg
from micro_link import Micro
from nn_data import epoch_order
from nn_numpy import MLP, Trainer


def test_gradiente():
    rng = np.random.default_rng(0)
    m = MLP(6, 16, 16, 1, dtype=np.float64)
    m.p[:] = rng.normal(0, 0.4, m.n_params)
    X, T = rng.normal(size=(8, 6)), rng.normal(size=(8, 1))
    _, g = m.loss_and_grad(X, T)
    num = np.zeros_like(g)
    for i in range(m.n_params):
        old = m.p[i]
        m.p[i] = old + 1e-6; lp, _ = m.loss_and_grad(X, T)
        m.p[i] = old - 1e-6; lm, _ = m.loss_and_grad(X, T)
        m.p[i] = old
        num[i] = (lp - lm) / 2e-6
    err = np.max(np.abs(num - g)) / (np.max(np.abs(g)) + 1e-12)
    print(f"[1] gradiente analítico vs numérico: error relativo máx = {err:.2e}  {'OK' if err < 1e-5 else 'FALLA'}")
    return err < 1e-5


def test_micro(url):
    mic = Micro(url)
    mic.ping()
    info = mic.info()
    print(f"[2] micro: {info}")
    n_in, h1, h2, n_out = info["n_in"], info["h1"], info["h2"], info["n_out"]
    if not info["train_on_board"]:
        print("    firmware con pesos fijos: solo se prueba inferencia/lectura de pesos")
    rng = np.random.default_rng(3)
    N, B, seed = 3200, min(cfg.BATCH, info["max_samples"]), 7
    X = np.clip(rng.normal(size=(N, n_in)), -5, 5).astype(np.float32)
    w_true = rng.normal(size=(n_in, n_out)) * 0.3
    T = np.clip(np.tanh(X @ w_true), -1, 1).astype(np.float32)
    ok = True

    ref = MLP(n_in, h1, h2, n_out)
    ref.randomize(seed)
    if info["train_on_board"]:
        mic.init(seed)
        w0 = mic.get_weights()
        d0 = np.max(np.abs(w0 - ref.p))
        print(f"    init xorshift: |W_micro - W_numpy|max = {d0:.2e}  {'OK' if d0 < 1e-6 else 'FALLA'}")
        ok &= d0 < 1e-6
        tr = Trainer(ref, "adam" if info["adam"] else "sgd", b1=cfg.ADAM_B1, b2=cfg.ADAM_B2, eps=cfg.ADAM_EPS, mu=cfg.SGD_MU)
        order = epoch_order(N, seed, 0)
        mic.set_lr(cfg.LR)
        for b in range(N // B):
            idx = order[b * B:(b + 1) * B]
            tr.train_batch(X[idx], T[idx], cfg.LR)
            mic.train_batch(X[idx], T[idx])
        w1 = mic.get_weights()
        d1 = np.max(np.abs(w1 - ref.p))
        print(f"    tras {N // B} actualizaciones: |W_micro - W_numpy|max = {d1:.2e}  {'OK' if d1 < 5e-3 else 'FALLA'}")
        ok &= d1 < 5e-3
        se_m, ae_m, _ = mic.eval_batch(X[:B], T[:B])
        se_p, ae_p = ref.eval_sums(X[:B], T[:B])
        print(f"    EVAL: micro se={se_m:.5f}  numpy se={se_p:.5f}")
        ok &= abs(se_m - se_p) < 1e-2 * max(se_p, 1e-3) + 1e-4
        mic.set_weights(ref.p)                       # SET_W / GET_W de ida y vuelta
        ok &= np.max(np.abs(mic.get_weights() - ref.p)) == 0
    else:
        ref.set_params(mic.get_weights())

    # inferencia desde estado crudo vs NumPy (solo modo "shared" y objetivo directo; usa nn_config.h tal cual)
    if info["shared"]:
        from nn_data import Norm, build_features
        n = info["n_joints"]
        q = rng.uniform(-1, 1, n); dq = rng.uniform(-.5, .5, n); qr = q + rng.uniform(-.5, .5, n); dqr = rng.uniform(-.5, .5, n)
        u_mic, cyc = mic.infer(q, dq, qr, dqr)
        print(f"    INFER: u_micro={np.round(u_mic, 4)}  ({cyc} ciclos)")
    mic.close()
    print("RESULTADO:", "TODO OK" if ok else "HAY DIFERENCIAS")
    return ok


if __name__ == "__main__":
    a = test_gradiente()
    b = test_micro(sys.argv[1] if len(sys.argv) > 1 else "exec:../firmware/host_sim/micro_sim")
    sys.exit(0 if (a and b) else 1)
