"""
Sesión 3 · Paso 14: evaluación de estabilidad y seguridad del controlador neuronal.

Responde con números a la pregunta de la práctica: ¿la salida de la red puede generar saturación, oscilaciones,
comandos excesivos, inestabilidad o movimientos inesperados?  Tiene dos partes:

  A) OFFLINE (no necesita robot ni CoppeliaSim; usa solo los pesos entrenados y el dataset):
     A1  salida acotada ante entradas extremas / NaN / inf
     A2  cuánto del dataset está en saturación (PID vs NN)
     A3  ganancias efectivas locales de la red:  kp_ef = -du/dq,  kd_ef = -du/dq̇   (linealización del lazo)
         La planta es casi un integrador (comando de velocidad, ganancia ≈ 1), así que el lazo es estable si
         kp_ef > 0 y kp_ef*dt < 2 (discreto). Se reporta qué fracción de estados cumple esto.
     A4  la capa de seguridad nunca deja pasar |u|>u_max ni movimiento hacia afuera de los límites articulares
     A5  figura: ley de control aprendida u(e) frente a kp*e del PID, incluso para errores fuera del dataset

  B) LAZO CERRADO (necesita robot; --mock para probar sin CoppeliaSim):
     PID vs red (o PID vs PID+NN con --objetivo residual) en escenarios adversos: ganancia del motor +-50 %,
     empujón fuerte, ruido x10 y un fallo de sensor (NaN durante 0.1 s). Por escenario se reporta:
     |u| máx antes de la capa de seguridad, % de pasos en que la seguridad intervino, violación de límites
     articulares, índice de oscilación (RMS de du/dt y cruces por cero del error), RMSE en régimen y si recupera.

Uso:   python 09_analisis_seguridad.py [--objetivo directo|residual] [--modelo pc|micro] [--mock] [--solo-offline]
Salidas: data/seguridad_<obj>_<modelo>.json / .csv  y  data/figs/09_seguridad_<obj>_<modelo>.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
from controllers import PID, NNController, ResidualController, safety
from nn_data import Norm, build_features, save_json
from nn_numpy import MLP


# --------------------------------------------------------------------------- utilidades
def load_model(objetivo, modelo):
    norm = Norm.load(np.load(cfg.DATA / f"norm_{objetivo}.npz"))
    m = np.load(cfg.DATA / f"model_{modelo}_{objetivo}.npz")
    mlp = MLP(int(m["n_in"]), int(m["h1"]), int(m["h2"]), int(m["n_out"]))
    mlp.set_params(m["params"])
    if norm.mode != "shared":
        raise SystemExit("Este análisis asume NET_MODE='shared' (una MLP 6-16-16-1 por articulación).")
    return norm, mlp


def u_nn(mlp, norm, X):
    """Salida física de la red [rad/s] para características crudas X (N,6): acotada a ±y_scale (igual que el firmware)."""
    y = mlp.forward(norm.x(X)).reshape(-1)
    return np.clip(y, -1.0, 1.0) * norm.y_scale


class FalloSensor:
    """Envuelve un controlador e inyecta NaN en la medición de q durante [t0, t0+dur) (fallo de sensor/encoder)."""

    def __init__(self, ctrl, dt, t0=2.0, dur=0.1):
        self.ctrl, self.dt, self.t0, self.dur, self.k = ctrl, dt, t0, dur, 0
        self.name = ctrl.name

    def reset(self):
        self.k = 0
        self.ctrl.reset()

    def __call__(self, q, dq, qr, dqr):
        t = self.k * self.dt
        self.k += 1
        if self.t0 <= t < self.t0 + self.dur:
            q = np.full_like(np.asarray(q, float), np.nan)
        return self.ctrl(q, dq, qr, dqr)

    def __getattr__(self, a):
        return getattr(self.ctrl, a)


# --------------------------------------------------------------------------- A) offline
def parte_offline(a, norm, mlp, rp, kp_pid):
    rng = np.random.default_rng(123)
    res = {}
    u_max = float(rp.get("u_max", cfg.U_MAX_CFG))
    q_min = np.array(rp.get("q_min_safe", rp["q_min"]), float)
    q_max = np.array(rp.get("q_max_safe", rp["q_max"]), float)
    residual = a.objetivo == "residual"

    # ---- A1: entradas extremas, NaN e inf
    N = 200_000
    X = rng.uniform(-100, 100, (N, 6))
    X[:1000] = np.nan
    X[1000:2000] = np.inf
    X[2000:3000] = -np.inf
    with np.errstate(invalid="ignore"):
        y_raw = mlp.forward(norm.x(X)).reshape(-1) * norm.y_scale
        u = u_nn(mlp, norm, X)
    q_fake = np.zeros((N, 1))
    u_s = safety(q_fake, u.reshape(-1, 1), u_max, np.array([-9.0]), np.array([9.0]))
    res["A1"] = dict(
        muestras=N, nan_inf_inyectados=3000,
        salidas_nan_antes_de_seguridad=int(np.isnan(u).sum()),
        u_nn_max_abs_finito=float(np.nanmax(np.abs(u))),
        salida_sin_cota_max_abs=float(np.nanmax(np.abs(y_raw[np.isfinite(y_raw)]))),
        u_final_max_abs=float(np.max(np.abs(u_s))), u_final_todo_finito=bool(np.all(np.isfinite(u_s))),
        cota_residual_o_umax=float(norm.y_scale))

    # ---- A2: saturación en el dataset de prueba
    D = np.load(cfg.DATA / f"proc_{a.objetivo}.npz", allow_pickle=True)
    Xte, Yte = D["X_test"], D["Y_test"].reshape(-1)
    y_nn = np.clip(mlp.forward(Xte).reshape(-1), -1, 1)
    sat_ref = float(np.mean(np.abs(Yte) >= 0.999))
    sat_nn = float(np.mean(np.abs(y_nn) >= 0.999))
    res["A2"] = dict(frac_saturado_objetivo=sat_ref, frac_saturado_red=sat_nn,
                     max_abs_objetivo=float(np.max(np.abs(Yte)) * norm.y_scale),
                     max_abs_red=float(np.max(np.abs(y_nn)) * norm.y_scale))

    # ---- A3: ganancias efectivas locales (linealización del lazo) en los estados de prueba
    Xr = Xte * norm.std + norm.mean                      # de normalizado a crudo (la zona recortada es mínima)
    d = 1e-2

    def f(Z):
        return u_nn(mlp, norm, Z)

    def grad(i_state, i_err):
        Xp, Xm = Xr.copy(), Xr.copy()
        Xp[:, i_state] += d
        Xp[:, i_err] -= d                                # e = qr - q  baja cuando q sube
        Xm[:, i_state] -= d
        Xm[:, i_err] += d
        return -(f(Xp) - f(Xm)) / (2 * d)                # kp_ef = -du/dq   (kd_ef = -du/dq̇)

    kp_nn, kd_nn = grad(0, 4), grad(1, 5)
    kp_tot = kp_nn + (kp_pid if residual else 0.0)       # en residual la ganancia total es la del PID + la de la red
    dt = cfg.DT
    pct = lambda v: {k: float(np.percentile(v, p)) for k, p in (("p1", 1), ("p5", 5), ("p50", 50), ("p95", 95), ("p99", 99))}
    # Modelo discreto congelado por estado: q(k+1) = q + dt*g*u(k),  dq(k) = g*u(k-1)  (lazo de velocidad sin lag, como mide 01),
    #   u(k) = -kp_ef*q(k) - kd_ef*dq(k)   ->   x=[q, u(k-1)],  A = [[1-dt*g*kp, -dt*g^2*kd], [-kp, -kd*g]]
    # Si kd_ef < 0 la red se comporta como realimentación POSITIVA de velocidad (copia la velocidad medida ~ u anterior).
    kd_tot = kd_nn + (float(np.mean(cfg.load_pid()["kd"])) if residual else 0.0)
    g_max = max(j["gain"] for j in rp["joints"]) if "joints" in rp else 1.0
    insat = (np.abs(kp_nn) < 1e-6) & (np.abs(kd_nn) < 1e-6)          # salida saturada: gradiente nulo (u constante)

    def rho(g):
        r = np.empty(len(kp_tot))
        for i, (kp, kd) in enumerate(zip(kp_tot, kd_tot)):
            r[i] = np.max(np.abs(np.linalg.eigvals(np.array([[1 - dt * g * kp, -dt * g * g * kd], [-kp, -kd * g]]))))
        return r

    rho1, rhog = rho(1.0), rho(g_max)
    act = ~insat
    res["A3b"] = dict(
        g_max_medida=float(g_max), frac_saturado_sin_gradiente=float(np.mean(insat)),
        kd_ef_frac_negativo=float(np.mean(kd_tot[act] < -1e-6)), kd_ef_min=float(kd_tot.min()),
        radio_espectral_max_g1=float(rho1[act].max()), radio_espectral_max_gmax=float(rhog[act].max()),
        frac_rho_ge_1_g1=float(np.mean(rho1[act] >= 1.0)), frac_rho_ge_1_gmax=float(np.mean(rhog[act] >= 1.0)),
        frac_rho_ge_1_gmax_sobre_total=float(np.mean(act & (rhog >= 1.0))))
    res["A3"] = dict(
        kp_pid=kp_pid, kp_total_pct=pct(kp_tot), kd_nn_pct=pct(kd_nn),
        kp_total_min=float(kp_tot.min()), kp_total_max=float(kp_tot.max()),
        frac_kp_positivo=float(np.mean(kp_tot > 0)), frac_kp_dt_lt_2=float(np.mean(kp_tot * dt < 2)),
        frac_estados_con_kp_ef_cero=float(np.mean(np.abs(kp_nn) < 1e-6)),   # salida saturada -> gradiente nulo
        kp_dt_max=float(kp_tot.max() * dt))

    # ---- A4: capa de seguridad con entradas hostiles
    M = 200_000
    q = rng.uniform(q_min - 0.5, q_max + 0.5, (M, 7))
    uu = rng.uniform(-3 * u_max, 3 * u_max, (M, 7))
    uu[:500] = np.nan
    uu[500:1000] = np.inf
    us = safety(q, uu, u_max, q_min, q_max)
    mg = cfg.JOINT_MARGIN
    res["A4"] = dict(
        pruebas=M * 7,
        viola_umax=int(np.sum(np.abs(us) > u_max + 1e-12)),
        viola_limite_inferior=int(np.sum((q <= q_min + mg) & (us < 0))),
        viola_limite_superior=int(np.sum((q >= q_max - mg) & (us > 0))),
        no_finitos=int(np.sum(~np.isfinite(us))))

    # ---- A5: figura de la ley de control u(e) (fuera del dataset incluido)
    e = np.linspace(-8, 8, 801)
    q0 = 0.5
    X5 = build_features(np.full_like(e, q0), np.zeros_like(e), q0 + e, np.zeros_like(e), "shared")
    u5 = u_nn(mlp, norm, X5)
    u_ctrl = np.clip(kp_pid * e + u5, -u_max, u_max) if residual else u5
    return res, dict(e=e, u_nn=u5, u_total=u_ctrl, kp_tot=kp_tot, kd_tot=kd_tot, kp_pid=kp_pid, residual=residual,
                     u_max=u_max, g_max=g_max)


# --------------------------------------------------------------------------- B) lazo cerrado
def parte_lazo_cerrado(a, norm, mlp):
    from coppelia_robot import make_robot
    from scenarios import nominal_scenarios
    from sim_loop import Perturbation, run_episode, tracking_metrics

    robot = make_robot(a.mock)
    n, dt = robot.n, robot.dt
    rng = np.random.default_rng(777)
    pid = PID(dt, robot.u_max, n)
    nn = NNController(mlp, norm)
    if a.objetivo == "residual":
        ctrls = {"PID": pid, "PID+NN": ResidualController(PID(dt, robot.u_max, n), nn)}
    else:
        ctrls = {"PID": pid, "NN": nn}

    sc = {s["name"]: s for s in nominal_scenarios(robot)}["sine"]
    push = rng.choice([-1.0, 1.0], n) * 1.5
    escenarios = [
        ("nominal", Perturbation(), False),
        ("motor +50%", Perturbation(motor_gain=1.5, label="motor +50%"), False),
        ("motor -50%", Perturbation(motor_gain=0.5, label="motor -50%"), False),
        ("empujón 1.5 rad/s", Perturbation(pulse_t0=2.0, pulse_dur=0.4, pulse_amp=push, label="empujón"), False),
        ("ruido x10", Perturbation(noise_q=0.02, noise_dq=0.2, label="ruido"), False),
        ("fallo sensor (NaN 0.1 s)", Perturbation(label="fallo"), True),
    ]
    rows = []
    for sname, pert, fallo in escenarios:
        for cname, c in ctrls.items():
            cc = FalloSensor(c, dt) if fallo else c
            with np.errstate(invalid="ignore"):
                ep = run_episode(robot, cc, sc["traj"], sc["T"], sc["q0"], pert, seed=sc["seed"])
            m = tracking_metrics(ep, dt)
            u_cmd = np.nan_to_num(ep["u_cmd"], nan=0.0, posinf=0.0, neginf=0.0)
            with np.errstate(invalid="ignore"):
                u_safe = safety(np.nan_to_num(ep["qm"]), ep["u_cmd"], robot.u_max, robot.q_min, robot.q_max)
            interv = float(np.mean(np.abs(u_safe - u_cmd) > 1e-9))
            viol = float(np.max(np.maximum(np.maximum(robot.q_min - ep["q"], ep["q"] - robot.q_max), 0.0)))
            e = ep["qr"] - ep["q"]
            k2 = int(2.0 / dt)
            cruces = np.sum(np.diff(np.sign(e[k2:]), axis=0) != 0) / n / (len(e[k2:]) * dt)   # cruces por cero [1/s] por junta
            k_fin = len(e) - int(2.0 / dt)
            rows.append(dict(escenario=sname, controlador=cname, u_cmd_max=float(np.max(np.abs(u_cmd))),
                             pct_seguridad_intervino=100 * interv, viol_limite_rad=viol,
                             nan_en_u_cmd=int(np.isnan(ep["u_cmd"]).sum()), osc_du_rms=m["u_rough"],
                             cruces_cero_por_s=float(cruces), rmse_regimen=m["rmse_ss"],
                             rmse_ult_2s=float(np.sqrt(np.mean(e[k_fin:] ** 2))), e_max=m["e_max"]))
    robot.close()
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objetivo", choices=["directo", "residual"], default="directo")
    ap.add_argument("--modelo", choices=["pc", "micro"], default="pc")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--solo-offline", action="store_true", help="solo la parte A (sin robot)")
    a = ap.parse_args()

    rp = cfg.load_robot_params()
    if not rp:
        raise SystemExit("Falta data/robot_params.json (corre 01_caracterizar_robot.py).")
    norm, mlp = load_model(a.objetivo, a.modelo)
    kp_pid = float(np.mean(cfg.load_pid()["kp"]))
    tag = f"{a.objetivo}_{a.modelo}"

    res, fig = parte_offline(a, norm, mlp, rp, kp_pid)
    print("\n=== A) OFFLINE ===")
    for k, v in res.items():
        print(k, v)

    # figura
    f, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(fig["e"], fig["u_total"], label="PID+NN" if fig["residual"] else "NN")
    if not fig["residual"]:
        ax[0].plot(fig["e"], np.clip(fig["kp_pid"] * fig["e"], -fig["u_max"], fig["u_max"]), "--", label=f"kp·e (kp={fig['kp_pid']:g}, sat.)")
    ax[0].axvspan(-cfg.E_CLIP, cfg.E_CLIP, color="g", alpha=.08, label="rango del dataset")
    ax[0].axhline(fig["u_max"], color="r", lw=.7); ax[0].axhline(-fig["u_max"], color="r", lw=.7)
    ax[0].set_xlabel("e = q_r - q [rad]"); ax[0].set_ylabel("u [rad/s]"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[0].set_title("Ley de control aprendida (dq=0, dqr=0)")
    ax[1].hist(fig["kp_tot"], bins=60)
    ax[1].axvline(0, color="r"); ax[1].set_xlabel("ganancia efectiva kp_ef = -du/dq [1/s]"); ax[1].set_ylabel("estados de prueba")
    ax[1].set_title("Ganancia proporcional efectiva"); ax[1].grid(alpha=.3)
    ax[2].hist(fig["kd_tot"], bins=60)
    ax[2].axvline(0, color="r"); ax[2].axvline(-1.0 / fig["g_max"], color="k", ls="--", label="-1/g_max (límite de polo interno)")
    ax[2].set_xlabel("kd_ef = -du/ddq  (<0: realimentación positiva de velocidad)"); ax[2].set_title("Ganancia de velocidad efectiva")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(cfg.FIGS / f"09_seguridad_{tag}.png", dpi=140)

    out = dict(offline=res)
    if not a.solo_offline:
        df = parte_lazo_cerrado(a, norm, mlp)
        pd.set_option("display.width", 220)
        print("\n=== B) LAZO CERRADO ===")
        print(df.round(4).to_string(index=False))
        df.to_csv(cfg.DATA / f"seguridad_{tag}.csv", index=False)
        out["lazo_cerrado"] = df.to_dict("records")
    save_json(cfg.DATA / f"seguridad_{tag}.json", out)


if __name__ == "__main__":
    main()
