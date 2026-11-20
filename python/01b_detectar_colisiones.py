"""
Sesión 1 · Paso 2b: detectar COLISIONES (piso, entorno y entre eslabones) y calcular un rango seguro.

Usa la detección geométrica de CoppeliaSim (sim.checkCollision) sobre colecciones de shapes:
una colección por eslabón + una con todo el entorno (todo shape de la escena que no es del robot).
Se prueban configuraciones con la simulación detenida, así que no depende de la dinámica ni de que
la junta "se atore". Los pares eslabón-eslabón que ya se solapan en la pose inicial de la escena se
consideran solapes de diseño y se ignoran (se listan); la base contra el piso también se ignora.

Fases:
  0. Resumen: qué eslabones/entorno se detectaron y cuántos shapes no tienen la propiedad "collidable".
  1. Pose home: debe estar libre; si no, se busca una pose libre y se avisa.
  2. Barrido por junta desde home en pasos de COLLISION_STEP: primer contacto -> límite seguro por junta,
     indicando QUÉ choca con QUÉ (p. ej. link2(...) <-> ResizableFloor).
  3. Muestreo de trayectorias: se reduce el rango (range_frac) hasta que >=50 % de las trayectorias candidatas
     sean libres de colisión; además se corre el PID y se verifica que ninguna junta se atore.
Escribe en data/robot_params.json: q_min_safe, q_max_safe, range_frac, collision_report.
03_generar_dataset.py, 02 y 06 solo usan poses/trayectorias verificadas libres de colisión.

Uso:   python 01b_detectar_colisiones.py [--mock] [--n 8] [--home 0,0,0,0,0,0,0]
"""
import argparse
from collections import Counter

import numpy as np

import config as cfg
from controllers import PID, safety
from coppelia_robot import make_robot
from nn_data import save_json
from trajectories import random_pose, sample_free

STALL_STEPS = 30


def fmt(pairs):
    return ", ".join(sorted({f"{a} <-> {b}" for a, b in pairs}))


def sweep_joint(robot, home, j, sgn):
    """Mueve la junta j desde home hasta su límite de hardware; devuelve (q_límite_libre, pares en el primer contacto|None)."""
    hw = robot.q_max_hw[j] if sgn > 0 else robot.q_min_hw[j]
    q = home.copy()
    last_free = home[j]
    for val in np.arange(home[j], hw + sgn * 1e-9, sgn * cfg.COLLISION_STEP):
        q[j] = val
        c = robot.collisions(q, all_pairs=True)
        if c:
            return last_free, c
        last_free = val
    return hw, None


def accept_rate(robot, n_per_kind, rng, names):
    """Fracción de (pose inicial + trayectoria) candidatas que son libres de colisión con los límites actuales."""
    v_ref = 0.6 * float(robot.u_max[0])
    from trajectories import make_trajectory
    ok = total = 0
    for kind in cfg.KINDS:
        for _ in range(n_per_kind):
            total += 1
            q0 = random_pose(rng, robot.q_min, robot.q_max)
            c = robot.collisions(q0)
            if not c:
                traj = make_trajectory(kind, rng, robot.q_min, robot.q_max, v_ref, cfg.EP_T, q0)
                good, c = robot.path_free(q0, traj, cfg.EP_T)
            else:
                good = False
            if good:
                ok += 1
            else:
                names.update(f"{a} <-> {b}" for a, b in c)
    return ok / max(total, 1)


def stall_test(robot, n_per_kind, seed=4242):
    """Con trayectorias ya verificadas libres, corre el PID y detecta juntas que reciben comando y no se mueven."""
    pid = PID(robot.dt, robot.u_max, robot.n)
    rng = np.random.default_rng(seed)
    v_ref = 0.6 * float(robot.u_max[0])
    bad, total = [], 0
    for kind in cfg.KINDS:
        for _ in range(n_per_kind):
            q0, traj, free, _ = sample_free(robot, kind, rng, v_ref, cfg.EP_T)
            if not free:
                continue
            total += 1
            robot.reset(q0)
            pid.reset()
            cnt = np.zeros(robot.n, int)
            for k in range(int(cfg.EP_T / robot.dt)):
                q, dq = robot.read()
                qr, dqr = traj.eval(k * robot.dt)
                u = safety(q, pid(q, dq, qr, dqr), robot.u_max, robot.q_min, robot.q_max)
                robot.apply(u)
                robot.step()
                blocked = ((np.abs(u) > 0.3) & (np.abs(dq) < 0.05) & (np.abs(qr - q) > 0.05)
                           & (q > robot.q_min_hw + 0.05) & (q < robot.q_max_hw - 0.05))
                cnt = np.where(blocked, cnt + 1, 0)
                if (cnt >= STALL_STEPS).any():
                    bad.append(dict(kind=kind, joint=int(np.argmax(cnt >= STALL_STEPS)) + 1, t=k * robot.dt,
                                    contactos=robot.contacts()))
                    break
    return bad, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-wall", action="store_true", help="(solo prueba) obstáculos simulados en el robot simulado")
    ap.add_argument("--n", type=int, default=8, help="candidatos por tipo de trayectoria en la fase 3")
    ap.add_argument("--home", default=None, help="pose home separada por comas (por defecto todo 0)")
    a = ap.parse_args()

    robot = make_robot(a.mock)
    n = robot.n
    robot.set_limits(robot.q_min_hw, robot.q_max_hw)               # partir SIEMPRE de los límites del modelo
    if a.mock_wall:
        robot.wall_hi[1], robot.wall_lo[3], robot.wall_hi[5] = 0.8, -2.0, 2.2
    rng = np.random.default_rng(7)

    # ---------------- 0) resumen
    print("FASE 0 · detección de colisiones")
    info = robot.collision_info
    if not robot.collision_ok or "error" in info:
        raise SystemExit(f"No se pudo preparar sim.checkCollision: {info}. Revisa que el robot tenga shapes.")
    for k, v in info.items():
        print(f"  {k}: {v}")
    if info.get("sin_flag_collidable"):
        print("  AVISO: hay shapes sin propiedad 'Collidable' y no se pudo activar por API. En Coppelia: selecciona el "
              "robot y el piso -> Object properties -> Common -> marca 'Collidable'.")
    if info.get("shapes_entorno", 1) == 0:
        print("  AVISO: no hay shapes de entorno (¿no hay piso en la escena?).")

    # ---------------- 1) home
    home = np.array([float(x) for x in a.home.split(",")]) if a.home else np.zeros(n)
    home = np.clip(home, robot.q_min_hw + 0.05, robot.q_max_hw - 0.05)
    c = robot.collisions(home, all_pairs=True)
    if c:
        print(f"\nFASE 1 · la pose home {np.round(home, 2)} YA colisiona: {fmt(c)}")
        for _ in range(500):
            cand = random_pose(rng, robot.q_min_hw, robot.q_max_hw, 0.25)
            if not robot.collisions(cand):
                home = cand
                break
        else:
            raise SystemExit("No encontré ninguna pose libre. Pásame una con --home q1,...,q7 (una pose despejada del robot).")
        print(f"  usando home libre: {np.round(home, 2)}  (mejor: pásala tú con --home)")
    else:
        print(f"\nFASE 1 · home {np.round(home, 2)} libre de colisión")

    # ---------------- 2) barrido por junta
    lo, hi = robot.q_min_hw.copy(), robot.q_max_hw.copy()
    names, sweep_rep = Counter(), []
    print("\nFASE 2 · barrido por junta (las demás en home)")
    print(f"{'junta':6s}{'lado':>5s}{'q límite libre':>16s}   primer contacto")
    for j in range(n):
        for sgn, side in ((-1, "min"), (+1, "max")):
            q_free, pairs = sweep_joint(robot, home, j, sgn)
            if pairs:
                names.update(f"{x} <-> {y}" for x, y in pairs)
                if sgn > 0:
                    hi[j] = q_free - cfg.COLLISION_BACKOFF
                else:
                    lo[j] = q_free + cfg.COLLISION_BACKOFF
            print(f"J{j + 1:<5d}{side:>5s}{q_free:>16.3f}   {fmt(pairs) if pairs else '(sin contacto hasta el límite)'}")
            sweep_rep.append(dict(joint=j + 1, side=side, q_free=float(q_free), pares=[f"{x}<->{y}" for x, y in (pairs or [])]))
    hi = np.maximum(hi, lo + 0.2)
    print("\nRango por junta (hardware -> seguro):")
    for j in range(n):
        print(f"  J{j + 1}: [{robot.q_min_hw[j]:+.2f}, {robot.q_max_hw[j]:+.2f}] -> [{lo[j]:+.2f}, {hi[j]:+.2f}]")

    # ---------------- 3) reducir rango hasta tener trayectorias libres
    print("\nFASE 3 · trayectorias candidatas (poses iniciales + referencias) y prueba dinámica con PID")
    ctr, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    chosen, history = None, []
    for frac in (1.0, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2):
        robot.set_limits(ctr - frac * half, ctr + frac * half)
        nm = Counter()
        rate = accept_rate(robot, a.n, rng, nm)
        row = dict(range_frac=frac, aceptadas=rate)
        print(f"  range_frac={frac:.2f}: {100 * rate:5.1f} % de candidatas libres de colisión", end="")
        if rate < 0.5:
            names.update(nm)
            top = nm.most_common(2)
            print(f"   (choques típicos: {'; '.join(k for k, _ in top)})" if top else "")
            history.append(row)
            continue
        bad, total = stall_test(robot, max(2, a.n // 2))
        row.update(episodios_dinamicos=total, con_bloqueo=len(bad))
        history.append(row)
        print(f"  | PID: {len(bad)}/{total} episodios con junta atorada")
        for b in bad[:3]:
            print(f"       · {b['kind']} J{b['joint']} a t={b['t']:.2f}s  {fmt(b['contactos'])}")
        if not bad:
            chosen = frac
            break
    if chosen is None:
        chosen = 0.2
        print("  AVISO: no se logró un rango sin choques; revisa --home, la escena y que los shapes sean 'Collidable'.")

    out = cfg.load_robot_params() or {}
    out.update(q_min_safe=lo, q_max_safe=hi, range_frac=chosen,
               collision_report=dict(deteccion=info, barrido=sweep_rep, fase3=history, choques_frecuentes=names.most_common(10)))
    save_json(cfg.DATA / "robot_params.json", out)
    print(f"\nrange_frac = {chosen}. Guardado en data/robot_params.json")
    if names:
        print("Choques más frecuentes (¿son los que ves en la simulación?):")
        for k, v in names.most_common(6):
            print(f"  {k}  x{v}")
    print("Ahora corre 02_evaluar_pid.py y 03_generar_dataset.py: solo usarán poses/trayectorias libres de colisión.")
    robot.close()


if __name__ == "__main__":
    main()
