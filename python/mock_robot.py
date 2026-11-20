"""
Interfaz común de robot + robot simulado en Python puro (MockRobot).

MockRobot NO sustituye a CoppeliaSim: sirve para depurar todo el pipeline
(dataset -> entrenamiento -> comparación -> micro) sin abrir el simulador.
Modela cada articulación como un lazo de velocidad de 1er orden con aceleración limitada
más un sesgo tipo gravedad que depende de q y de la carga (por eso el PID necesita el término I).
"""
import numpy as np

import config as cfg


class RobotBase:
    """Contrato que usan todos los scripts (CoppeliaRobot y MockRobot lo cumplen)."""
    n = cfg.N_JOINTS
    dt = cfg.DT
    names = None
    q_min = None
    q_max = None
    u_max = None          # vector [rad/s]
    info = {}

    q_min_hw = None
    q_max_hw = None

    def apply_safe_limits(self):
        """Guarda los límites de hardware y aplica el rango seguro de data/robot_params.json
        (lo escribe 01b_detectar_colisiones.py). Todos los scripts usan robot.q_min/q_max."""
        self.q_min_hw, self.q_max_hw = self.q_min.copy(), self.q_max.copy()
        p = cfg.load_robot_params()
        if not p or "q_min_safe" not in p:
            return
        lo = np.maximum(np.array(p["q_min_safe"], float), self.q_min_hw)
        hi = np.minimum(np.array(p["q_max_safe"], float), self.q_max_hw)
        c, h = 0.5 * (lo + hi), 0.5 * (hi - lo) * float(p.get("range_frac", 1.0))
        self.q_min, self.q_max = c - h, c + h

    def set_limits(self, lo, hi):
        self.q_min, self.q_max = np.array(lo, float), np.array(hi, float)

    def contacts(self):
        return []

    # ---- colisiones geométricas (MockRobot: paredes simuladas; CoppeliaRobot: sim.checkCollision)
    collision_ok = True
    collision_info = {}

    def collisions(self, q, all_pairs=False):
        """Lista de pares (a, b) en colisión para la configuración q. Vacía = libre."""
        return []

    def config_free(self, q):
        return not self.collisions(q)

    def path_free(self, q0, traj, T, dt_check=cfg.PATH_CHECK_DT):
        """Verifica el segmento q0 -> qr(0) y la trayectoria de referencia muestreada."""
        q0 = np.asarray(q0, float)
        qr0 = traj.eval(0.0)[0]
        pts = [q0 + (qr0 - q0) * a for a in np.linspace(0, 1, 10)]
        pts += [traj.eval(t)[0] for t in np.arange(0, T, dt_check)]
        for q in pts:
            c = self.collisions(q)
            if c:
                return False, c
        return True, []

    def start(self): ...
    def reset(self, q0, payload_kg=0.0): raise NotImplementedError
    def read(self): raise NotImplementedError          # -> (q, dq) copias
    def apply(self, u): raise NotImplementedError      # u: velocidad comandada [rad/s]
    def step(self): raise NotImplementedError
    def close(self): ...


class MockRobot(RobotBase):
    def __init__(self, seed=0, v_max=2.0, a_max=15.0):
        rng = np.random.default_rng(seed)
        self.names = [f"mock_joint{i + 1}" for i in range(self.n)]
        self.q_min = cfg.MOCK_Q_MIN.copy()
        self.q_max = cfg.MOCK_Q_MAX.copy()
        self.tau0 = np.linspace(0.05, 0.10, self.n)            # constante de tiempo del lazo de velocidad
        self.phase = rng.uniform(0, 2 * np.pi, self.n)
        self.v_max, self.a_max = v_max, a_max
        self.q = np.zeros(self.n)
        self.dq = np.zeros(self.n)
        self.u_cmd = np.zeros(self.n)
        self.payload = 0.0
        self.t = 0.0
        p = cfg.load_robot_params()
        umax = p.get("u_max", cfg.U_MAX_CFG) if p else cfg.U_MAX_CFG
        self.u_max = np.full(self.n, float(umax))
        self.wall_lo = np.full(self.n, -np.inf)     # "obstáculos" de prueba (simulan colisiones)
        self.wall_hi = np.full(self.n, np.inf)
        self.info = dict(kind="mock")
        self.apply_safe_limits()

    def reset(self, q0, payload_kg=0.0):
        self.q = np.array(q0, dtype=float).copy()
        self.dq = np.zeros(self.n)
        self.u_cmd = np.zeros(self.n)
        self.payload = float(payload_kg)
        self.t = 0.0

    def read(self):
        return self.q.copy(), self.dq.copy()

    def apply(self, u):
        self.u_cmd = np.clip(np.asarray(u, float), -3.0, 3.0)   # el "driver" acepta hasta 3 rad/s

    def step(self):
        sub = 5
        h = self.dt / sub
        tau = self.tau0 * (1.0 + 0.25 * self.payload)
        for _ in range(sub):
            bias = (0.12 + 0.06 * self.payload) * np.sin(self.q + self.phase)   # "gravedad" [rad/s]
            target = np.clip(self.u_cmd - bias, -self.v_max, self.v_max)
            ddq = np.clip((target - self.dq) / tau, -self.a_max, self.a_max)
            self.dq = self.dq + ddq * h
            self.q = self.q + self.dq * h
            lo = (self.q < self.q_min_hw) | (self.q < self.wall_lo)
            hi = (self.q > self.q_max_hw) | (self.q > self.wall_hi)
            lim_lo = np.maximum(self.q_min_hw, self.wall_lo)
            lim_hi = np.minimum(self.q_max_hw, self.wall_hi)
            self.q = np.where(lo, lim_lo, np.where(hi, lim_hi, self.q))
            self.dq = np.where(lo & (self.dq < 0), 0.0, np.where(hi & (self.dq > 0), 0.0, self.dq))
        self.t += self.dt

    def collisions(self, q, all_pairs=False):
        q = np.asarray(q, float)
        out = [(f"link{j + 1}", "pared+") for j in np.where(q > self.wall_hi)[0]]
        out += [(f"link{j + 1}", "pared-") for j in np.where(q < self.wall_lo)[0]]
        return out

    def mass_info(self):
        return dict(tip_link_mass_kg=None, note="mock")
