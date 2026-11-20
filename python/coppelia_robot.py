"""
Interfaz con CoppeliaSim (ZeroMQ remote API) para el manipulador de 7 GDL.

- Detecta las juntas de revolución del robot (o usa cfg.JOINT_PATHS si las das a mano).
- Control por VELOCIDAD articular: el motor interno de Coppelia sigue u [rad/s].
- Modo "stepping": el simulador avanza solo cuando llamas a step() -> resultados repetibles.
- reset(): detiene y reinicia la simulación en cada episodio (estado dinámico limpio).

Requisitos del modelo en la escena (ver README): juntas dinámicas, base fija, sin colisiones propias.
"""
import time

import numpy as np

import config as cfg
from mock_robot import MockRobot, RobotBase


def _try(fn, *args):
    try:
        fn(*args)
        return True
    except Exception:
        return False


class CoppeliaRobot(RobotBase):
    def __init__(self, host=cfg.ZMQ_HOST, port=cfg.ZMQ_PORT):
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient   # pip install coppeliasim-zmqremoteapi-client

        self.client = RemoteAPIClient(host, port)
        try:
            self.sim = self.client.require("sim")
        except Exception:
            self.sim = self.client.getObject("sim")
        sim = self.sim
        if sim.getSimulationState() != sim.simulation_stopped:
            sim.stopSimulation()
            while sim.getSimulationState() != sim.simulation_stopped:
                time.sleep(0.01)
        _try(sim.setFloatParam, sim.floatparam_simulation_time_step, cfg.DT)
        self.client.setStepping(True)
        self._running = False
        self._find_joints()
        self._read_limits()
        p = cfg.load_robot_params()
        umax = p.get("u_max", cfg.U_MAX_CFG) if p else cfg.U_MAX_CFG
        self.u_max = np.full(self.n, float(umax))
        self._find_tip_shape()
        self._find_robot_shapes()
        self._setup_collision()
        self.apply_safe_limits()
        self.info = dict(kind="coppelia", joints=self.names, kinematic_fallback=self._kinematic)

    # ------------------------------------------------------------------ escena
    def _depth(self, h):
        d, p = 0, self.sim.getObjectParent(h)
        while p != -1:
            d += 1
            p = self.sim.getObjectParent(p)
        return d

    def _find_joints(self):
        sim = self.sim
        if cfg.JOINT_PATHS:
            hs = [sim.getObject(p) for p in cfg.JOINT_PATHS]
        else:
            root = sim.getObject(cfg.ROBOT_ROOT) if cfg.ROBOT_ROOT else sim.handle_scene
            allj = sim.getObjectsInTree(root, sim.object_joint_type, 0)
            rev = [h for h in allj if sim.getJointType(h) == sim.joint_revolute_subtype]
            rev = sorted(rev, key=self._depth)           # orden cadena base -> efector
            hs = rev[: cfg.N_JOINTS]
        if len(hs) != cfg.N_JOINTS:
            raise RuntimeError(f"Se esperaban {cfg.N_JOINTS} juntas y se encontraron {len(hs)}. "
                               "Define cfg.ROBOT_ROOT o cfg.JOINT_PATHS.")
        self.joints = hs
        self.n = len(hs)
        names = []
        for h in hs:
            try:
                names.append(self.sim.getObjectAlias(h, 2))
            except Exception:
                names.append(str(h))
        self.names = names
        # ¿las juntas son dinámicas? si no, se usa integración cinemática (ver apply/step)
        self._kinematic = False
        for h in hs:
            try:
                mode = sim.getJointMode(h)
                mode = mode[0] if isinstance(mode, (list, tuple)) else mode
                if mode != sim.jointmode_dynamic:
                    _try(sim.setJointMode, h, sim.jointmode_dynamic)
                    mode = sim.getJointMode(h)
                    mode = mode[0] if isinstance(mode, (list, tuple)) else mode
                if mode != sim.jointmode_dynamic:
                    self._kinematic = True
            except Exception:
                pass

    def _read_limits(self):
        qmin, qmax = [], []
        for h in self.joints:
            cyc, itv = self.sim.getJointInterval(h)
            if cyc:
                qmin.append(-np.pi)
                qmax.append(np.pi)
            else:
                qmin.append(itv[0])
                qmax.append(itv[0] + itv[1])
        self.q_min, self.q_max = np.array(qmin), np.array(qmax)

    def _find_tip_shape(self):
        sim = self.sim
        self.tip_shape, self.tip_mass0 = None, None
        try:
            shapes = sim.getObjectsInTree(self.joints[-1], sim.object_shape_type, 0)
            direct = [s for s in shapes if sim.getObjectParent(s) == self.joints[-1]]
            self.tip_shape = (direct or shapes)[0]
            self.tip_mass0 = sim.getShapeMass(self.tip_shape)
        except Exception:
            pass

    def _find_robot_shapes(self):
        """Shapes del robot completo (sube desde la junta 1 hasta el objeto raíz) para nombrar contactos."""
        sim = self.sim
        top = self.joints[0]
        while sim.getObjectParent(top) != -1:
            top = sim.getObjectParent(top)
        try:
            self._robot_shapes = set(sim.getObjectsInTree(top, sim.object_shape_type, 0))
        except Exception:
            self._robot_shapes = set()

    # ------------------------------------------------------------------ colisiones geométricas
    def _alias(self, h):
        try:
            return str(self.sim.getObjectAlias(h)).split("/")[-1]
        except Exception:
            return str(h)

    def _link_index(self, h):
        idx = {jh: i for i, jh in enumerate(self.joints)}
        p = h
        while p != -1:
            if p in idx:
                return idx[p] + 1
            p = self.sim.getObjectParent(p)
        return 0                                   # base (no cuelga de ninguna junta)

    def _set_collidable(self, h):
        sim = self.sim
        for get, setp, const in ((sim.getObjectSpecialProperty, sim.setObjectSpecialProperty, "objectspecialproperty_collidable"),
                                 (sim.getObjectProperty, sim.setObjectProperty, "objectproperty_collidable")):
            try:
                setp(h, get(h) | getattr(sim, const))
                return True
            except Exception:
                continue
        return False

    def _coll(self, a, b):
        r = self.sim.checkCollision(a, b)
        r = r[0] if isinstance(r, (list, tuple)) else r
        return bool(r)

    def _self_pairs(self):
        ks = sorted(self._link_coll)
        return [(a, b) for a in ks for b in ks if b - a > cfg.SELF_COLLISION_SKIP]

    def _setup_collision(self):
        """Una colección por eslabón + una colección con TODO el entorno (piso, mesa, objetos)."""
        sim = self.sim
        self.collision_ok, self.collision_info = False, {}
        try:
            shapes = sorted(self._robot_shapes)
            if not shapes:
                raise RuntimeError("no se encontraron shapes del robot")
            by_link = {}
            for s in shapes:
                by_link.setdefault(self._link_index(s), []).append(s)
            obst = [s for s in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type, 0) if s not in self._robot_shapes]
            self._obst_shapes = obst
            fail = sum(0 if self._set_collidable(h) else 1 for h in shapes + obst)
            self._link_coll, self._link_name = {}, {}
            for li, ss in by_link.items():
                c = sim.createCollection(0)
                for s in ss:
                    sim.addItemToCollection(c, sim.handle_single, s, 0)
                self._link_coll[li] = c
                self._link_name[li] = "base" if li == 0 else f"link{li}({self._alias(ss[0])})"
            self._obst_coll = None
            if obst:
                self._obst_coll = sim.createCollection(0)
                for s in obst:
                    sim.addItemToCollection(self._obst_coll, sim.handle_single, s, 0)
            self._q_scene0 = np.array([sim.getJointPosition(h) for h in self.joints])
            self._ignored = set()
            if cfg.IGNORE_BASELINE_SELF:                   # solapes de diseño en la pose de la escena
                for a, b in self._self_pairs():
                    if self._coll(self._link_coll[a], self._link_coll[b]):
                        self._ignored.add((a, b))
            self.collision_ok = True
            self.collision_info = dict(
                eslabones=len(by_link), shapes_robot=len(shapes), shapes_entorno=len(obst), sin_flag_collidable=fail,
                entorno=[self._alias(s) for s in obst][:12],
                pares_ignorados=[(self._link_name[a], self._link_name[b]) for a, b in sorted(self._ignored)])
        except Exception as ex:
            self.collision_info = dict(error=str(ex))
            print(f"AVISO: no se pudo preparar la detección de colisiones ({ex}). Se seguirá sin validar colisiones.")

    def ensure_stopped(self):
        if self._running:
            self.sim.stopSimulation()
            while self.sim.getSimulationState() != self.sim.simulation_stopped:
                time.sleep(0.01)
            self._running = False

    def _obst_name(self, c):
        for s in self._obst_shapes:
            if self._coll(c, s):
                return self._alias(s)
        return "entorno"

    def collisions(self, q, all_pairs=False):
        """Pares en colisión para la configuración q (colocada con la simulación detenida)."""
        if not self.collision_ok:
            return []
        sim = self.sim
        self.ensure_stopped()
        for h, qq in zip(self.joints, q):
            sim.setJointPosition(h, float(qq))
        out = []
        if self._obst_coll is not None:
            for li, c in self._link_coll.items():
                if li == 0 and cfg.IGNORE_BASE_OBSTACLE:
                    continue
                if self._coll(c, self._obst_coll):
                    out.append((self._link_name[li], self._obst_name(c)))
                    if not all_pairs:
                        return out
        for a, b in self._self_pairs():
            if (a, b) in self._ignored:
                continue
            if self._coll(self._link_coll[a], self._link_coll[b]):
                out.append((self._link_name[a], self._link_name[b]))
                if not all_pairs:
                    return out
        return out

    def contacts(self):
        """Pares de shapes en contacto dinámico que involucran al robot (mejor esfuerzo; depende de la versión de Coppelia)."""
        sim, out = self.sim, []
        try:
            for i in range(300):
                r = sim.getContactInfo(sim.handle_all, sim.handle_all, i)
                pair = r[0] if isinstance(r, (list, tuple)) and r and isinstance(r[0], (list, tuple)) else r
                if not pair or len(pair) < 2 or pair[0] in (-1, None):
                    break
                a, b = pair[0], pair[1]
                if a in self._robot_shapes or b in self._robot_shapes:
                    out.append((sim.getObjectAlias(a, 2), sim.getObjectAlias(b, 2)))
        except Exception:
            pass
        return out

    def _configure_joints(self):
        sim = self.sim
        for h in self.joints:
            _try(sim.setObjectInt32Param, h, sim.jointintparam_dynctrlmode, sim.jointdynctrl_velocity)
            if not _try(sim.setJointMaxForce, h, cfg.JOINT_MAX_FORCE):
                _try(sim.setJointTargetForce, h, cfg.JOINT_MAX_FORCE)

    # ------------------------------------------------------------------ API
    def start(self):
        if not self._running:
            self.reset(np.clip(np.zeros(self.n), self.q_min, self.q_max))

    def reset(self, q0, payload_kg=0.0):
        sim = self.sim
        if self._running:
            sim.stopSimulation()
            while sim.getSimulationState() != sim.simulation_stopped:
                time.sleep(0.01)
        self.client.setStepping(True)
        sim.startSimulation()
        self._running = True
        self._configure_joints()
        self._q_kin = np.array(q0, float).copy()
        for h, qq in zip(self.joints, q0):
            sim.setJointPosition(h, float(qq))
            _try(sim.setJointTargetVelocity, h, 0.0)
        self._u = np.zeros(self.n)
        self.set_payload(payload_kg)

    def set_payload(self, kg):
        if self.tip_shape is not None:
            _try(self.sim.setShapeMass, self.tip_shape, float(self.tip_mass0 + kg))

    def read(self):
        sim = self.sim
        if self._kinematic:
            return self._q_kin.copy(), self._u.copy()
        q = np.array([sim.getJointPosition(h) for h in self.joints])
        dq = np.array([sim.getJointVelocity(h) for h in self.joints])
        return q, dq

    def apply(self, u):
        self._u = np.asarray(u, float).copy()
        if not self._kinematic:
            for h, uu in zip(self.joints, self._u):
                self.sim.setJointTargetVelocity(h, float(uu))

    def step(self):
        if self._kinematic:
            self._q_kin = np.clip(self._q_kin + self._u * self.dt, self.q_min, self.q_max)
            for h, qq in zip(self.joints, self._q_kin):
                self.sim.setJointPosition(h, float(qq))
        self.client.step()

    def mass_info(self):
        return dict(tip_link_mass_kg=self.tip_mass0)

    def close(self):
        self.ensure_stopped()
        if self.collision_ok:
            try:
                for h, qq in zip(self.joints, self._q_scene0):
                    self.sim.setJointPosition(h, float(qq))
                for c in list(self._link_coll.values()) + ([self._obst_coll] if self._obst_coll else []):
                    try:
                        self.sim.destroyCollection(c, 0)
                    except Exception:
                        self.sim.destroyCollection(c)
            except Exception:
                pass


def make_robot(mock=False, seed=0):
    """Devuelve el robot de Coppelia o el MockRobot (para depurar sin simulador)."""
    r = MockRobot(seed=seed) if mock else CoppeliaRobot()
    r.start()
    return r
