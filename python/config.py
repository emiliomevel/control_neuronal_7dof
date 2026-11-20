"""
Configuración global de la práctica (lado PC).
Todo lo que normalmente querrás cambiar está aquí; los scripts no tienen números "mágicos".
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGS = DATA / "figs"
FW_SRC = ROOT.parent / "firmware" / "src"       # aquí se escriben nn_config.h / nn_weights.h
for _d in (DATA, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# CoppeliaSim (ZeroMQ remote API)
# ----------------------------------------------------------------------------
ZMQ_HOST = "localhost"
ZMQ_PORT = 23000
ROBOT_ROOT = None        # alias del robot (p. ej. "/Franka"); None = busca en toda la escena
JOINT_PATHS = None       # lista explícita de 7 paths si la autodetección elige mal las juntas
N_JOINTS = 7
DT = 0.01                # periodo de control = paso de simulación [s]
JOINT_MAX_FORCE = 300.0  # tope de par del motor interno de cada junta [N·m]

# ----------------------------------------------------------------------------
# Colisiones (geométricas, con sim.checkCollision)
# ----------------------------------------------------------------------------
SELF_COLLISION_SKIP = 1        # se ignoran pares de eslabones a <= este número de juntas (adyacentes se tocan por diseño)
IGNORE_BASE_OBSTACLE = True    # la base apoyada en el piso/mesa no cuenta como colisión
IGNORE_BASELINE_SELF = True    # pares eslabón-eslabón que ya chocan en la pose inicial de la escena = solape de diseño
COLLISION_BACKOFF = 0.10       # [rad] margen que se retira antes del primer contacto
COLLISION_STEP = 0.02          # [rad] resolución del barrido por junta
PATH_CHECK_DT = 0.15           # [s] cada cuánto se verifica una trayectoria de referencia

# ----------------------------------------------------------------------------
# Señal de control y seguridad
#   u = velocidad articular comandada [rad/s]  (el motor interno de Coppelia la sigue)
# ----------------------------------------------------------------------------
U_MAX_CFG = 1.5          # |u| máximo permitido [rad/s]  (01_caracterizar_robot lo puede reducir)
JOINT_MARGIN = 0.03      # distancia al límite articular donde se bloquea el movimiento hacia afuera [rad]
RESIDUAL_MAX = 0.4       # autoridad máxima de la red residual [rad/s]
PAYLOAD_MAX_KG = 2.0     # carga extra máxima en el efector final usada en perturbaciones

# ----------------------------------------------------------------------------
# PID de referencia (ganancias por articulación; escalares o vectores de 7)
#   Si existe data/pid_gains.json (lo escribe 02_evaluar_pid.py --tune) se usa ese.
# ----------------------------------------------------------------------------
PID_DEFAULT = dict(kp=4.0, ki=0.8, kd=0.15, i_lim=0.5)   # i_lim: tope de |ki*∫e| [rad/s]


def load_pid():
    p = dict(PID_DEFAULT)
    f = DATA / "pid_gains.json"
    if f.exists():
        p.update(json.loads(f.read_text()))
    return p


# ----------------------------------------------------------------------------
# Parámetros del robot (los escribe 01_caracterizar_robot.py)
# ----------------------------------------------------------------------------
MOCK_Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
MOCK_Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def load_robot_params():
    f = DATA / "robot_params.json"
    return json.loads(f.read_text()) if f.exists() else None


# ----------------------------------------------------------------------------
# Red neuronal
#   "shared": una MLP 6->H1->H2->1 que se evalúa en cada articulación (misma red, 7 veces)
#   "full"  : una MLP (6*n)->H1->H2->n con el estado de todas las articulaciones
# ----------------------------------------------------------------------------
NET_MODE = "shared"
H1 = 16
H2 = 16
N_FEAT = 6               # [q, dq, qr, dqr, e, de]
X_CLIP = 5.0             # saturación de las entradas normalizadas (±)

# ----------------------------------------------------------------------------
# Entrenamiento (idéntico en PC y micro para poder compararlos)
# ----------------------------------------------------------------------------
OPTIMIZER = "adam"       # "adam" | "sgd" (SGD con momentum). Debe coincidir con NN_USE_ADAM del firmware
LR = 3e-3
LR_MIN_FRAC = 0.05       # el lr decae (coseno) hasta LR*LR_MIN_FRAC
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
SGD_MU = 0.9
BATCH = 1               # muestras por trama/actualización (1 = SGD puro muestra por muestra)
EPOCHS_PC = 40
EPOCHS_MICRO = 10
SEED = 1

# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
N_EPISODES = 80
EP_T = 8.0               # duración de cada episodio [s]
DECIM = 5                # se registra 1 de cada DECIM pasos (20 Hz con DT=0.01)
KINDS = ["step", "sine", "triangle", "random"]
KFF = 1.0                # ganancia de la acción "maestra" del objetivo residual (PID + KFF*dqr)
TEST_FRAC = 0.15         # del total de episodios (no se usa en entrenamiento)
VAL_FRAC = 0.15          # del total, tomado de dentro del 85 % restante
E_CLIP = 3.0             # saturación física del error [rad]
Z_OUT = 10.0             # umbral z-score para descartar muestras anómalas

# ----------------------------------------------------------------------------
# Enlace con el micro
# ----------------------------------------------------------------------------
MICRO_URL = "COM10@921600"   # "COMx@baud", "/dev/ttyUSBx@baud" o "exec:../firmware/host_sim/micro_sim"
