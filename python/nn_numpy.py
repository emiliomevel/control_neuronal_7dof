"""
MLP en NumPy que replica EXACTAMENTE lo que hace el firmware (nn_mlp.c):
  - mismo orden/disposición de parámetros (vector plano),
  - misma inicialización (xorshift32 + He uniforme),
  - misma pérdida, mismo backprop y mismo optimizador (Adam o SGD+momentum),
  - gradiente acumulado por trama y UNA actualización por trama.
Así lo que validas en PC es lo mismo que luego corre en el micro.

Disposición del vector de parámetros (float32):
  [ W1 (H1*IN, fila=neurona) | b1 (H1) | W2 (H2*H1) | b2 (H2) | W3 (OUT*H2) | b3 (OUT) ]
"""
import numpy as np

M32 = 0xFFFFFFFF


def xorshift32(s):
    s ^= (s << 13) & M32
    s ^= s >> 17
    s ^= (s << 5) & M32
    return s & M32


class MLP:
    def __init__(self, n_in, h1, h2, n_out, dtype=np.float32):
        self.n_in, self.h1, self.h2, self.n_out = n_in, h1, h2, n_out
        self.dtype = dtype
        self.o_w1 = 0
        self.o_b1 = self.o_w1 + h1 * n_in
        self.o_w2 = self.o_b1 + h1
        self.o_b2 = self.o_w2 + h2 * h1
        self.o_w3 = self.o_b2 + h2
        self.o_b3 = self.o_w3 + n_out * h2
        self.n_params = self.o_b3 + n_out
        self.p = np.zeros(self.n_params, dtype)
        self._bind()

    def _bind(self):
        p, h1, h2, n_in, n_out = self.p, self.h1, self.h2, self.n_in, self.n_out
        self.W1 = p[self.o_w1:self.o_b1].reshape(h1, n_in)
        self.b1 = p[self.o_b1:self.o_w2]
        self.W2 = p[self.o_w2:self.o_b2].reshape(h2, h1)
        self.b2 = p[self.o_b2:self.o_w3]
        self.W3 = p[self.o_w3:self.o_b3].reshape(n_out, h2)
        self.b3 = p[self.o_b3:self.n_params]

    # ---------------------------------------------------------------- init
    def randomize(self, seed):
        """Idéntica a nn_randomize() del firmware (seed 0 -> semilla por defecto)."""
        s = seed & M32 or 0x2545F491
        f32 = np.float32
        self.p[:] = 0
        for off, n, fan_in in ((self.o_w1, self.h1 * self.n_in, self.n_in),
                               (self.o_w2, self.h2 * self.h1, self.h1),
                               (self.o_w3, self.n_out * self.h2, self.h2)):
            lim = f32(np.sqrt(f32(6.0) / f32(fan_in)))
            for i in range(n):
                s = xorshift32(s)
                u = f32(s >> 8) * f32(1.0 / 16777216.0)
                self.p[off + i] = (f32(2.0) * u - f32(1.0)) * lim

    def set_params(self, p):
        self.p[:] = np.asarray(p, self.dtype)

    # ---------------------------------------------------------------- forward / backward
    def forward(self, X):
        X = np.asarray(X, self.dtype)
        a1 = np.maximum(X @ self.W1.T + self.b1, 0)
        a2 = np.maximum(a1 @ self.W2.T + self.b2, 0)
        return a2 @ self.W3.T + self.b3

    def loss_and_grad(self, X, T):
        """Devuelve (suma de pérdidas por muestra, gradiente de esa suma) -> el optimizador divide por n."""
        X = np.asarray(X, self.dtype)
        T = np.asarray(T, self.dtype)
        a1 = np.maximum(X @ self.W1.T + self.b1, 0)
        a2 = np.maximum(a1 @ self.W2.T + self.b2, 0)
        Y = a2 @ self.W3.T + self.b3
        D = Y - T
        loss_sum = float((D * D).sum()) / self.n_out
        dY = self.dtype(2.0 / self.n_out) * D
        gW3 = dY.T @ a2
        gb3 = dY.sum(0)
        dz2 = (dY @ self.W3) * (a2 > 0)
        gW2 = dz2.T @ a1
        gb2 = dz2.sum(0)
        dz1 = (dz2 @ self.W2) * (a1 > 0)
        gW1 = dz1.T @ X
        gb1 = dz1.sum(0)
        g = np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2, gW3.ravel(), gb3]).astype(self.dtype)
        return loss_sum, g

    def eval_sums(self, X, T):
        """(Σ_i mean_k d², Σ_i mean_k |d|) en unidades normalizadas, como EVAL del firmware."""
        D = self.forward(X) - np.asarray(T, self.dtype)
        return float((D * D).mean(1).sum()), float(np.abs(D).mean(1).sum())


class Optimizer:
    """Adam (con corrección de sesgo) o SGD+momentum. Mismo cálculo que nn_apply_update()."""

    def __init__(self, n_params, kind="adam", b1=0.9, b2=0.999, eps=1e-8, mu=0.9, dtype=np.float32):
        self.kind, self.dtype = kind, dtype
        self.b1, self.b2, self.eps, self.mu = dtype(b1), dtype(b2), dtype(eps), dtype(mu)
        self.m = np.zeros(n_params, dtype)
        self.v = np.zeros(n_params, dtype)
        self.b1t = dtype(1.0)
        self.b2t = dtype(1.0)

    def reset(self):
        self.m[:] = 0
        self.v[:] = 0
        self.b1t = self.dtype(1.0)
        self.b2t = self.dtype(1.0)

    def step(self, p, g_sum, n, lr):
        f = self.dtype
        g = g_sum * f(1.0 / n)
        lr = f(lr)
        if self.kind == "adam":
            self.b1t = f(self.b1t * self.b1)
            self.b2t = f(self.b2t * self.b2)
            c1 = f(1.0) / (f(1.0) - self.b1t)
            c2 = f(1.0) / (f(1.0) - self.b2t)
            self.m[:] = self.b1 * self.m + (f(1.0) - self.b1) * g
            self.v[:] = self.b2 * self.v + (f(1.0) - self.b2) * g * g
            p -= lr * (self.m * c1) / (np.sqrt(self.v * c2) + self.eps)
        else:
            self.m[:] = self.mu * self.m + g
            p -= lr * self.m


class Trainer:
    def __init__(self, mlp, kind="adam", **kw):
        self.mlp = mlp
        self.opt = Optimizer(mlp.n_params, kind, dtype=mlp.dtype, **kw)

    def train_batch(self, X, T, lr):
        loss_sum, g = self.mlp.loss_and_grad(X, T)
        self.opt.step(self.mlp.p, g, len(X), lr)
        return loss_sum
