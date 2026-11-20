"""
Enlace PC <-> micro (S32K3) con tramas binarias:

  | 0xA5 | 0x5A | cmd | seq | len (u16 LE) | payload (len bytes) | crc16 (u16 LE) |
  CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) calculado sobre  cmd..payload.
  Respuesta: cmd|0x80, mismo seq, payload[0] = status (0 = OK).

Parada y espera: la PC manda UNA trama y espera su respuesta; así el micro nunca se desborda
y nada del dataset se guarda en el micro (se entrena muestra por muestra con backprop).
Si el micro recibe un seq repetido (reintento) reenvía la respuesta anterior sin re-ejecutar.
"""
import binascii
import struct
import subprocess
import time

import numpy as np

SOF = b"\xA5\x5A"
CMD_PING, CMD_INFO, CMD_INIT, CMD_SET_LR = 0x01, 0x02, 0x03, 0x04
CMD_TRAIN, CMD_EVAL = 0x10, 0x11
CMD_GET_W, CMD_SET_W = 0x20, 0x21
CMD_INFER = 0x30
STATUS = {0: "OK", 1: "comando desconocido", 2: "longitud inválida", 3: "no disponible en este firmware "
          "(NN_TRAIN_ON_BOARD=0)", 4: "argumento inválido"}


class SerialTransport:
    def __init__(self, port, baud, timeout=2.0):
        import serial
        self.s = serial.Serial(port, baud, timeout=timeout)

    def write(self, b):
        self.s.write(b)
        self.s.flush()

    def read(self, n):
        return self.s.read(n)

    def flush_input(self):
        self.s.reset_input_buffer()

    def close(self):
        self.s.close()


class ProcessTransport:
    """Micro virtual: el firmware compilado en la PC (firmware/host_sim) conectado por pipes."""

    def __init__(self, exe):
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)

    def write(self, b):
        self.p.stdin.write(b)
        self.p.stdin.flush()

    def read(self, n):
        buf = b""
        while len(buf) < n:
            c = self.p.stdout.read(n - len(buf))
            if not c:
                break
            buf += c
        return buf

    def flush_input(self):
        pass

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=2)
        except Exception:
            self.p.kill()


def open_transport(url):
    if url.startswith("exec:"):
        return ProcessTransport(url[5:])
    port, _, baud = url.partition("@")
    return SerialTransport(port, int(baud or 921600))


class MicroError(RuntimeError):
    pass


class Micro:
    def __init__(self, url, retries=3):
        self.tp = open_transport(url) if isinstance(url, str) else url
        self.retries = retries
        self.seq = 0
        self.cfg = None
        self.bytes_tx = self.bytes_rx = 0

    def close(self):
        self.tp.close()

    # ------------------------------------------------------------------ bajo nivel
    def _read_frame(self):
        b = self.tp.read(1)
        guard = 0
        while True:
            if not b:
                return None
            if b == SOF[:1]:
                b2 = self.tp.read(1)
                if b2 == SOF[1:]:
                    break
                b = b2
                continue
            guard += 1
            if guard > 4096:
                return None
            b = self.tp.read(1)
        h = self.tp.read(4)
        if len(h) < 4:
            return None
        cmd, seq, ln = h[0], h[1], struct.unpack("<H", h[2:4])[0]
        body = self.tp.read(ln + 2)
        if len(body) < ln + 2:
            return None
        payload, crc = body[:ln], struct.unpack("<H", body[ln:])[0]
        if binascii.crc_hqx(h + payload, 0xFFFF) != crc:
            return None
        self.bytes_rx += 8 + ln
        return cmd, seq, payload

    def _txrx(self, cmd, payload=b"", fresh_seq=True):
        if fresh_seq:
            self.seq = (self.seq + 1) & 0xFF
        body = bytes([cmd, self.seq]) + struct.pack("<H", len(payload)) + payload
        frame = SOF + body + struct.pack("<H", binascii.crc_hqx(body, 0xFFFF))
        for _ in range(self.retries):
            self.tp.flush_input()
            self.tp.write(frame)
            self.bytes_tx += len(frame)
            r = self._read_frame()
            if r and r[0] == (cmd | 0x80) and r[1] == self.seq:
                if r[2][0] != 0:
                    raise MicroError(f"cmd 0x{cmd:02X}: {STATUS.get(r[2][0], r[2][0])}")
                return r[2][1:]
        raise MicroError(f"sin respuesta válida del micro (cmd 0x{cmd:02X})")

    # ------------------------------------------------------------------ comandos
    def ping(self):
        self.seq = int(np.random.randint(1, 255))
        return self._txrx(CMD_PING, fresh_seq=False)[0]

    def info(self):
        d = self._txrx(CMD_INFO)
        f = struct.unpack("<BBBBHHHHIHHIIII", d[:36])
        self.cfg = dict(proto=f[0], train_on_board=bool(f[1] & 1), shared=bool(f[1] & 2), residual=bool(f[1] & 4),
                        adam=bool(f[1] & 8), weights_in_header=bool(f[1] & 16), n_joints=f[2],
                        n_in=f[4], h1=f[5], h2=f[6], n_out=f[7], n_params=f[8], max_samples=f[9],
                        core_hz=f[11], ram_bytes=f[12], flash_bytes=f[13], infer_cycles=f[14])
        return self.cfg

    def init(self, seed):
        """seed>0: pesos aleatorios (xorshift32, igual que NumPy). seed=0: carga pesos de nn_weights.h."""
        self._txrx(CMD_INIT, struct.pack("<I", seed))

    def set_lr(self, lr):
        self._txrx(CMD_SET_LR, struct.pack("<f", lr))

    def _samples(self, X, T):
        X = np.asarray(X, "<f4")
        T = np.asarray(T, "<f4")
        return struct.pack("<H", len(X)) + np.concatenate([X, T], axis=1).astype("<f4").tobytes()

    def train_batch(self, X, T):
        d = self._txrx(CMD_TRAIN, self._samples(X, T))
        loss_sum, cyc = struct.unpack("<fI", d[:8])
        return loss_sum, cyc

    def eval_batch(self, X, T):
        d = self._txrx(CMD_EVAL, self._samples(X, T))
        se, ae, cyc = struct.unpack("<ffI", d[:12])
        return se, ae, cyc

    def get_weights(self, chunk=256):
        n = self.cfg["n_params"] if self.cfg else self.info()["n_params"]
        out = np.zeros(n, np.float32)
        for off in range(0, n, chunk):
            c = min(chunk, n - off)
            d = self._txrx(CMD_GET_W, struct.pack("<IH", off, c))
            out[off:off + c] = np.frombuffer(d[6:6 + 4 * c], "<f4")
        return out

    def set_weights(self, w, chunk=256):
        w = np.asarray(w, "<f4")
        for off in range(0, len(w), chunk):
            c = w[off:off + chunk]
            self._txrx(CMD_SET_W, struct.pack("<IH", off, len(c)) + c.tobytes())

    def infer(self, q, dq, qr, dqr):
        pl = np.concatenate([q, dq, qr, dqr]).astype("<f4").tobytes()
        d = self._txrx(CMD_INFER, pl)
        n = len(d) - 4
        return np.frombuffer(d[:n], "<f4").astype(float), struct.unpack("<I", d[n:n + 4])[0]
