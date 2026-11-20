#!/usr/bin/env python3
"""Lee los pesos activos del S32K3 por UART usando CMD_GET_W.

Paso 8: NO entrena, NO inicializa la red y NO envía muestras del dataset.

Por defecto guarda los pesos en el nombre canónico que consume el paso 6:
    data/model_micro_<objetivo>.npz

También genera el CSV correspondiente y firmware/src/nn_weights.h.

CMD_GET_W permite leer:
- NN_TRAIN_ON_BOARD=1: parámetros activos en RAM (después del entrenamiento).
- NN_TRAIN_ON_BOARD=0: constantes NN_WEIGHTS del encabezado, en Flash.

Uso desde la carpeta python:
    python 08_extraer_pesos_micro.py
    python 08_extraer_pesos_micro.py --url COM10@921600
    python 08_extraer_pesos_micro.py --salida data/model_micro_directo_extraido.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np

import config as cfg
from export_c_header import write_weights
from micro_link import Micro, MicroError


def _backup_if_different(path: Path, new_weights: np.ndarray) -> Path | None:
    """Respalda el NPZ que se va a reemplazar si contiene otros pesos."""
    if not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=True) as old:
            old_weights = np.asarray(old["params"], dtype=np.float32)
        if old_weights.shape == new_weights.shape and np.array_equal(old_weights, new_weights):
            return None
    except Exception:
        # Si el archivo existente no es un NPZ válido, se conserva antes de sustituirlo.
        pass

    backup = path.with_name(f"{path.stem}_antes_extraccion{path.suffix}")
    index = 1
    while backup.exists():
        backup = path.with_name(
            f"{path.stem}_antes_extraccion_{index}{path.suffix}"
        )
        index += 1
    shutil.copy2(path, backup)
    return backup


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Extrae por UART los pesos activos del micro y los guarda con el "
            "nombre que utiliza el paso 6. No ejecuta entrenamiento."
        )
    )
    ap.add_argument(
        "--url",
        default=cfg.MICRO_URL,
        help=f"Puerto@baud o exec:... (predeterminado: {cfg.MICRO_URL})",
    )
    ap.add_argument(
        "--salida",
        type=Path,
        default=None,
        help=(
            "Ruta opcional del NPZ. Por defecto: "
            "data/model_micro_<objetivo>.npz (nombre que consume el paso 6)."
        ),
    )
    ap.add_argument(
        "--sin-header",
        action="store_true",
        help="Solo guardar NPZ/CSV; no escribir firmware/src/nn_weights.h.",
    )
    args = ap.parse_args()

    mic = None
    try:
        print(f"Conectando con el micro en {args.url} ...")
        mic = Micro(args.url)
        version = mic.ping()
        info = mic.info()

        print(f"Protocolo: {version}")
        print(
            "Firmware: "
            f"train_on_board={info['train_on_board']}, "
            f"weights_in_header={info['weights_in_header']}, "
            f"red={info['n_in']}-{info['h1']}-{info['h2']}-{info['n_out']}, "
            f"parámetros={info['n_params']}"
        )

        # En modo fijo, GET_W lee NN_WEIGHTS (constante en Flash).
        # En modo de entrenamiento, GET_W lee los parámetros activos en RAM.
        if not info["train_on_board"] and not info["weights_in_header"]:
            raise RuntimeError(
                "El firmware está en modo de pesos fijos, pero INFO indica "
                "weights_in_header=False. No hay un encabezado válido para extraer."
            )

        n_in, h1, h2, n_out = (
            info["n_in"], info["h1"], info["h2"], info["n_out"]
        )
        expected = h1 * n_in + h1 + h2 * h1 + h2 + n_out * h2 + n_out
        if info["n_params"] != expected:
            raise RuntimeError(
                f"Cantidad de parámetros incoherente: INFO={info['n_params']}, "
                f"arquitectura={expected}. Revisa nn_config.h/firmware."
            )

        print("Leyendo los pesos activos con CMD_GET_W ...")
        weights = mic.get_weights(chunk=256).astype(np.float32, copy=False)

        if weights.size != info["n_params"]:
            raise RuntimeError(
                f"Se recibieron {weights.size} pesos, se esperaban {info['n_params']}."
            )
        if not np.isfinite(weights).all():
            raise RuntimeError("El vector recibido contiene NaN o infinito; no se exportó.")

        mode = "shared" if info["shared"] else "full"
        objetivo = "residual" if info["residual"] else "directo"
        arch = f"{n_in}-{h1}-{h2}-{n_out}"

        if args.salida is None:
            # Nombre canónico: el paso 6 ya carga este archivo con --modelo micro.
            output_npz = cfg.DATA / f"model_micro_{objetivo}.npz"
        else:
            output_npz = args.salida
            if not output_npz.is_absolute():
                output_npz = Path.cwd() / output_npz
        output_npz = output_npz.resolve()
        output_npz.parent.mkdir(parents=True, exist_ok=True)

        backup = _backup_if_different(output_npz, weights)
        source = (
            "UART CMD_GET_W: parámetros activos leídos de RAM (modo entrenamiento)"
            if info["train_on_board"]
            else "UART CMD_GET_W: constantes NN_WEIGHTS leídas de Flash (modo fijo)"
        )

        np.savez(
            output_npz,
            params=weights,
            arch=arch,
            mode=mode,
            objetivo=objetivo,
            n_in=n_in,
            h1=h1,
            h2=h2,
            n_out=n_out,
            n_joints=info["n_joints"],
            train_on_board=info["train_on_board"],
            weights_in_header=info["weights_in_header"],
            core_hz=info["core_hz"],
            source=source,
        )

        output_csv = output_npz.with_suffix(".csv")
        np.savetxt(
            output_csv,
            np.column_stack((np.arange(weights.size), weights)),
            delimiter=",",
            header="indice,peso",
            comments="",
            fmt=["%d", "%.9g"],
        )

        if not args.sin_header:
            header = cfg.FW_SRC / "nn_weights.h"
            write_weights(
                header,
                weights,
                arch,
                note=f"leído del micro por UART CMD_GET_W; origen: {source}; objetivo {objetivo}",
            )
            print(f"Header C     : {header}")

        print("\nExtracción terminada; no se ejecutó entrenamiento ni se reinició la red.")
        print(f"Origen       : {'RAM' if info['train_on_board'] else 'Flash'}")
        print(f"Objetivo     : {objetivo}")
        print(f"Modo         : {mode}")
        print(f"Arquitectura : {arch}")
        print(f"Pesos leídos : {weights.size}")
        print(f"Rango        : [{weights.min():.7g}, {weights.max():.7g}]")
        print(f"NPZ          : {output_npz}")
        print(f"CSV          : {output_csv}")
        if backup:
            print(f"Respaldo     : {backup}")
        if args.salida is None:
            print(
                "\nEl NPZ quedó con el nombre canónico model_micro_*.npz; "
                "el paso 6 lo leerá sin modificar 06_comparar_pid_vs_nn.py."
            )
        return 0

    except (MicroError, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if mic is not None:
            mic.close()


if __name__ == "__main__":
    raise SystemExit(main())
