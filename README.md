# Control neuronal de un manipulador de 7 GDL

**CoppeliaSim + Python + NXP S32K312 mini / S32K344**  
**Estrategia:** seguimiento articular de trayectorias mediante una red neuronal entrenada por imitación de un controlador PID.

Este repositorio contiene el flujo de trabajo de la práctica: caracterización del manipulador, detección de colisiones, control PID, adquisición y preprocesamiento de datos, entrenamiento de una MLP en PC o en el microcontrolador, exportación de pesos, simulación comparativa y análisis de seguridad.

El manipulador y la planta se simulan en **CoppeliaSim**. El S32K312 se utiliza como plataforma embebida para entrenar la red y/o ejecutar su inferencia; no se controla un manipulador físico desde el microcontrolador en esta práctica.

---

## 1. Estructura del repositorio

```text
control_neuronal_7dof/
├── README.md
├── .gitignore
├── python/
│   ├── 01_caracterizar_robot.py
│   ├── 01b_detectar_colisiones.py
│   ├── 02_evaluar_pid.py
│   ├── 03_generar_dataset.py
│   ├── 04_preprocesar.py
│   ├── 05_entrenar_pc.py
│   ├── 06_comparar_pid_vs_nn.py
│   ├── 07_entrenar_en_micro.py
│   ├── 08_extraer_pesos_micro.py
│   ├── 09_analisis_seguridad.py
│   ├── config.py
│   ├── controllers.py
│   ├── coppelia_robot.py
│   ├── micro_link.py
│   ├── nn_data.py
│   ├── nn_numpy.py
│   ├── export_c_header.py
│   ├── sim_loop.py
│   ├── scenarios.py
│   ├── trajectories.py
│   ├── mock_robot.py
│   ├── tests_paridad.py
│   ├── requirements.txt
│   └── data/                 # datasets, modelos, métricas y figuras
└── firmware/
    ├── src/
    │   ├── main.c
    │   ├── bsp.h
    │   ├── bsp_s32k3.c
    │   ├── nn_user_cfg.h
    │   ├── nn_config.h
    │   ├── nn_weights.h
    │   ├── nn_mlp.c / nn_mlp.h
    │   ├── nn_proto.c / nn_proto.h
    │   └── ring_buffer.c / ring_buffer.h
    └── host_sim/              # compilación del firmware como proceso de PC
```

### Función de los programas numerados

| Script | Función |
|---|---|
| `01_caracterizar_robot.py` | Lee límites articulares y mide la respuesta del manipulador a comandos de velocidad. Guarda `robot_params.json`. |
| `01b_detectar_colisiones.py` | Detecta contactos geométricos, calcula rangos articulares seguros y conserva el informe de colisiones en `robot_params.json`. |
| `02_evaluar_pid.py` | Evalúa el PID de referencia. `--tune` ejecuta una búsqueda de ganancias y guarda `pid_gains.json`. |
| `03_generar_dataset.py` | Ejecuta episodios con PID y guarda estados, referencias, acciones de control y metadatos. |
| `04_preprocesar.py` | Limpia los datos, divide por episodios, normaliza las entradas y genera `nn_config.h`. |
| `05_entrenar_pc.py` | Entrena la MLP en Python y guarda el modelo, historial, métricas y figura de entrenamiento. |
| `06_comparar_pid_vs_nn.py` | Compara PID, red neuronal de PC y, cuando se conecta, inferencia del microcontrolador (PIL). Evalúa condiciones nominales, perturbaciones y generalización. |
| `07_entrenar_en_micro.py` | Envía lotes de muestras por UART; el micro realiza forward y backpropagation sin guardar el dataset completo. Guarda el modelo extraído y las métricas. |
| `08_extraer_pesos_micro.py` | Lee los pesos activos del micro mediante `CMD_GET_W`. No entrena ni reinicia la red. Guarda un `.npz`, un `.csv` y actualiza `nn_weights.h`. |
| `09_analisis_seguridad.py` | Evalúa saturación, límites, ganancias efectivas y escenarios adversos en lazo cerrado. Carga el modelo guardado en PC; no realiza inferencia UART en cada paso. |

Los módulos restantes contienen los controladores, la interfaz de CoppeliaSim, la comunicación UART, la MLP NumPy, el preprocesamiento, las trayectorias y las utilidades compartidas.

---

## 2. Requisitos

### Software en la PC

- Windows 10/11 (los comandos de este documento usan PowerShell).
- Python 3.x.
- CoppeliaSim con una escena que contenga el manipulador de siete grados de libertad.
- Visual Studio Code (opcional, recomendado).
- Git (opcional, para clonar y versionar el proyecto).

### Hardware embebido

- NXP S32K312 mini-EVB (utilizado en las pruebas) 
- Proyecto de S32 Design Studio con RTD y configuración de reloj, pines e interfaz LPUART6.
- Conexión USB/serie entre la PC y la tarjeta.

> **Importante:** la carpeta `firmware/` contiene los fuentes de la aplicación y el BSP, pero no es un proyecto S32 Design Studio autocontenido. Se debe integrar en un proyecto S32K3 existente que tenga los archivos generados por Config Tools/RTD (`.mex`, configuración de Clock, SIUL2, IntCtrl y LPUART). 

---

## 3. Preparar el entorno de Python

Abrir PowerShell en la carpeta raíz del repositorio y ejecutar:

```powershell
cd .\python
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Las dependencias están listadas en `python/requirements.txt`:

- NumPy
- SciPy
- Matplotlib
- pandas
- pyserial
- coppeliasim-zmqremoteapi-client

Los scripts deben ejecutarse **desde la carpeta `python/`**, porque `config.py` define las rutas de datos y firmware relativas a la ubicación del proyecto.

Para confirmar la instalación:

```powershell
python -c "import numpy, scipy, matplotlib, pandas, serial; import coppeliasim_zmqremoteapi_client; print('Entorno Python listo')"
```

---

## 4. Configurar CoppeliaSim

1. Abrir la escena que contiene el modelo del manipulador **7 DoF**.
2. Comprobar que el modelo tiene siete juntas de revolución y que la base está fija.
3. Habilitar el servidor ZeroMQ Remote API de CoppeliaSim. La configuración incluida en `config.py` intenta conectarse a `localhost:23000`.
4. Detener la simulación antes de ejecutar los scripts. La interfaz del proyecto gestiona el arranque, el avance por pasos y el reinicio de episodios.
5. Verificar que los shapes relevantes del robot y del entorno tengan habilitada la propiedad **Collidable** para la detección geométrica.
6. Confirmar en la consola qué siete juntas detectó el programa y que están ordenadas desde la base hasta el efector.

La configuración de conexión y del robot está en `python/config.py`:

```python
ZMQ_HOST = "localhost"
ZMQ_PORT = 23000
ROBOT_ROOT = None
JOINT_PATHS = None
N_JOINTS = 7
DT = 0.01
```

Si la autodetección selecciona juntas de otro objeto de la escena, definir `ROBOT_ROOT` o especificar las siete rutas exactas en `JOINT_PATHS`. No continuar con la adquisición hasta comprobar el orden de las juntas.

### Uso de `--mock`

Algunos scripts permiten `--mock` para probar el pipeline con un manipulador simulado en Python, sin CoppeliaSim. Es útil para depuración, pero **los resultados con `--mock` no sustituyen las pruebas finales de la escena CoppeliaSim**.

`--mock` está disponible en `01_caracterizar_robot.py`, `01b_detectar_colisiones.py`, `02_evaluar_pid.py`, `03_generar_dataset.py`, `06_comparar_pid_vs_nn.py` y `09_analisis_seguridad.py`. No todos los scripts aceptan esta opción.

---

## 5. Flujo de trabajo completo en PC

Ejecutar los comandos desde `python/`, con CoppeliaSim configurado y accesible.

### Paso 1. Caracterizar el manipulador

```powershell
python 01_caracterizar_robot.py
```

Guarda `data/robot_params.json` y `data/figs/01_respuesta_escalon.png`. Se obtienen límites articulares, velocidades y aceleraciones máximas, respuesta al comando de velocidad y un límite de control `u_max`.

### Paso 2. Detectar colisiones y obtener rangos seguros

```powershell
python 01b_detectar_colisiones.py --n 8
```

Hace un barrido por articulación y prueba trayectorias candidatas. Actualiza `robot_params.json` con `q_min_safe`, `q_max_safe`, `range_frac` y el informe de contactos. Si la pose inicial está en colisión, se puede indicar otra mediante:

```powershell
python 01b_detectar_colisiones.py --n 8 --home 0,0,0,0,0,0,0
```

La pose del ejemplo debe sustituirse por una pose que esté libre en la escena. No ignorar advertencias de colisión ni desactivar la propiedad Collidable para forzar la ejecución.

### Paso 3. Evaluar el controlador PID

```powershell
python 02_evaluar_pid.py
```

Guarda `data/pid_metrics.json` y `data/figs/02_pid.png`. Si se desea buscar otras ganancias:

```powershell
python 02_evaluar_pid.py --tune
```

`--tune` es opcional y sobrescribe/crea `data/pid_gains.json`. Las ganancias guardadas en ese archivo se usan en las ejecuciones posteriores. Registrar las ganancias utilizadas para que los resultados puedan reproducirse.

### Paso 4. Generar el dataset

```powershell
python 03_generar_dataset.py --episodes 80 --seed 1
```

El script usa cuatro tipos de referencia: escalón, senoidal, triangular y aleatoria limitada. Incluye variaciones de carga, ganancia de motor, ruido y empujones pequeños. Registra una de cada cinco muestras del periodo de simulación de 10 ms (frecuencia efectiva de adquisición de 20 Hz).

Archivos generados:

- `data/dataset_raw.npz`
- `data/dataset_raw.csv`

Si algunos episodios son omitidos por colisiones, revisar el resultado del paso 2 y la pose/rangos de la escena.

### Paso 5. Preprocesar los datos

Para el controlador directo utilizado en los resultados finales:

```powershell
python 04_preprocesar.py --objetivo directo
```

Genera `proc_directo.npz`, `norm_directo.npz`, `split_directo.json` y **reescribe `firmware/src/nn_config.h`** con la arquitectura, normalización, límites y objetivo seleccionados.

La división se realiza por episodios y de forma estratificada por tipo de trayectoria: 70 % entrenamiento, 15 % validación y 15 % prueba. Los estadísticos de normalización se calculan únicamente con el conjunto de entrenamiento.

El proyecto utiliza por defecto una MLP compartida de arquitectura `6-16-16-1` (401 parámetros), aplicada individualmente a cada articulación. Las entradas por articulación son `[q, dq, qr, dqr, e, de]`; la salida es el comando de velocidad articular aprendido a partir de la acción PID.

> **Cuidado:** `04_preprocesar.py` sobrescribe `firmware/src/nn_config.h`. Si se cambia entre objetivos `directo` y `residual`, hay que compilar el firmware con el `nn_config.h` correspondiente al mismo objetivo que los pesos.

### Paso 6. Entrenar la red en PC

```powershell
python 05_entrenar_pc.py --objetivo directo --epochs 40
```

Guarda, entre otros:

- `data/model_pc_directo.npz`
- `data/history_pc_directo.csv`
- `data/metrics_pc_directo.json`
- `data/figs/05_entrenamiento_pc_directo.png`

La mejor época se selecciona con la pérdida de validación y las métricas se evalúan también sobre el conjunto de prueba.

La opción `--export` escribe los pesos del entrenamiento de PC en `firmware/src/nn_weights.h`. **No usar `--export` si se desea conservar en ese encabezado el modelo final entrenado en el microcontrolador**, porque sobrescribiría los pesos del micro.

### Paso 7. Comparar PID y red entrenada en PC

```powershell
python 06_comparar_pid_vs_nn.py --objetivo directo --modelo pc
```

Genera `comparacion_directo_pc.csv`, `tabla_comparacion_directo_pc.csv`, `resumen_directo_pc.json` y las figuras de comparación/perturbaciones en `data/figs/`. Compara las condiciones nominales, perturbaciones y generalización con nuevas semillas, poses, velocidades y cargas.

---

## 6. Integrar y compilar el firmware en S32 Design Studio

### 6.1 Integración del código

La implementación `firmware/src/` está preparada para integrarse en un proyecto S32K3 con RTD. La capa `bsp_s32k3.c` utiliza la configuración de LPUART6 y los símbolos generados por Config Tools. Para el proyecto de S32K312 utilizado en esta práctica se espera:

- LPUART6, instancia 6.
- RX/TX en los pines configurados para LPUART6 (en la configuración usada: PTA15/PTA16).
- Callback del driver con nombre `Uart_Callback`.
- IRQ de LPUART6 habilitada.
- Configuración generada `Lpuart_Uart_Ip_xHwConfigPB_6`.
- Configuración de Clock, SIUL2 y OsIf consistente con la inicialización del BSP.

Procedimiento general:

1. Duplicar el proyecto S32K3 existente que ya tenga RTD, Clock, pines y LPUART6 configurados.
2. Conservar sus archivos `.mex` y los archivos generados por Config Tools.
3. Excluir de la compilación el `main.c` anterior y los módulos de la aplicación que ya no se utilicen.
4. Agregar los archivos de `firmware/src/` al proyecto: `main.c`, `bsp_s32k3.c`, `bsp.h`, `nn_mlp.c/.h`, `nn_proto.c/.h`, `ring_buffer.c/.h` y los encabezados `nn_*.h`.
5. **No agregar `firmware/host_sim/` al proyecto embebido.** Esa carpeta es únicamente para compilar el firmware como proceso de PC.
6. Verificar que exista un solo `main()` y que los nombres de los símbolos generados coincidan con los usados en `bsp_s32k3.c`.

La aplicación usa `BSP_CORE_CLOCK_HZ` para convertir ciclos del contador DWT a tiempo. Para el S32K312 utilizado, debe coincidir con el reloj real configurado (120 MHz en las pruebas descritas). No usar ese valor para otra tarjeta/reloj sin comprobarlo.

### 6.2 Entrenamiento en el microcontrolador

Antes de entrenar, `nn_config.h` debe corresponder a `directo` y a la arquitectura del dataset. En `firmware/src/nn_user_cfg.h` configurar:

```c
#define NN_TRAIN_ON_BOARD 1
#define NN_INIT_FROM_HEADER 0
#define NN_USE_ADAM 1
#define BSP_CORE_CLOCK_HZ 120000000UL
#define NN_UART_BAUD 921600UL
```

`NN_UART_BAUD` debe mantenerse coherente como referencia, pero la velocidad real se configura en Config Tools para LPUART6. El baud de Config Tools debe coincidir con el usado en `--url` (en las pruebas: `921600`). También debe coincidir el optimizador (`NN_USE_ADAM`) con `OPTIMIZER` de `python/config.py`.

Compilar y grabar el firmware en el S32K312. Cerrar cualquier monitor serie que tenga abierto el mismo puerto COM. Desde `python/` ejecutar, por ejemplo, para el entrenamiento de 30 épocas utilizado en la práctica:

```powershell
python 07_entrenar_en_micro.py --url COM10@921600 --objetivo directo --epochs 30 --batch 1 --export
```

Con `BATCH=1`, la PC envía una muestra por actualización. El micro ejecuta forward y backpropagation y actualiza los parámetros; el dataset completo permanece en la PC. La validación también se transmite por UART y el micro devuelve las sumas de error/ciclos. `--max-train` y `--max-val` permiten limitar las muestras por época; el valor predeterminado `0` significa utilizar todas.

El programa guarda `model_micro_directo.npz`, `history_micro_directo.csv`, `metrics_micro_directo.json` y la figura `07_entrenamiento_micro_directo.png`. Con `--export`, también escribe `firmware/src/nn_weights.h` con los pesos finales de la mejor época.

> El entrenamiento completo puede tardar bastante debido al envío de los datos por UART. No ejecutar este paso si ya se dispone de los pesos entrenados y solo se necesita extraerlos o validar la inferencia.

### 6.3 Extraer los pesos ya entrenados (sin volver a entrenar)

Si el micro ya tiene los pesos entrenados en RAM o está ejecutando el firmware fijo con los pesos en Flash, se pueden leer directamente con:

```powershell
python 08_extraer_pesos_micro.py --url COM10@921600
```

Este script **no ejecuta entrenamiento, no envía el dataset y no reinicia la red**. Utiliza `CMD_GET_W` para leer los parámetros activos:

- Si `NN_TRAIN_ON_BOARD=1`, lee los pesos activos en RAM.
- Si `NN_TRAIN_ON_BOARD=0` y `weights_in_header=True`, lee las constantes de `nn_weights.h` en Flash.

Por defecto guarda/actualiza:

- `data/model_micro_directo.npz` (nombre que utiliza el comparador del paso 6).
- `data/model_micro_directo.csv`.
- `firmware/src/nn_weights.h`.

Si el `.npz` existente tiene otros pesos, el script crea un respaldo `*_antes_extraccion.npz` antes de reemplazarlo. Para guardar en otra ruta se puede usar `--salida ruta\archivo.npz`. Para no modificar el header se puede agregar `--sin-header`.

### 6.4 Cambiar a inferencia con pesos fijos

Una vez que `nn_weights.h` contiene los pesos correctos, cambiar en `nn_user_cfg.h`:

```c
#define NN_TRAIN_ON_BOARD 0
#define NN_INIT_FROM_HEADER 0
#define NN_USE_ADAM 1
#define BSP_CORE_CLOCK_HZ 120000000UL
#define NN_UART_BAUD 921600UL
```

Verificar también:

```c
#define NN_WEIGHTS_VALID 1
#define NN_WEIGHTS_COUNT 401u
```

Y que `nn_config.h` sea el generado para `objetivo: directo` (en particular `NN_IS_RESIDUAL 0`). Recompilar y volver a grabar el firmware. Con `NN_TRAIN_ON_BOARD=0`, el firmware utiliza los pesos constantes de Flash y ejecuta inferencia sin volver a entrenar al encender.

Después de reiniciar, consultar `INFO` usando los scripts de comunicación. Para la versión final se espera `train_on_board=False`, `weights_in_header=True`, arquitectura `6-16-16-1` y `n_params=401`.

---

## 7. Validación con el micro dentro del lazo de CoppeliaSim (PIL)

Con CoppeliaSim abierto, el S32K312 conectado y el firmware fijo ejecutándose:

```powershell
python 06_comparar_pid_vs_nn.py --objetivo directo --modelo micro --micro COM10@921600
```

Este ensayo utiliza el S32K312 para responder inferencias durante la simulación (Processor-in-the-Loop). La opción `--modelo micro` hace que el comparador cargue `data/model_micro_directo.npz` para la referencia NumPy, mientras que `--micro COM10@921600` agrega el controlador que consulta el firmware por UART. No agregar `--push-weights` en el modo fijo: esa opción intenta escribir los parámetros en el micro y está destinada a un firmware con entrenamiento habilitado.

El comparador informa:

- Seguimiento nominal para escalón, senoidal, triangular y aleatoria.
- Degradación ante carga, variación de ganancia del motor, empujón y ruido.
- Generalización con referencias/poses nuevas.
- Tiempo de inferencia, ciclos, RAM/Flash informadas por el firmware y porcentaje del periodo de control.
- Fidelidad entre la inferencia del micro y NumPy, evaluada durante 300 muestras.

Una diferencia máxima micro-NumPy menor que `1e-3 rad/s` se muestra como `OK` en el propio script. En la ejecución validada de esta práctica se obtuvo `5.29e-07 rad/s`.

---

## 8. Análisis de seguridad y estabilidad

El análisis se ejecuta con:

```powershell
python 09_analisis_seguridad.py --objetivo directo --modelo micro
```

El script carga `norm_directo.npz` y `model_micro_directo.npz`. Realiza una evaluación offline de salidas acotadas, NaN/Inf, saturación y ganancias efectivas. Además, ejecuta pruebas de lazo cerrado en CoppeliaSim con perturbaciones fuertes y fallos de sensor simulados.

**Aclaración:** aunque se seleccione `--modelo micro`, este script utiliza en la simulación el modelo NumPy guardado con los pesos obtenidos del micro; no envía cada inferencia al S32K por UART. La evaluación PIL real del firmware se realiza con `06_comparar_pid_vs_nn.py --modelo micro --micro ...`.

Para ejecutar solo la parte offline, sin CoppeliaSim:

```powershell
python 09_analisis_seguridad.py --objetivo directo --modelo micro --solo-offline
```

Para una prueba de depuración con el robot simulado de Python:

```powershell
python 09_analisis_seguridad.py --objetivo directo --modelo micro --mock
```

Los resultados se guardan en `data/seguridad_directo_micro.json`, `data/seguridad_directo_micro.csv` (cuando se ejecuta la parte de lazo cerrado) y `data/figs/09_seguridad_directo_micro.png`.

---

## 9. Pruebas opcionales de paridad y micro virtual

`python/tests_paridad.py` verifica el gradiente analítico frente a diferencias finitas y puede comparar el firmware C con NumPy. Para usar el micro virtual es necesario compilarlo primero con GCC/Make disponible en Windows:

```powershell
make -C ..\firmware\host_sim
python tests_paridad.py exec:..\firmware\host_sim\micro_sim.exe
```

Si el ejecutable generado se llama `micro_sim` en lugar de `micro_sim.exe`, ajustar el nombre en el comando. También se puede indicar un micro físico:

```powershell
python tests_paridad.py COM10@921600
```

La prueba con micro físico puede cambiar pesos si el firmware está en modo entrenamiento; utilizarla con precaución y respaldar el modelo antes de ejecutarla.

---

## 10. Archivos de resultados

Los resultados se guardan bajo `python/data/`:

| Archivo/patrón | Contenido |
|---|---|
| `robot_params.json` | Parámetros del manipulador, límites y reporte de colisiones. |
| `pid_metrics.json`, `pid_gains.json` | Métricas y, si se ajustó, ganancias del PID. |
| `dataset_raw.npz`, `dataset_raw.csv` | Dataset original y versión legible. |
| `proc_<objetivo>.npz`, `norm_<objetivo>.npz`, `split_<objetivo>.json` | Datos procesados, normalización y división por episodios. |
| `model_pc_<objetivo>.npz`, `model_micro_<objetivo>.npz` | Parámetros de los modelos entrenados. |
| `history_*.csv`, `metrics_*.json` | Historial y métricas de entrenamiento. |
| `comparacion_*.csv`, `tabla_comparacion_*.csv`, `resumen_*.json` | Resultados de comparación y tablas agregadas. |
| `seguridad_*.json`, `seguridad_*.csv` | Resultados del análisis de seguridad. |
| `data/figs/*.png` | Gráficas generadas por cada etapa. |

Los archivos existentes en `data/` son resultados de ejecuciones previas. Los scripts pueden sobrescribirlos al repetir los pasos. Respaldar los datos que se quieran conservar antes de volver a generar datasets, normalizaciones, modelos o configuraciones.

---

## 11. Recursos de memoria del firmware

El comando `INFO` del firmware informa recursos asociados a la red neuronal. Para conocer los recursos del firmware completo, usar el resultado **Print Size** de S32 Design Studio:

```bash
arm-none-eabi-size --format=berkeley NeuroController.elf
```

Interpretación:

- `text`: código y constantes de solo lectura ubicados en Flash.
- `data`: variables inicializadas; ocupan Flash y RAM.
- `bss`: variables estáticas no inicializadas, ocupan RAM.
- Estimación habitual de Flash: `text + data`.
- RAM estática: `data + bss`.

En una compilación reportada para esta práctica se obtuvo `text=69532`, `data=4` y `bss=17012`, por lo que la estimación fue 69,536 B de Flash (`text + data`) y 17,016 B de RAM estática (`data + bss`). Estos valores corresponden a esa compilación concreta y deben volver a medirse si cambia el firmware, la configuración o el archivo de pesos.

---

## 12. Solución de problemas

| Síntoma | Comprobación |
|---|---|
| No conecta con CoppeliaSim | Confirmar que ZeroMQ Remote API está activo, que CoppeliaSim escucha en el puerto configurado (`ZMQ_PORT=23000`) y que la escena está abierta. |
| Se detectan menos o más de siete juntas | Ajustar `ROBOT_ROOT` o `JOINT_PATHS` en `config.py`; confirmar que las juntas son de revolución y pertenecen al manipulador. |
| La detección de colisiones no encuentra shapes | Activar `Collidable` en los shapes del robot y del entorno. Confirmar una pose `home` libre. |
| `No such file` al preprocesar/entrenar | Ejecutar desde `python/` y comprobar que existe el archivo anterior requerido en `python/data/`. |
| El micro no responde | Comprobar el COM, cerrar otros monitores serie, usar el baud configurado en LPUART6 y verificar RX/TX, GND y alimentación/conexión de la tarjeta. |
| `07_entrenar_en_micro.py` indica firmware de pesos fijos | Para entrenar, recompilar con `NN_TRAIN_ON_BOARD=1`. Para inferencia fija, no ejecutar el paso 7. |
| Arquitectura u objetivo no coinciden | Ejecutar `04_preprocesar.py --objetivo directo` (o `residual` si corresponde), recompilar y grabar. El firmware y los NPZ deben usar la misma arquitectura y objetivo. |
| Advertencia de pesos diferentes en el paso 6 | Ejecutar `08_extraer_pesos_micro.py --url COMx@921600` para actualizar el archivo canónico `model_micro_<objetivo>.npz`, y repetir el paso 6. |
| `weights_in_header=False` en modo fijo | Comprobar `NN_WEIGHTS_VALID`, `NN_WEIGHTS_COUNT`, que `nn_weights.h` esté incluido en la compilación y que `NN_TRAIN_ON_BOARD=0`. |

---

## 13. Alcance y reproducibilidad

El flujo final de referencia para el controlador directo es:

```text
CoppeliaSim
    ↓
Caracterización y colisiones (01, 01b)
    ↓
Control PID y adquisición del dataset (02, 03)
    ↓
Preprocesamiento y nn_config.h directo (04)
    ↓
Entrenamiento de referencia en PC (05)
    ↓
Entrenamiento en S32K312 (07) — si se requiere repetirlo
    ↓
Extracción de pesos activos sin entrenar (08)
    ↓
Firmware con NN_TRAIN_ON_BOARD=0 y pesos fijos
    ↓
Comparación PIL en CoppeliaSim (06, --modelo micro)
    ↓
Análisis de seguridad/estabilidad en simulación (09)
```

Para reutilizar los resultados ya guardados, no es necesario repetir el entrenamiento. Se puede extraer/actualizar el modelo desde el micro con el paso 8, comprobar que `nn_config.h` corresponde a `directo`, configurar el firmware fijo y ejecutar las pruebas de comparación y seguridad.

El modo residual (`objetivo=residual`) está soportado por los scripts como una extensión. Cambiar a residual requiere volver a preprocesar, entrenar el objetivo correspondiente y generar/compilar los encabezados con ese mismo objetivo. No mezclar `nn_config.h`, `nn_weights.h`, normalización ni modelos de objetivos diferentes.

---

## 14. Licencia y citas

Agregar aquí la licencia del proyecto y las referencias académicas/técnicas que correspondan antes de publicar el repositorio. No se declara una licencia de uso en este README.
