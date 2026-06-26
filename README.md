# CSI Image Recognition — Marco de Observabilidad Inteligente para Plataformas Edge AI

Detección de objetos en tiempo real con cámara CSI IMX219 y YOLOv8 sobre **NVIDIA Jetson Orin Nano** (JetPack 6.2, Ubuntu 22.04, aarch64).

Cada frame capturado por la cámara pasa por el pipeline GStreamer de NVIDIA, se infiere con YOLOv8s en la GPU (PyTorch `.pt` o TensorRT FP16 `.engine`) y se visualiza con bounding boxes, etiquetas de clase, porcentaje de confianza y HUD de exhibición.

Este módulo es el componente de inferencia del marco de observabilidad inteligente descrito en:

> **"Marco de Observabilidad Inteligente para Plataformas Edge AI basadas en NVIDIA Jetson Orin Nano"**
> *Submitted to IEEE — 2026*
>
> Datos experimentales y scripts de análisis: https://github.com/ProfOmarPinzon/jetson-edge-ai-observability

---

## Requisitos de hardware

| Componente | Detalle |
|---|---|
| Plataforma | NVIDIA Jetson Orin Nano |
| Cámara | IMX219 (conector CSI) |
| SO | JetPack 6.2 / L4T 36.5 / Ubuntu 22.04 |

---

## Procedimiento de instalación

### 1. Verificar el daemon de la cámara ISP

El hardware CSI requiere que el servicio `nvargus-daemon` esté corriendo antes de cualquier acceso a la cámara:

```bash
sudo systemctl start nvargus-daemon
sudo systemctl status nvargus-daemon   # debe aparecer "active (running)"
```

Para que inicie automáticamente con el sistema:

```bash
sudo systemctl enable nvargus-daemon
```

### 2. Verificar soporte GStreamer en OpenCV

El OpenCV instalado via `pip` **no incluye soporte GStreamer**. Se necesita el OpenCV del sistema (precompilado con GStreamer en JetPack):

```bash
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep GStreamer
# Debe mostrar:  GStreamer:                   YES (...)
```

Si muestra `NO`, no uses el OpenCV de pip — usa el del sistema.

### 3. Crear o reutilizar el entorno virtual

Se usa el venv compartido en `~/python-camara/venv-jetson/`. Si no existe:

```bash
python3 -m venv ~/python-camara/venv-jetson --system-site-packages
```

> El flag `--system-site-packages` es **crítico**: permite que el venv vea el OpenCV del sistema (con GStreamer) sin necesidad de reinstalarlo.

Activar el entorno:

```bash
source ~/python-camara/venv-jetson/bin/activate
```

### 4. Instalar PyTorch desde el índice Jetson

Los builds estándar de PyPI no tienen soporte CUDA en aarch64. Se debe usar el índice específico de Jetson AI Lab:

```bash
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

Verificar que CUDA esté disponible:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
# Debe imprimir: True
```

### 5. Fijar NumPy en 1.x

El OpenCV del sistema (4.8.x) fue compilado contra NumPy 1.x. Instalar NumPy 2.x rompe el ABI en tiempo de ejecución:

```bash
pip install "numpy>=1.24,<2.0"
# o bien la versión exacta probada:
pip install numpy==1.26.4
```

Verificar:

```bash
python3 -c "import numpy; print(numpy.__version__)"
# Debe mostrar 1.26.x
```

### 6. Instalar Ultralytics (YOLOv8)

```bash
pip install ultralytics==8.4.21
```

### 7. Descargar el modelo YOLOv8s

El peso se descarga automáticamente en la primera ejecución. Para pre-descargarlo:

```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
```

El archivo queda en `~/.cache/ultralytics/`. El script busca el peso en `~/python-camara/yolov8s.pt`; copiarlo si es necesario:

```bash
cp ~/.cache/ultralytics/assets/yolov8s.pt ~/python-camara/yolov8s.pt
```

### 8. Clonar el repositorio

```bash
git clone <url-del-repo> ~/csi-image-recognition
cd ~/csi-image-recognition
```

---

## Resumen de paquetes instalados

| Paquete | Versión | Fuente |
|---|---|---|
| torch | 2.8.0 | índice Jetson (`pypi.jetson-ai-lab.io`) |
| torchvision | 0.23.0 | índice Jetson |
| opencv-python | 4.8.1.78 | sistema (via `--system-site-packages`) |
| numpy | 1.26.4 | PyPI (restringido a 1.x) |
| ultralytics | 8.4.21 | PyPI |

---

## Antes de ejecutar

```bash
# Obligatorio — el daemon ISP debe estar corriendo
sudo systemctl start nvargus-daemon

# Opcional — máximo rendimiento GPU
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## Ejecución

```bash
source ~/python-camara/venv-jetson/bin/activate

# Modo por defecto — 720p60, umbral de confianza 0.35
python3 csi_classifier.py

# 1080p, confianza más alta
python3 csi_classifier.py --mode 1080p --conf 0.4

# Reducir carga GPU (inferir 1 de cada 2 frames)
python3 csi_classifier.py --infer-every 2

# Cámara montada al revés (giro 180°)
python3 csi_classifier.py --flip 2

# Sesión desatendida (sin ventana) de 60 min con log CSV de latencia/FPS
python3 csi_classifier.py --headless --duration 3600 --log-file logs/run.csv
```

Con el script de conveniencia:

```bash
bash run_app.sh
```

### Argumentos disponibles

| Argumento | Default | Descripción |
|---|---|---|
| `--mode` | `720p60` | Modo de sensor: `full`, `wide`, `1080p`, `1232p`, `720p60` |
| `--sensor-id` | `0` | Índice del sensor CSI |
| `--flip` | `0` | flip-method: 0=ninguno, 2=180°, 4=horizontal, 6=vertical |
| `--conf` | `0.35` | Umbral mínimo de confianza (0.0–1.0) |
| `--infer-every` | `1` | Ejecutar YOLO cada N frames |
| `--headless` | `False` | Correr sin ventana (para sesiones SSH/desatendidas) |
| `--duration` | sin límite | Detener automáticamente tras N segundos |
| `--log-file` | ninguno | Ruta CSV con timestamp/latencia/fps/detecciones por frame inferido |
| `--model` | `yolov8s.pt` | Ruta al modelo: `.pt` (PyTorch) o `.engine` (TensorRT FP16) |

El HUD en pantalla muestra `FPS xx.x | Inferencia: xx.x ms | Objetos: N` — la latencia se ve siempre, con o sin `--log-file`.

### Controles en ventana

| Tecla | Acción |
|---|---|
| `q` / `ESC` | Salir |
| `s` | Guardar snapshot `snapshot_YYYYMMDD_HHMMSS.jpg` |

---

## Ejecución remota (desde Windows via SSH)

La ventana de OpenCV (`cv2.imshow`) viaja al equipo remoto mediante **X11 forwarding**.

### Opción A — MobaXterm (recomendado, sin configuración extra)

1. Instalar [MobaXterm Free Edition](https://mobaxterm.mobatek.net/) en el equipo Windows.
2. Crear una sesión SSH hacia el Jetson (el servidor X integrado se activa automáticamente).
3. En la terminal de MobaXterm:

```bash
source ~/python-camara/venv-jetson/bin/activate
python3 ~/csi-image-recognition/csi_classifier.py
```

La ventana de detección aparece en la pantalla Windows.

### Opción B — PowerShell + VcXsrv

1. Instalar [VcXsrv](https://sourceforge.net/projects/vcxsrv/) en Windows.
2. Ejecutar **XLaunch** → *Multiple windows* → marcar *Disable access control*.
3. Conectar al Jetson con X11 forwarding:

```powershell
ssh -X opinzon@<ip-del-jetson>
```

4. En la sesión SSH:

```bash
source ~/python-camara/venv-jetson/bin/activate
python3 ~/csi-image-recognition/csi_classifier.py
```

### Opción C — VSCode Remote-SSH

VSCode Remote-SSH no reenvía X11 por sí solo. La forma más práctica es:

- Usar VSCode para **editar** el código.
- Abrir una sesión de MobaXterm en paralelo para **ejecutar y visualizar**.

---

## Pipeline GStreamer

```
nvarguscamerasrc (sensor ISP, NVMM, AE/AWB)
    → nvvidconv  (resize, formato, flip — zero-copy)
    → video/x-raw, format=BGRx
    → videoconvert
    → video/x-raw, format=BGR
    → appsink  → OpenCV
```

`flip-method`: 0=ninguno, 1=ccw90°, 2=180°, 3=cw90°, 4=horizontal, 6=vertical.

---

## Modos del sensor IMX219

| Modo | Resolución | FPS |
|---|---|---|
| `full` | 3280×2464 | 21 |
| `wide` | 3280×1848 | 28 |
| `1080p` | 1920×1080 | 30 |
| `1232p` | 1640×1232 | 30 |
| `720p60` | 1280×720 | 60 |

---

## Exportación a TensorRT FP16

Para máximo rendimiento en Jetson, exportar el modelo a TensorRT antes de ejecutar:

```bash
source .venv/bin/activate
python3 -c "
from ultralytics import YOLO
YOLO('yolov8s.pt').export(format='engine', half=True, imgsz=640, device=0)
"
# Genera yolov8s.engine (~25 MB, tiempo de build ≈8 min primera vez)
```

Usar el engine en inferencia:

```bash
python3 csi_classifier.py --model yolov8s.engine --headless --duration 3600 --log-file logs/run.csv
```

Resultados medidos (Jetson Orin Nano 15W, `jetson_clocks`):

| Modelo | Latencia media | FPS |
|--------|---------------|-----|
| YOLOv8s `.pt` (FP32) | ~90 ms | ~11 |
| YOLOv8s `.engine` (FP16) | **43.9 ms** | **22.8** |

> El archivo `.engine` es específico para cada combinación de GPU + TensorRT + imagen de entrada. No es portable entre versiones de JetPack.

---

## Monitoreo de GPU

| Herramienta | Muestra GPU | Comando |
|---|---|---|
| `tegrastats` | Sí (nativa Jetson) | `sudo tegrastats` |
| `jtop` | Sí (CPU, GPU, potencia, temperatura) | `sudo jtop` |
| `btop` | No (sin soporte NVIDIA) | `btop` |

Instalar `jtop` (solo la primera vez):

```bash
sudo pip install jetson-stats
```

---

## Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `Could not open CSI camera` | `nvargus-daemon` detenido | `sudo systemctl start nvargus-daemon` |
| `GStreamer: NO` en OpenCV | pip wheel sin GStreamer | Usar el OpenCV del sistema (venv con `--system-site-packages`) |
| `torch.cuda.is_available()` = False | PyTorch de PyPI estándar | Reinstalar desde `pypi.jetson-ai-lab.io` |
| NumPy ABI error al importar OpenCV | NumPy 2.x instalado | `pip install numpy==1.26.4` |
| Ventana no aparece en remoto | X11 forwarding inactivo | Usar MobaXterm o `ssh -X` con VcXsrv |
| `CANCELLED` en logs de argus | Mensaje normal de salida ISP | No es un error — ignorar |

---

## Proyectos de referencia

| Proyecto | Ruta | Qué aporta |
|---|---|---|
| csi-test | `~/csi-test/` | Pipeline CSI, modos de sensor, patrones GStreamer |
| python-camara | `~/python-camara/` | YOLOv8 en GPU, detección de contornos, configuración del venv |
