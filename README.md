# CSI Image Recognition — FISE-UPB-BGA 2026

Detección de objetos en tiempo real con cámara CSI IMX219 y YOLOv8 sobre **NVIDIA Jetson Orin Nano** (JetPack 6.2, Ubuntu 22.04, aarch64).

Cada frame capturado por la cámara pasa por el pipeline GStreamer de NVIDIA, se infiere con YOLOv8s en la GPU y se visualiza con bounding boxes, etiquetas de clase, porcentaje de confianza y HUD de exhibición.

---

## Requisitos de hardware

| Componente | Detalle |
|---|---|
| Plataforma | NVIDIA Jetson Orin Nano |
| Cámara | IMX219 (conector CSI) |
| SO | JetPack 6.2 / L4T 36.5 / Ubuntu 22.04 |

---

## Requisitos de software

### Entorno virtual

Usa el venv compartido en `~/python-camara/venv-jetson/`:

| Paquete | Versión |
|---|---|
| torch | 2.8.0 (índice Jetson) |
| torchvision | 0.23.0 (índice Jetson) |
| opencv-python | sistema (`/usr/lib/python3/dist-packages`) |
| numpy | 1.26.4 (debe ser 1.x) |
| ultralytics | 8.4.21 |

> **OpenCV del sistema** — el wheel de pip no incluye GStreamer. El venv apunta al OpenCV del sistema vía el archivo `.pth` en `site-packages`. No reinstalar `opencv-python` desde pip.

> **PyTorch Jetson** — debe instalarse desde el índice específico:
> ```bash
> pip install torch==2.8.0 torchvision==0.23.0 \
>     --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
> ```

### Modelo YOLOv8s

El peso se descarga automáticamente la primera ejecución y queda en `~/.cache/ultralytics/`. También puede pre-descargarse:
```bash
source ~/python-camara/venv-jetson/bin/activate
python3 -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
```

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

## Ejecución local (monitor conectado al Jetson)

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
```

### Argumentos disponibles

| Argumento | Default | Descripción |
|---|---|---|
| `--mode` | `720p60` | Modo de sensor: `full`, `wide`, `1080p`, `1232p`, `720p60` |
| `--sensor-id` | `0` | Índice del sensor CSI |
| `--flip` | `0` | flip-method: 0=ninguno, 2=180°, 4=horizontal, 6=vertical |
| `--conf` | `0.35` | Umbral mínimo de confianza (0.0–1.0) |
| `--infer-every` | `1` | Ejecutar YOLO cada N frames |

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

## Monitoreo de GPU

| Herramienta | Muestra GPU | Instalación | Comando |
|---|---|---|---|
| `tegrastats` | Sí (Jetson nativa) | preinstalada | `sudo tegrastats` |
| `jtop` | Sí (CPU, GPU, memoria, potencia) | `pip install jetson-stats` | `sudo jtop` |
| `btop` | No* | `sudo apt install btop` | `btop` |

> \* **btop** no tiene soporte para GPUs NVIDIA por limitación propia del software; solo muestra CPU, RAM y disco.

### tegrastats

Salida en texto plano con métricas de CPU, GPU, memoria y temperatura:

```bash
sudo tegrastats
```

### jtop

Monitor interactivo con TUI, diseñado específicamente para Jetson. Muestra uso de CPU por núcleo, GPU, NVDLA, encoder/decoder de video, consumo de potencia y temperatura:

```bash
sudo pip install jetson-stats   # instalación única
sudo jtop
```

### btop

Monitor de recursos general (CPU, RAM, procesos). Útil para ver carga del sistema, pero **no muestra la GPU** ya que btop no integra soporte para GPUs NVIDIA por limitaciones propias del proyecto:

```bash
sudo apt install btop   # instalación única
btop
```

---

## Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `Could not open CSI camera` | `nvargus-daemon` detenido | `sudo systemctl start nvargus-daemon` |
| `GStreamer: NO` en OpenCV | pip wheel sin GStreamer | Usar el OpenCV del sistema (ver sección de requisitos) |
| `torch.cuda.is_available()` = False | PyTorch de PyPI estándar | Reinstalar desde `pypi.jetson-ai-lab.io` |
| NumPy ABI error | NumPy 2.x instalado | `pip install numpy==1.26.4` |
| Ventana no aparece en remoto | X11 forwarding inactivo | Usar MobaXterm o `ssh -X` con VcXsrv |
