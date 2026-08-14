# ALPR - Automatic License Plate Recognition

Sistema de reconocimiento automático de patentes vehiculares
(Automatic License Plate Recognition - ALPR).

Proyecto de investigación y desarrollo.

---

## 1. Objetivo

El proyecto tiene como objetivo desarrollar y evaluar un sistema de
reconocimiento automático de patentes vehiculares a partir de imágenes
y/o secuencias de video.

El sistema contempla, inicialmente, las siguientes etapas:

1. Captura de imágenes o video.
2. Detección de vehículos.
3. Detección de la región correspondiente a la patente.
4. Recorte y preprocesamiento de la patente.
5. Reconocimiento óptico de caracteres (OCR).
6. Postprocesamiento del texto reconocido.
7. Evaluación de la precisión del sistema.
8. Evaluación de desempeño en tiempo de ejecución.

La arquitectura se diseñará de manera modular para permitir comparar
distintos algoritmos de detección y reconocimiento.

---

# 2. Tecnologías principales

El entorno inicial utiliza:

| Componente        |            Versión |
| ----------------- | -----------------: |
| Sistema operativo | Windows 11 64 bits |
| Python            |             3.11.x |
| PyTorch           |             2.11.0 |
| TorchVision       |             0.26.0 |
| CUDA Runtime      |               12.8 |
| Ultralytics       |             8.4.91 |
| OpenCV            |          4.11.0.86 |
| NumPy             |             1.26.4 |
| Pillow            |             11.3.0 |
| Pandas            |              2.3.1 |
| SciPy             |             1.16.0 |
| Scikit-learn      |              1.7.1 |
| Matplotlib        |             3.10.3 |
| PyYAML            |              6.0.2 |
| tqdm              |             4.67.1 |
| psutil            |              7.0.0 |

El entorno utiliza una GPU NVIDIA cuando está disponible.

---

# 3. Hardware recomendado

El entorno de desarrollo de referencia utiliza:

- GPU: NVIDIA GeForce RTX 4060
- VRAM: 8 GB
- CPU: Intel Core i5-12400
- RAM: 32 GB
- Sistema operativo: Windows 11 64 bits

El sistema puede funcionar con otras GPU NVIDIA compatibles con la
versión de CUDA utilizada por PyTorch.

---

# 4. Requisitos previos

Antes de crear el entorno Python se debe disponer de:

- Windows 11 64 bits.
- Python 3.11.
- NVIDIA GPU compatible, si se desea utilizar aceleración CUDA.
- NVIDIA Driver actualizado.
- Git.
- Visual Studio Code.

No es necesario instalar Anaconda.

No es necesario instalar Miniconda.

No es necesario instalar CUDA Toolkit manualmente para utilizar
PyTorch mediante los wheels oficiales.

---

# 5. Instalación de Python

## 5.1. Verificar si Python está instalado

Abrir PowerShell y ejecutar:

```powershell
python --version
```
