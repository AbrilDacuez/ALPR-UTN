import time
import cv2
from ultralytics import RTDETR

RUTA_VIDEO_ENTRADA = "Etapa 2/inputs/trafico.mp4"
RUTA_VIDEO_SALIDA = "Etapa 2/outputs/resultado_video.mp4"

UMBRAL_CONFIANZA = 0.5
CLASES_INTERES = [2, 3, 5, 7]  # 2=auto, 3=moto, 5=colectivo, 7=camión

NOMBRES_ESPANOL = {
    "car": "Auto",
    "motorcycle": "Moto",
    "bus": "Colectivo",
    "truck": "Camion",
}

print("[INFO] Cargando modelo RT-DETR...")
modelo = RTDETR("rtdetr-l.pt")
video = cv2.VideoCapture(RUTA_VIDEO_ENTRADA)
if not video.isOpened():
    raise FileNotFoundError(f"No se pudo abrir el video en: {RUTA_VIDEO_ENTRADA}")

ancho = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_original = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

codec = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(RUTA_VIDEO_SALIDA, codec, fps_original, (ancho, alto))

tiempos_inferencia = []
numero_frame = 0

print(f"[INFO] Procesando video ({total_frames} frames totales)...")

while True:
    ret, frame = video.read()
    if not ret:
        break  # Fin del video

    numero_frame += 1

    t_inicio = time.perf_counter()
    resultados = modelo.predict(
        source=frame, conf=UMBRAL_CONFIANZA, classes=CLASES_INTERES, verbose=False
    )
    t_fin = time.perf_counter()

    tiempo_inferencia_ms = (t_fin - t_inicio) * 1000
    fps_procesamiento = 1.0 / (t_fin - t_inicio) if (t_fin - t_inicio) > 0 else 0
    tiempos_inferencia.append(tiempo_inferencia_ms)

    detecciones = resultados[0].boxes

    for caja in detecciones:
        x1, y1, x2, y2 = map(int, caja.xyxy[0])
        confianza = float(caja.conf[0])
        id_clase = int(caja.cls[0])

        nombre_ingles = modelo.names[id_clase]
        etiqueta = NOMBRES_ESPANOL.get(nombre_ingles, nombre_ingles)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        texto_det = f"{etiqueta}: {confianza:.2f}"
        (w_txt, h_txt), _ = cv2.getTextSize(texto_det, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            frame, (x1, y1 - h_txt - 8), (x1 + w_txt, y1), (0, 0, 255), cv2.FILLED
        )
        cv2.putText(
            frame,
            texto_det,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    texto_fps = (
        f"Inferencia: {tiempo_inferencia_ms:.1f} ms | FPS: {fps_procesamiento:.1f}"
    )
    texto_info = (
        f"Frame: {numero_frame}/{total_frames} | Detectados: {len(detecciones)}"
    )

    cv2.putText(
        frame,
        texto_fps,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        texto_info,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    writer.write(frame)

    if numero_frame % 30 == 0 or numero_frame == total_frames:
        print(f"[PROGRESO] Frame {numero_frame}/{total_frames} procesado.")

video.release()
writer.release()

promedio_ms = (
    sum(tiempos_inferencia) / len(tiempos_inferencia) if tiempos_inferencia else 0
)
promedio_fps = 1000 / promedio_ms if promedio_ms > 0 else 0

print(f"Video guardado en: {RUTA_VIDEO_SALIDA}")
print(f"Tiempo de inferencia promedio: {promedio_ms:.2f} ms")
print(f"Rendimiento del modelo: {promedio_fps:.2f} FPS")
