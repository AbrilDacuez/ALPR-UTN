import time
import cv2
from ultralytics import RTDETR

RUTA_IMAGEN = "Etapa 2/inputs/autos.jpg"
RUTA_SALIDA = "Etapa 2/outputs/resultado_imagen.jpg"

UMBRAL_CONFIANZA = 0.50
CLASES_INTERES = [2, 3, 5, 7]  # 2=auto, 3=moto, 5=colectivo/autobús, 7=camión

NOMBRES_ESPANOL = {
    "car": "Auto",
    "motorcycle": "Moto",
    "bus": "Colectivo",
    "truck": "Camion",
}

print("[INFO] Cargando modelo RT-DETR...")
modelo = RTDETR("rtdetr-l.pt")
imagen = cv2.imread(RUTA_IMAGEN)

if imagen is None:
    raise FileNotFoundError(f"No se encontró la imagen en: {RUTA_IMAGEN}")

tiempo_inicio = time.perf_counter()

resultados = modelo.predict(
    source=imagen, conf=UMBRAL_CONFIANZA, classes=CLASES_INTERES, verbose=False
)

tiempo_fin = time.perf_counter()

tiempo_inferencia_ms = (tiempo_fin - tiempo_inicio) * 1000
print(f"[INFO] Tiempo de procesamiento: {tiempo_inferencia_ms:.2f} ms")

detecciones = resultados[0].boxes
cantidad_objetos = len(detecciones)
print(f"[INFO] Se detectaron {cantidad_objetos} vehículos")

for caja in detecciones:
    x1, y1, x2, y2 = map(int, caja.xyxy[0])
    confianza = float(caja.conf[0])
    id_clase = int(caja.cls[0])
    nombre_ingles = modelo.names[id_clase]
    etiqueta = NOMBRES_ESPANOL.get(nombre_ingles, nombre_ingles)
    cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 0, 255), 2)
    texto = f"{etiqueta}: {confianza:.2f}"
    (ancho_texto, alto_texto), _ = cv2.getTextSize(
        texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    )
    cv2.rectangle(
        imagen,
        (x1, y1 - alto_texto - 10),
        (x1 + ancho_texto, y1),
        (0, 0, 255),
        cv2.FILLED,
    )
    cv2.putText(
        imagen,
        texto,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

cv2.imwrite(RUTA_SALIDA, imagen)
print(f"[OK] Imagen procesada guardada en: '{RUTA_SALIDA}'")
