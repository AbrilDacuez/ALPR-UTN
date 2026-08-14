import cv2

imagen = cv2.imread("auto.jpg")
alto, ancho, canales = imagen.shape

print(f"La imagen original mide: {ancho} pixeles de ancho x {alto} píxeles de alto.")
print(f"Tiene {canales} canales de color (BGR).")

imagen_grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
cv2.putText(
    imagen_grises,
    "ETAPA =: PROCESADA OK",
    (20, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    255,
    2,
)
nombre_archivo_salida = "auto_procesado.jpg"
cv2.imwrite(nombre_archivo_salida, imagen_grises)
print(f"\n[OK] Imagen modificada guardada con éxito como: '{nombre_archivo_salida}'")
