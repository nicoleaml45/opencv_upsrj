import cv2
import numpy as np
import os

# Load video file
input_video = "../data/peces.mp4"
output_folder = "../out"
output_path = os.path.join(output_folder, 'peces_background.mp4')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Error: No se pudo abrir el archivo de video. Verifica que esté en la carpeta 'data'.")
    exit()
    
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False) #movimiento del agua

print("Procesando video de peces... Presiona 'Esc' para detener.")

while True:
    ret, frame = cap.read()
    if not ret:
        break # fin del video

    fgmask = fgbg.apply(frame)

    # morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    # guardar en el archivo y mostrar
    out.write(fgmask)
    cv2.imshow('Peces background', fgmask)

    if cv2.waitKey(30) & 0xFF == 27:
        break

# liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"¡Proceso terminado! Video guardado en: {output_path}")