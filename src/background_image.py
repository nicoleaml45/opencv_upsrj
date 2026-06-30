import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo acceder a la webcam.")
    exit()

_, img = cap.read()
averageValue1 = np.float32(img)

print("Presiona 's' para guardar la imagen actual en '../data/background_image.jpg'.")
print("Presiona 'ESC' para salir y guardar el fondo en '../out/background_image.jpg'.")

try:
    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        # Actualizar modelo de fondo
        cv2.accumulateWeighted(img, averageValue1, 0.02)
        resultingFrames1 = cv2.convertScaleAbs(averageValue1)

        cv2.imshow('Original Frame', img)
        cv2.imshow('Background (Running Average)', resultingFrames1)
        
        key = cv2.waitKey(30) & 0xFF
        
        # Guardar frame actual en 'data' al presionar 's'
        if key == ord('s'):
            cv2.imwrite('../data/background_image.jpg', img)
            print("Foto capturada y guardada en '../data/background_image.jpg'")
            
        # Salir y guardar fondo en 'out' al presionar 'ESC'
        elif key == 27:
            cv2.imwrite('../out/background_image.jpg', resultingFrames1)
            print("Imagen de fondo guardada en '../out/background_image.jpg'")
            break
            
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Cámara liberada y proceso finalizado.")