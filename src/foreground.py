import cv2
import numpy as np
import os

img_path = '../data/toad2.jpg' 
output_path = 'out/toad2_foreground.png'

if not os.path.exists('out'):
    os.makedirs('out')

# 1. Cargar imagen
image = cv2.imread(img_path)
if image is None:
    print(f"Error: No se encontró la imagen en {img_path}")
    exit()

# 2. Convertir a espacio de color HSV para detectar el fondo gris
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_bg = np.array([80, 20, 80])
upper_bg = np.array([120, 255, 255])

mask_bg = cv2.inRange(hsv, lower_bg, upper_bg)
mask = cv2.bitwise_not(mask_bg)

kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


final_image = cv2.bitwise_and(image, image, mask=mask)

cv2.imwrite(output_path, final_image)
print(f"Éxito: Toad 2 extraído con fondo negro en {output_path}")