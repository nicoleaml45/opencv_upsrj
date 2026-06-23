import cv2
import numpy as np
import os

img_ref_path = "../data/nenuks.jpeg"
img_chueca_path = "../data/nenuks_chuecas.jpeg"
output_folder = "../out"
output_path = os.path.join(output_folder, 'output.jpeg')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# cargar imágenes
img_ref_color = cv2.imread(img_ref_path)
img_chueca_color = cv2.imread(img_chueca_path)

if img_ref_color is None or img_chueca_color is None:
    print("Error: Revisa que las rutas de las imágenes sean correctas.")
    exit()

# convertir a escala de grises
img_ref = cv2.cvtColor(img_ref_color, cv2.COLOR_BGR2GRAY)
img_chueca = cv2.cvtColor(img_chueca_color, cv2.COLOR_BGR2GRAY)
height, width = img_ref.shape[:2] 

# crear detector ORB
orb_detector = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_HARRIS_SCORE)
kp1, d1 = orb_detector.detectAndCompute(img_chueca, None)
kp2, d2 = orb_detector.detectAndCompute(img_ref, None)

# emparejar características
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = list(matcher.match(d1, d2))

if len(matches) < 10:
    print("No se encontraron suficientes puntos en común.")
    exit()

# ordenar y filtrar
matches.sort(key=lambda x: x.distance)
matches = matches[:int(len(matches) * 0.9)]
no_of_matches = len(matches)

# extraer coordenadas
p_chueca = np.zeros((no_of_matches, 2))
p_ref = np.zeros((no_of_matches, 2))

for i in range(len(matches)):
    p_chueca[i, :] = kp1[matches[i].queryIdx].pt
    p_ref[i, :] = kp2[matches[i].trainIdx].pt

# calcular homografía 
homography, mask = cv2.findHomography(p_chueca, p_ref, cv2.RANSAC, 3.0)
transformed_img = cv2.warpPerspective(img_chueca_color, homography, (width, height))

# guardar resultado
cv2.imwrite(output_path, transformed_img)
print(f"¡Éxito! La imagen enderezada se guardó en: {output_path}")