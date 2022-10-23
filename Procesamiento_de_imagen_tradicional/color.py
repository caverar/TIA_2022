import numpy as np
import cv2
import matplotlib.pyplot as plt

# Cargamos la imagen
img_original = cv2.imread('Procesamiento_de_imagen_tradicional/Fig/Placa2.jpg')
img = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

# Convertimos la imagen a HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

# Definimos los colores que queremos detectar   
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

print(np.max(h), np.min(h))

for i in range(h.shape[0]):
    for j in range(h.shape[1]):
        if h[i,j] < 90 or h[i, j] > 110:
            img[i,j] = 0

# Mostramos la imagen
plt.subplot(2,2,1)
plt.imshow(img_original)

plt.subplot(2,2,2)
plt.imshow(img)

plt.subplot(2,1,2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cmap='gray')

plt.show()