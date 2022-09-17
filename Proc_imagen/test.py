import numpy as np
import cv2 #This is openCV
import matplotlib.pyplot as plt 

#Upload images and save it in gray scale
plate1 = cv2.imread('Fig/Placa1.jpg', cv2.IMREAD_COLOR)
plate1 = cv2.cvtColor(plate1, cv2.COLOR_BGR2GRAY)
bilateral_blur = cv2.bilateralFilter(plate1,11,17,17)
edged = cv2.Canny(bilateral_blur, 30, 150)

plt.title("Canny image")
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
plt.show()

contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

location = []
for i, cnt in enumerate(contours):
  approx = cv2.approxPolyDP(cnt, 8, True)
  if(len(approx) == 4):
    location.append(approx)

print(f"Posibles opciones: {len(location)}")
location = sorted(location, key = cv2.contourArea, reverse = True)

black_background = np.zeros_like(plate1)
white_background = black_background + 255

cv2.drawContours(white_background, location, 0, (0), 1)
plt.title("Contours")
plt.imshow(cv2.cvtColor(white_background, cv2.COLOR_BGR2RGB))
plt.show()
