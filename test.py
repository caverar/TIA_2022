import cv2
import numpy as np

img = cv2.imread("data/Car_License_Plate_Detection/images/Cars0.png", cv2.IMREAD_UNCHANGED)

num_rows, num_cols = img.shape[:2]   

translation_matrix = np.float32([ [1,0,7], [0,1,11] ])   
img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))   
cv2.imshow('Translation', img_translation)    
cv2.waitKey()



