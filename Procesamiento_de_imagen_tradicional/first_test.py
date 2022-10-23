import numpy as np
import cv2 
from verify_plate_image import verify_plate

def test_plate_method_1(img):
    """
    Detects zone plate of an image using found contours 

        Parameters:
            img (ndarray): Image in BGR

        Returns:
            bool: Found plate
            ndarray: possible zone plate image
            string: message of the procedure
    """
    plate_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bilateral_blur = cv2.bilateralFilter(plate_gray,11,17,17)
    edged = cv2.Canny(bilateral_blur, 30, 150)    

    contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    location = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 8, True)
        if(len(approx) == 4):
            location.append(approx)

    location = sorted(location, key = cv2.contourArea, reverse = True)
    selected_contour = 0
    white_background = np.zeros_like(plate_gray) + 255
    cv2.drawContours(white_background, location, selected_contour, (0), 1)

    mask = np.zeros_like(plate_gray)
    cv2.drawContours(mask, location, selected_contour, 255, -1)
    cv2.bitwise_and(img, img, mask = mask)

    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    plate_zone = img[x1:x2+1, y1:y2+1]
    
    return verify_plate(plate_zone)[0], plate_zone, "Method 1\n" + verify_plate(plate_zone)[1]