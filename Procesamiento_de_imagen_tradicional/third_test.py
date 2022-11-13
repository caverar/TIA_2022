import numpy as np
import cv2 
from verify_plate_image import verify_plate

def test_plate_method3(img):
    """
    Detects white zone plate of an image using thresholds and found contours 

        Parameters:
            img (ndarray): Image in BGR

        Returns:
            bool: Found plate
            ndarray: possible zone plate image
            string: message of the procedure
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thres = cv2.threshold(img_gray, 170, 255, cv2.THRESH_BINARY_INV)[1]

    contours = cv2.findContours(thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    ratio = 3.07692307692
    min_w = 80
    max_w = 110
    min_h = 25
    max_h = 52 
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if (np.isclose(aspect_ratio, ratio, atol=0.7) and (max_w > w > min_w) and
        (max_h > h > min_h)):
            candidates.append(cnt)  
    try:
        ys = []
        for cnt in candidates:
            x, y, w, h = cv2.boundingRect(cnt)
            ys.append(y)
        selected_contour = candidates[np.argmax(ys)]

        x, y, w, h = cv2.boundingRect(selected_contour)
        plate_zone = img[y:y+h,x:x+w]

        return verify_plate(plate_zone)[0], plate_zone, "METHOD 3\n" + verify_plate(plate_zone)[1]
    except:
        return False, img, "METHOD 3\n" + "Not selected contours"