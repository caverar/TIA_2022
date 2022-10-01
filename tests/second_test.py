import numpy as np
import cv2 #This is openCV
from verify_plate_image import verify_plate


def test_plate_method2(img):
    """
    Detects zone plate of an image using vertical axes, erosion, closing and counting number of 
    white pixels

        Parameters:
            img (ndarray): Image in BGR

        Returns:
            bool: Found plate
            ndarray: possible zone plate image
            string: message of the procedure
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bilateral_blur = cv2.bilateralFilter(img_gray,11,17,17)
    edged = cv2.Canny(bilateral_blur, 240, 250)

    kernel_v_line = np.ones((5, 1), np.uint8)
    
    erode_v = cv2.erode(edged, kernel_v_line) 

    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(erode_v, cv2.MORPH_CLOSE, kernel)

    processed_image = cv2.erode(closing, kernel_v_line) 
    
    white_in_rows = np.zeros(processed_image.shape[0])
    for i, row in enumerate(processed_image):
        for value in row:
            if value == 255: white_in_rows[i]+=1
            
    white_in_columns = np.zeros(processed_image.shape[1])
    for i in range(processed_image.shape[1]):
        for value in processed_image[:, i]:
            if value == 255: white_in_columns[i]+=1

    min_white_pixels_columns = 20
    min_white_pixels_rows = 6
    selected_rows = [i for i, row in enumerate(white_in_rows) if row > min_white_pixels_rows]
    selected_columns = [i for i, column in enumerate(white_in_columns) if column > min_white_pixels_columns]

    try:
        row_security_factor = 13
        column_security_factor = 3

        x1 = min(selected_rows)
        x2 = max(selected_rows)
        y1 = min(selected_columns)
        y2 = max(selected_columns)

        if x1 - row_security_factor > 0: x1 -= row_security_factor
        if y1 - column_security_factor > 0: y1 -= column_security_factor
        if x2 + row_security_factor > 0: x2 += row_security_factor
        if y2 + column_security_factor > 0: y2 += column_security_factor

        plate_zone2 = img[x1:x2, y1:y2]

        f, m = verify_plate(plate_zone2)

        return f, plate_zone2, "Method 2\n" + m
    except:
        return False, img, "No selected rows or columns"
