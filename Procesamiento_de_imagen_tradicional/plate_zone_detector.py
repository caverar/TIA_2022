import numpy as np
import cv2 #This is openCV
from first_test import test_plate_method_1
from second_test import test_plate_method2

def detect_plate(img):
  """
  Detects zone plate from an image using two different methods (contours and vertical boundaries). 
  If zone plate is not found, original image is returned

  Parameters:
      img (ndarray): image in BGR

  Returns:
      bool: Found plate
      ndarray: possible zone plate
      string: info of plate verification process

  """
  b1, i1, m1 = test_plate_method_1(img)
  if not b1:
    return test_plate_method2(img)
  return b1, i1, m1


