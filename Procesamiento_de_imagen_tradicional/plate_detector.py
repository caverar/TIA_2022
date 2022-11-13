from first_test import test_plate_method1
from second_test import test_plate_method2
from third_test import test_plate_method3

def detect_plate(img):  
    b, i, m = test_plate_method1(img)
    if b:
        return b, i, m
    else:
        b, i, m = test_plate_method3(img)
        if b:
            return b, i, m
        else:
            b, i, m = test_plate_method2(img)
            if b:
                return b, i, m
            else:
                return False, img, "Plate not found with any method"