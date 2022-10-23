import numpy as np
import cv2 

def verify_plate(plate_zone):  
    
    plate_zone_gray = cv2.cvtColor(plate_zone, cv2.COLOR_BGR2GRAY)
    plate_zone_binary = cv2.threshold(plate_zone_gray, 127, 255, cv2.THRESH_BINARY)[1]
    h_plate, w_plate  = plate_zone_binary.shape
    
    min_height = 10
    bool_h = h_plate > min_height
    
    min_width = 50
    bool_w = w_plate > min_width
    
    height_width_relation = (h_plate / w_plate) * 100
    bool_h_w = height_width_relation > 20 
    
    plate_contours = cv2.findContours(plate_zone_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]  
    bool_contours = len(plate_contours) > 10
    
    hist = cv2.calcHist([plate_zone_binary],[0],None,[256],[0,256])
    white_pixels = hist[-1]
    black_pixels = hist[0]
    black_white_factor = 0
    
    bool_white_black = (white_pixels - black_pixels) > black_white_factor
    
    info = ""
    info += f"Plate Zone Image\nHeight: {h_plate}, Width: {w_plate}, Area: {h_plate*w_plate}\n"
    info += f"Height of plate is greater than {min_height}: {bool_h}\n"
    info += f"Width of plate is greater than {min_width}: {bool_w}\n"
    info += f"Height and width relation is {round(height_width_relation, 2)}. Valid plate: {bool_h_w}\n"
    info += f"Found contours: {len(plate_contours)}\n"
    info += f"Number of contours is greater than 10: {bool_contours}"
    info += f"Binary Image...\nWhite pixels: {white_pixels} Black pixels: {black_pixels}\n"
    info += f"White-Black-Difference is greater than {black_white_factor}: {bool_white_black}, {white_pixels - black_pixels}\n"
  
    plate_found = bool_h and bool_w and bool_h_w and bool_contours and  bool_white_black
    
    return plate_found, info