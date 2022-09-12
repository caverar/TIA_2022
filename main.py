# Pack 1 license-plate-dataset: https://github.com/RobertLucian/license-plate-dataset
# Pack 2 Car_License_Plate_Detection: https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
# Pack 3 Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano:  
#                               https://github.com/winter2897/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano

import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

pack1_path ="data/license-plate-dataset/"
pack2_path ="data/Car_License_Plate_Detection/"
pack3_path ="data/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano/yolo_plate_dataset/"

# Pack1

def pack1_train_get_data(path: str)->list:
    """
    Return a list with the boundary data ordered like this
    [xmin,ymin,xmax,ymax]
    """
    
    data = []
    file = ET.parse(pack1_path+"dataset/train/annots/"+path[:-3]+"xml")
    
    data.append(float(file.find("object").find("bndbox").find("xmin").text))
    data.append(float(file.find("object").find("bndbox").find("ymin").text))
    data.append(float(file.find("object").find("bndbox").find("xmax").text))
    data.append(float(file.find("object").find("bndbox").find("ymax").text))

    return data

# Pack2

def pack1_valid_get_data(path: str)->list:
    """
    Return a list with the boundary data ordered like this
    [xmin,ymin,xmax,ymax]
    """
    
    data = []
    file = ET.parse(pack1_path+"dataset/valid/annots/"+path[:-3]+"xml")
    
    data.append(float(file.find("object").find("bndbox").find("xmin").text))
    data.append(float(file.find("object").find("bndbox").find("ymin").text))
    data.append(float(file.find("object").find("bndbox").find("xmax").text))
    data.append(float(file.find("object").find("bndbox").find("ymax").text))

    return data

def pack2_train_get_data(path: str)->list:
    """
    Return a list with the boundary data ordered like this
    [xmin,ymin,xmax,ymax]
    """
    
    data = []
    file = ET.parse(pack2_path+"annotations/"+path[:-3]+"xml")
    
    data.append(int(file.find("object").find("bndbox").find("xmin").text))
    data.append(int(file.find("object").find("bndbox").find("ymin").text))
    data.append(int(file.find("object").find("bndbox").find("xmax").text))
    data.append(int(file.find("object").find("bndbox").find("ymax").text))
    return data

# Pack3

def pack3_train_get_data(path: str)->list:
    """
    Return a list with the boundary data ordered like this
    [xmin,ymin,xmax,ymax]
    """
    
    data = []

    file = np.loadtxt( pack3_path+path[:-3]+"txt")
    print(file)
    data.append(file[1])
    data.append(file[2])
    data.append(file[3])
    data.append(file[4])
    return data

def main():
    # Load Pack 1:
    pack1_train_images_paths = []
    pack1_valid_images_paths = []
    
    for file in os.listdir(pack1_path+"dataset/train/images"):
        if file.endswith(".jpg"):
            pack1_train_images_paths.append(os.path.join(file))    
    for file in os.listdir(pack1_path+"dataset/valid/images"):
        if file.endswith(".jpg"):
            pack1_valid_images_paths.append(os.path.join(file))

    print(pack1_valid_images_paths[0])
    print(pack1_valid_get_data(pack1_valid_images_paths[0]))
    

    # Load Pack 2:
    pack2_train_images_paths = []
    for file in os.listdir(pack2_path+"images"):
        if file.endswith(".png"):
            pack2_train_images_paths.append(os.path.join(file)) 
    print(pack2_train_images_paths[0])
    print(pack2_train_get_data(pack2_train_images_paths[0]))

    # Load Pack 3:
    pack3_train_images_paths = []
    for file in os.listdir(pack3_path):
        if file.endswith(".jpg"):
            pack3_train_images_paths.append(os.path.join(file)) 
    print(pack3_train_images_paths[0])
    print(pack3_train_get_data(pack3_train_images_paths[0]))

if __name__ == "__main__":
    main()
