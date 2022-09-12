# Pack 1 license-plate-dataset: https://github.com/RobertLucian/license-plate-dataset
# Pack 2 Car_License_Plate_Detection: https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
# Pack 3 Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano:  https://github.com/winter2897/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano

from inspect import getmembers
import os
import numpy as np
from PIL import Image

import xml.etree.ElementTree as ET



# Pack1

pack1_path ="data/license-plate-dataset/"

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



def main():
    # Load Pack 1:
    pack1_train_images_paths = []
    pack1_valid_images_paths = []
    
    for file in os.listdir("data/license-plate-dataset/dataset/train/images"):
        if file.endswith(".jpg"):
            pack1_train_images_paths.append(os.path.join(file))    
    for file in os.listdir("data/license-plate-dataset/dataset/valid/images"):
        if file.endswith(".jpg"):
            pack1_valid_images_paths.append(os.path.join(file))

    print(pack1_valid_images_paths[0])
    print(pack1_valid_get_data(pack1_valid_images_paths[0]))
    

    # Load Pack 2:

    # Load Pack 3:


if __name__ == "__main__":
    main()
