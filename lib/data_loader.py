from dataclasses import dataclass
from typing import Union

import os
import xml.etree.ElementTree as ET

from PIL import Image

import numpy as np
import cv2

# TODO Improve documentation
@dataclass
class DataSample:
    """
    This dataclass represents a sample of the dataset including the file name, image resolution and expected
    segmented output.
    """
    image_path: str
    resolution: tuple[int,int]
    segmented_border: Union[list[float], list[list[float]]]
    """ [xmin,ymin,xmax,ymax] or [xmin,ymin,xmax,ymax] or [value,xmin,ymin,xmax,ymax]"""

@dataclass
class DataLocation:
    """
    This dataclass represents a dataset including the path to the files, subdirectories to the images and metadata 
    files, and the metadata file type, either xml or txt.
    """

    directory_path: str
    image_subdirectory: str
    annotation_subdirectory: str
    boundary_tag_type: str
    """ "xml_yolo", "txt_raw" """

@dataclass
class ScaledDataSample:
    name: str
    image: cv2.Mat
    segmented_border: Union[list[float], list[list[float]]]
    """ [xmin,ymin,xmax,ymax] or [xmin,ymin,xmax,ymax] or [value,xmin,ymin,xmax,ymax]"""



def verify_xml_yolo(file_path: str)->bool:
    """
    Return  False if the file contains only one tag and False otherwise.
    """
    file = ET.parse(file_path+".xml").findall('.object')
    return len(file) == 1

def get_xml_yolo(file_path: str)->list[float]:
    """
    Return a tuple with a bool value (True if the data is only one array, False in another case) , and a list with
    the boundary data ordered like this
    [xmin,ymin,xmax,ymax]
    """

    data = [0.0 for i in range(4)]

    file: ET.ElementTree = ET.parse(file_path+".xml")

    data[0] = float(file.find("object").find("bndbox").find("xmin").text)  # type: ignore
    data[1] = float(file.find("object").find("bndbox").find("ymin").text)  # type: ignore
    data[2] = float(file.find("object").find("bndbox").find("xmax").text)  # type: ignore  
    data[3] = float(file.find("object").find("bndbox").find("ymax").text)  # type: ignore

    return data

def get_multiple_xml_yolo(file_path: str)->list[list[float]]:
    """
    Return  a list of lists, each of which is a list of floats.
    Args:
        file_path (str): Path to the file

    Returns:
        list[list[float]]: A list of boundaries of . 
    """
    data = []
    data_list = ET.parse(file_path+".xml").findall('object')
    for element in data_list:
        data.append([
            float(element.find("bndbox").find("xmin").text),  # type: ignore
            float(element.find("bndbox").find("ymin").text),  # type: ignore
            float(element.find("bndbox").find("xmax").text),  # type: ignore
            float(element.find("bndbox").find("ymax").text)   # type: ignore
        ])
    return data

def verify_txt_raw(file_path: str)->bool:
    """
    pending
    """
    return isinstance(np.loadtxt(file_path+".txt")[0],float)

def get_multiple_txt_raw(file_path: str, resolution: tuple[int, int])->list[list[float]]:
    """_summary_

    Args:
        file_path (str): _description_
        resolution (tuple[int, int]): _description_

    Returns:
        list[list[float]]: _description_
    """
    data = []
    file = np.loadtxt(file_path+".txt")
    for i in file:
        data.append([
            resolution[0] * float(i[1]),
            resolution[1] * float(i[2]),
            resolution[0] * float(i[3]),
            resolution[1] * float(i[4]),
        ])
    return data

def get_txt_raw(file_path: str, resolution: tuple[int, int])->list[float]:
    """
    Return a list with the boundary data ordered like this
    [xmin,ymin,xmax,ymax]
    """

    data = [0.0 for i in range(4)]
    file = np.loadtxt(file_path+".txt")

    data[0] = resolution[0] * float(file[1])
    data[1] = resolution[1] * float(file[2])
    data[2] = resolution[0] * float(file[3])
    data[3] = resolution[1] * float(file[4])
    return data



class DataLoader:
    """_summary_

    Returns:
        _type_: _description_
    """

    one_car_data: dict[str, DataSample] = {}
    multiple_car_data: dict[str, DataSample] = {}
    plates_data = []
    counter = 0

    def __init__(self, scale_resolution: tuple[int, int], dataset_paths: list[DataLocation])->None:

        current_directory = os.getcwd() + "/"
        self.scale_resolution = scale_resolution
        name = None
        resolution = None
        segmented_border = None

        # Identify segmented images
        for data_set in dataset_paths:
            path = current_directory+ data_set.directory_path
            for file in os.listdir(path + data_set.image_subdirectory):
                if file.endswith(".jpg") or file.endswith(".png"):


                    # get image name and resolution
                    name = os.path.join(file)
                    image_path = path + data_set.image_subdirectory + name
                    resolution = Image.open(path + data_set.image_subdirectory + name).size

                    # Verify the tag format and append images description
                    if data_set.boundary_tag_type == "xml_yolo" :
                        if verify_xml_yolo(path + data_set.annotation_subdirectory + name[:-4]):
                            segmented_border = get_xml_yolo(path + data_set.annotation_subdirectory + name[:-4])
                            self.one_car_data[name] = DataSample(image_path, resolution,segmented_border)
                        else:
                            segmented_border = get_multiple_xml_yolo(path +data_set.annotation_subdirectory +name[:-4])
                            self.multiple_car_data[name] = DataSample(image_path, resolution, segmented_border)

                    elif data_set.boundary_tag_type == "txt_raw":
                        if verify_txt_raw(path + data_set.annotation_subdirectory + name[:-4]):

                            segmented_border = get_txt_raw(path +data_set.annotation_subdirectory+ name[:-4],resolution)
                            self.one_car_data[name] = DataSample(image_path, resolution,segmented_border)
                        else:
                            segmented_border = get_multiple_txt_raw(path + data_set.annotation_subdirectory + name[:-4], 
                                                                    resolution)
                            self.multiple_car_data[name] = DataSample(image_path, resolution, segmented_border)

        # Identify segmented plates



    def scale_image(self, name: str, image_type: str = "one_car")->tuple[cv2.Mat, list[int]]:

        # Load image and get size.
        if image_type == "multiple_car":
            image = cv2.imread(self.multiple_car_data[name].image_path, cv2.IMREAD_COLOR)
        else:   # "one_car"
            image = cv2.imread(self.one_car_data[name].image_path, cv2.IMREAD_COLOR)
        image_width = float(image.shape[1])
        image_height = float(image.shape[0])

        # Calculate scale factor and padding taking into account aspect ratio.
        scale_factor = 1.0
        x_padding = 0
        y_padding = 0

        if (float(self.scale_resolution[0])/image_width) <= (float(self.scale_resolution[1])/image_height):
            scale_factor = float(self.scale_resolution[0])  / image_width
            y_padding = int((self.scale_resolution[1] - int(scale_factor*image_height))/2)
        else:
            scale_factor = float(self.scale_resolution[1])  / image_height
            x_padding = int((self.scale_resolution[0] - int(scale_factor*image_width))/2)

        # Calculate new height and width of the final image.

        new_height = int(image_height * scale_factor)
        new_width = int(image_width * scale_factor)

        print("New height and width: " + str(new_width) + ", height: " + str(new_height))

        # Resize image, pad edges and resize again to fix the resolution.
        image = cv2.resize(image, (new_width, new_height))
        x_padding_offset = int(self.scale_resolution[0] - new_width - (2*x_padding))
        y_padding_offset = int(self.scale_resolution[1] - new_height - (2*y_padding))
        image = np.pad(image, ((y_padding ,y_padding + y_padding_offset), (x_padding,x_padding + x_padding_offset),
                               (0,0)),"edge" )  # type: ignore        
        image = cv2.resize(image, (self.scale_resolution[0],self.scale_resolution[1]))

        #print(background_image.shape)

        print("scale_factor: " + str(scale_factor))
        print("x_padding: " + str(x_padding)) 
        print("y_padding: " + str(y_padding))

        return image


if __name__ == '__main__':

    data_paths=[
        DataLocation("data/Car_License_Plate_Detection/", "images/", "annotations/", "xml_yolo"),
        DataLocation("data/license-plate-dataset/dataset/train/", "images/", "annots/", "xml_yolo"),
        DataLocation("data/license-plate-dataset/dataset/valid/", "images/", "annots/", "xml_yolo"),
        DataLocation("data/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano/yolo_plate_dataset/", "", "",
                     "txt_raw")
    ]

    data_loader = DataLoader((400,400),data_paths)
    # print("One Car Data---------------------------------------------------------------------")
    # for item in data_loader.one_car_data:
    #     print("->"+ str(item.resolution) + " " + str(item.segmented_border))

    # print("Multiple Car Data----------------------------------------------------------------")
    # for item in data_loader.multiple_car_data:
    #     print("->"+ str(item.resolution) + " " + str(item.segmented_border))

    print("One car len " + str(len(data_loader.one_car_data)))
    print("Multiple car len " + str(len(data_loader.multiple_car_data)))

    org_img = cv2.imread((data_loader.one_car_data[list(data_loader.one_car_data.keys())[500]]).image_path,
                         cv2.IMREAD_COLOR)
    print("Original shape:" + str(org_img.shape))
    cv2.imshow("Original shape",org_img)
    cv2.waitKey(2000)

    img = data_loader.scale_image(list(data_loader.one_car_data.keys())[500])
    print("Resized shape:" + str(np.shape(img)))


    cv2.imshow("Resized image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
