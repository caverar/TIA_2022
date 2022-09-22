import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Union

from PIL import Image
import numpy as np

# TODO Improve documentation
@dataclass
class DataSample:
    """
    This dataclass represents a sample of the dataset including the file name, image resolution and expected
    segmented output.
    """

    name: str
    resolution: tuple[int,int]
    segmented_border: Union[list[float], list[list[float]]]
    """ [xmin,ymin,xmax,ymax] """

@dataclass
class DataLocation:
    """
    This dataclass represents a dataset including the path to the files, subdirectories to the images and metadata 
    files, and the metadata file type, either xml or txt.
    """

    directory_path: str
    image_subdirectory: str
    label_subdirectory: str
    boundary_tag_type: str 
    """ "xml_yolo", "txt_raw" """

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

    file = ET.parse(file_path+".xml")

    data[0] = float(file.find("object").find("bndbox").find("xmin").text)
    data[1] = float(file.find("object").find("bndbox").find("ymin").text)
    data[2] = float(file.find("object").find("bndbox").find("xmax").text)
    data[3] = float(file.find("object").find("bndbox").find("ymax").text)

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
            float(element.find("bndbox").find("xmin").text),
            float(element.find("bndbox").find("ymin").text),
            float(element.find("bndbox").find("xmax").text),
            float(element.find("bndbox").find("ymax").text)
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

    one_car_data = []
    multiple_car_data = []
    plates_data = []
    counter = 0

    def __init__(self, scale_resolution: tuple[int, int], data_paths: list[DataLocation])->None:

        current_directory = os.getcwd() + "/"
        self.scale_resolution = scale_resolution
        name = None
        resolution = None
        segmented_border = None

        # Identify segmented images
        for data_set in data_paths:
            path = current_directory+ data_set.directory_path
            for file in os.listdir(path + data_set.image_subdirectory):
                if file.endswith(".jpg") or file.endswith(".png"):


                    # get image name and resolution
                    name = os.path.join(file)
                    resolution = Image.open(path + data_set.image_subdirectory + name).size

                    # Verify the tag format and append images description
                    if data_set.boundary_tag_type == "xml_yolo" :
                        if verify_xml_yolo(path + data_set.label_subdirectory + name[:-4]):
                            segmented_border = get_xml_yolo(path + data_set.label_subdirectory + name[:-4])
                            self.one_car_data.append(DataSample(name,resolution,segmented_border))
                        else:
                            segmented_border = get_multiple_xml_yolo(path + data_set.label_subdirectory + name[:-4])
                            self.multiple_car_data.append(DataSample(name,resolution,segmented_border))

                    elif data_set.boundary_tag_type == "txt_raw":
                        if verify_txt_raw(path + data_set.label_subdirectory + name[:-4]):

                            segmented_border = get_txt_raw(path + data_set.label_subdirectory + name[:-4], resolution)
                            self.one_car_data.append(DataSample(name,resolution, segmented_border))
                        else:
                            segmented_border = get_multiple_txt_raw(path + data_set.label_subdirectory + name[:-4], 
                                                                    resolution)
                            self.multiple_car_data.append(DataSample(name,resolution, segmented_border))

        # Identify segmented plates


    # def load_image(self)->Image:
        # TODO implement
    #     pass

def main():

    data_paths=[
        DataLocation("data/Car_License_Plate_Detection/", "images/", "annotations/", "xml_yolo"),
        DataLocation("data/license-plate-dataset/dataset/train/", "images/", "annots/", "xml_yolo"),
        DataLocation("data/license-plate-dataset/dataset/valid/", "images/", "annots/", "xml_yolo"),
        DataLocation("data/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano/yolo_plate_dataset/", "", "",
                     "txt_raw")
    ]

    data_loader = DataLoader((640,480),data_paths)
    # print("One Car Data---------------------------------------------------------------------")
    # for item in data_loader.one_car_data:
    #     print("->"+ str(item.resolution) + " " + str(item.segmented_border))

    # print("Multiple Car Data----------------------------------------------------------------")
    # for item in data_loader.multiple_car_data:
    #     print("->"+ str(item.resolution) + " " + str(item.segmented_border))

    print("One car len " + str(len(data_loader.one_car_data)))
    print("Multiple car len " + str(len(data_loader.multiple_car_data)))

if __name__ == '__main__':
    main()
