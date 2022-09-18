import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from PIL import Image
import numpy as np

@dataclass
class DataSample:
    """
    This dataclass represents a sample of the dataset including the file name, image resolution and expected 
    segmented output.
    """

    name: str
    resolution: list[int]
    segmented_border: list[float]
    """ [xmin,ymin,xmax,ymax] """

@dataclass
class DataLocation:
    """
    This dataclass represents a dataset including the path to the files, subdirectories to the images and metadata 
    files, and the metadata file type, either xml or txt.
    """

    directory_path: str
    image_subdirectory: str
    output_subdirectory: str
    output_type: str 
    """ "xml_yolo", "txt_raw" """


class DataLoader:
    one_car_data = []
    counter = 0
   
    def __init__(self, data_paths: list[DataLocation] = [DataLocation("data/Car_License_Plate_Detection/", 
                                                                     "images/", "annots/", "xml_yolo")])->None:
        current_directory = os.getcwd() + "/"
        name = None
        resolution = None
        segmented_border = None
        # Identify the file names
        for data_set in data_paths:
            path = current_directory+ data_set.directory_path
            for file in os.listdir(path + data_set.image_subdirectory):
                if file.endswith(".jpg") or file.endswith(".png"):

                    segmented_flag = True   # True if the boundary date is correct
                    
                    name = os.path.join(file)
                    resolution = Image.open(path + data_set.image_subdirectory + name).size
                    
                    #print(path + data_set.image_subdirectory + name)
                    #print(resolution)
                    
                    if(data_set.output_type == "xml_yolo"):
                        segmented_border = self.get_xml_yolo(path + data_set.output_subdirectory + name[:-4])
                        
                    elif(data_set.output_type == "txt_raw"):
                        segmented_flag, segmented_border = self.get_txt_raw(path + data_set.output_subdirectory +
                                                                            name[:-4], resolution)
                    #print(segmented_border)
                    
                    if(segmented_flag):
                        self.one_car_data.append(DataSample(name,list(resolution),segmented_border))
                    else:
                        self.counter = self.counter + 1
                        
        # TODO everything
        pass


    def get_xml_yolo(self, path: str)->list[float]:
        """
        Return a list with the boundary data ordered like this
        [xmin,ymin,xmax,ymax]
        """

        data = [0.0 for i in range(4)]
        
        file = ET.parse(path+".xml")
        data[0] = float(file.find("object").find("bndbox").find("xmin").text)
        data[1] = float(file.find("object").find("bndbox").find("ymin").text)
        data[2] = float(file.find("object").find("bndbox").find("xmax").text)
        data[3] = float(file.find("object").find("bndbox").find("ymax").text)

        return data

    def get_txt_raw(self, path: str, resolution: tuple[int, int] )->list[float]:
        """
        Return a list with the boundary data ordered like this
        [xmin,ymin,xmax,ymax]
        """

        data = [0.0 for i in range(4)]


        file = np.loadtxt(path+".txt")

        if(type(file[0]) == np.float64):
            data[0] = resolution[0] * float(file[1])
            data[1] = resolution[1] * float(file[2])
            data[2] = resolution[0] * float(file[3])
            data[3] = resolution[1] * float(file[4])
            return (True, data)
        else:
            return (False, data)

    def load_image(self)->Image:
        # TODO implement
        pass
    
    def scale_dataset(self, resolution: list[int] = [640,480])->None:
        # TODO implement
        pass
    
def main():
     
    data_paths=[
        DataLocation("data/Car_License_Plate_Detection/", "images/", "annotations/", "xml_yolo"),
        DataLocation("data/license-plate-dataset/dataset/train/", "images/", "annots/", "xml_yolo"),
        DataLocation("data/license-plate-dataset/dataset/valid/", "images/", "annots/", "xml_yolo"),
        DataLocation("data/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano/yolo_plate_dataset/", "", "", 
                     "txt_raw")
    ]

    data_loader = DataLoader(data_paths)
    for item in data_loader.one_car_data:
        print("->"+ str(item.resolution) + " " + str(item.segmented_border)) 
    print(data_loader.counter)
    print(len(data_loader.one_car_data))

if __name__ == '__main__':
    main()
