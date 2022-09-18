import os
from dataclasses import dataclass
from PIL import Image

@dataclass
class DataSample:
    """This dataclass represents a sample of the dataset including path to the file, image resolution and expected 
    segmented output."""

    path: str
    resolution: list[int]
    segmented_border: list[int]
    """ [xmin,ymin,xmax,ymax] """

@dataclass
class DataLocation:
    """This dataclass represents dataset ibncluding."""

    directory_path: str
    image_subdirectory: str 
    output_subdirectory: str
    output_type: str 
    """ "xml_yolo", "txt_raw" """
    



class DataLoader:
    #data: list[DataSample]
   
    def __init__(self, data_paths: list[DataLocation] = DataLocation("data/Car_License_Plate_Detection/", 
                                                                     "images", "annots", "xml_yolo"))->None:
        current_directory = os.getcwd()

        # Identify the file names
        # TODO everything
        pass
        
        
    def get_xml_yolo(self, path: str)->str:
        #TODO implement
        pass
    
    def txt_raw(self, path: str)->str:
        #TODO implement
        pass

    def load_image(self)->Image:
        # TODO implement
        pass
    
    def scale_dataset(self, resolution: list[int] = [640,480])->None:
        # TODO implement
        pass
    
def main():
     
    data_paths=[
        DataLocation("data/Car_License_Plate_Detection/", "images", "annots", "xml_yolo"),
        DataLocation("data/license-plate-detection/dataset/train/", "images", "annots", "xml_yolo"),
        DataLocation("data/license-plate-detection/dataset/valid/", "images", "annots", "xml_yolo"),
        DataLocation("data/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano/yolo_plate_dataset/",
                     "", "", "txt_raw")
    ]

    data_loader = DataLoader()


if __name__ == '__main__':
    main()
