from sys import platform
import os
import shutil

from termcolor import colored
from gdrivedl import gdrivedl as gdl


DATA_DOWNLOAD_LINK = "https://drive.google.com/drive/folders/1iL811t_-eqnuNwVBGeeU-HG3k6Whd3U9"

folder_list = {
    "Car_License_Plate_Detection.zip": "data/Car_License_Plate_Detection",
    "yolo_plate_dataset.zip": "data/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano",
    "yolo_plate_ocr_dataset.zip": "data/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano",
    "plateless_cars_dataset.zip": "data/plateless_cars",
    "unlabeled_plates_dataset.zip": "data/unlabeled_plates",
    "Dataset_Tradicional.zip": "data/Dataset_Tradicional",
    "plate_detector.zip": "models",
    "ResNet50Box_01.zip": "models"
}
""" List of downloadable files with their respective save path. """


def main():
    """
    Main function.
    """

    current_path = os.getcwd() + "/"

    if platform =="'Windows":
        os.system('color')
    elif platform == "Linux":
        pass

    print(colored("Welcome!","green"))

    # Verify if the download is necessary
    must_download = False
    for file in folder_list:
        if not os.path.isfile(current_path + file):
            must_download = True

    # Download zip files
    if must_download:
        print(colored("Downloading...","cyan"))
        gdl.main([DATA_DOWNLOAD_LINK])
    else:
        print(colored("Files already downloaded, skipping download","yellow"))

    # Create folders
    print("Creating folders...")
    for path in folder_list.values():
        if not os.path.isdir(current_path+path):
            os.makedirs(current_path+path)

    # Extract zip files
    print(colored("Extracting zip files...","cyan"))
    for file in folder_list.items():
        shutil.unpack_archive(current_path + file[0], current_path + file[1])

    # Delete zip files
    print(colored("Deleting zip files...","cyan"))
    for file in folder_list:
        if os.path.isfile(current_path+file):
            os.remove(current_path+file)

    print(colored("Done!","green"))

if __name__ == "__main__":
    main()
