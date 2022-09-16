from sys import platform
import sys
import os
import shutil

sys.path.insert(1, os.getcwd() + "/" + "/lib/termcolor/src/termcolor")
from termcolor import colored
from lib.gdrivedl import gdrivedl as gdl


data_download_link = "https://drive.google.com/drive/folders/1iL811t_-eqnuNwVBGeeU-HG3k6Whd3U9"

folder_list = {
    "Car_License_Plate_Detection.zip": "data/Car_License_Plate_Detection",
    "yolo_plate_dataset.zip": "data/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano",
    "yolo_plate_ocr_dataset.zip": "data/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano",
}  
""" List of downloadable files with their respective save path """

def main():

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
    if(must_download): 
        print(colored("Downloading...","cyan"))
        gdl.main([data_download_link])  # TODO Add directory_prefix
    else:
        print(colored("Files already downloaded, skipping download","yellow"))

    # Create folders
    print("Creating folders...")
    for path in folder_list.values():
        if not os.path.isdir(current_path+path):
            os.makedirs(current_path+path)

    # Extract zip files
    print(colored("Extracting zip files...","cyan"))
    for file in folder_list:
        shutil.unpack_archive(current_path+file, current_path+folder_list[file])

    # Delete zip files
    print(colored("Deleting zip files...","cyan"))
    for file in folder_list:
        if os.path.isfile(current_path+file):
            os.remove(current_path+file)

    print(colored("Done!","green"))

if __name__ == "__main__":
    main()
