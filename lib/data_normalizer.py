from dataclasses import dataclass
import os
import shutil
import xml.etree.ElementTree as ET
import csv

import numpy as np
import cv2
import matplotlib.pyplot as plt
from termcolor import colored

@dataclass
class DataSetLocation:
    """
    This dataclass represents a dataset including the path to the files, subdirectories to the images and annotations
    files, and the annotations format type, either plate_xml_voc, plate_txt_yolo, ocr_txt_yolo etc.
    """
    dataset_id: int
    """ Use to classify data for specific usage cases."""
    directory_path: str
    image_subdirectory_path: str
    annotation_subdirectory_path: str
    annotation_format: str
    """ "plate_xml_voc", "plate_txt_yolo" , "ocr_txt_yolo" """


@dataclass
class DataSample:
    """
    This dataclass represents a sample of a dataset including file path, resolution and expected output.
    """

    image_path: str
    #number_of_boxes: int
    expected_output: list[list[float]]
    """[[tag, xmin, ymin, xmax, ymax]]"""
    dataset_id: int


def get_number_of_boxes_xml_voc(file_path: str)->int:
    """
    This functions looks for the number of tags present in a xml file a return the number as an integer.
    """

    return len (ET.parse(file_path+".xml").findall('.object'))


def get_number_of_boxes_txt_yolo(file_path: str)->int:
    """
    This functions looks for the number of tags present in a txt file a return the number as an integer.
    """

    data = np.loadtxt(file_path+".txt")
    if isinstance(data[0], float):
        return 1
    else:
        return len(data)


def load_xml_voc(file_path: str, resolution: tuple[int, int])->list[list[float]]:
    """
    This function load the data from a xml file and returns a list of data entries.
    """

    data = []
    data_list = ET.parse(file_path+".xml").findall('object')
    for element in data_list:
        xmin = float(element.find("bndbox").find("xmin").text)  # type: ignore
        ymin = float(element.find("bndbox").find("ymin").text)  # type: ignore
        xmax = float(element.find("bndbox").find("xmax").text)  # type: ignore
        ymax = float(element.find("bndbox").find("ymax").text)  # type: ignore

        if (0 <= xmin <= 1) and (0 <= xmax <= 1) and (0 <= ymin <= 1) and (0 <= ymax <= 1):
            data.append([0, resolution[0] * xmin, resolution[1] * ymin, resolution[0] * xmax, resolution[1] * ymax])
        else:
            data.append([0, xmin, ymin, xmax, ymax])
    return data


def load_txt_yolo(file_path: str, resolution: tuple[int, int])->list[list[float]]:
    """
    This function load the data from a txt file and returns a list of data entries.
    """

    data = []
    file = np.loadtxt(file_path+".txt")

    # Verify the number of elements
    if get_number_of_boxes_txt_yolo(file_path) > 1:
        for entry in file:

            # Convert yolo format to voc
            tag    = float(entry[0])
            x_1    = float(entry[1])
            y_1    = float(entry[2])
            w_size = float(entry[3])
            h_size = float(entry[4])
            xmin = x_1 - (w_size/2)
            ymin = y_1 - (h_size/2)
            xmax = xmin + w_size
            ymax = ymin + h_size

            # Verify if the data is normalized
            if (0 <= xmin <= 1) and (0 <= xmax <= 1) and (0 <= ymin <= 1) and (0 <= ymax <= 1):
                data.append([tag, resolution[0] * xmin, resolution[1] * ymin, resolution[0] * xmax,
                             resolution[1] * ymax])
            else:
                data.append([tag, xmin, ymin, xmax, ymax])

    else:
        # Convert yolo format to voc
        tag    = float(file[0])
        x_1    = float(file[1])
        y_1    = float(file[2])
        w_size = float(file[3])
        h_size = float(file[4])
        xmin   = x_1 - (w_size/2)
        ymin   = y_1 - (h_size/2)
        xmax   = xmin + w_size
        ymax   = ymin + h_size

        # Verify if the data is normalized
        if (0 <= xmin <= 1) and (0 <= xmax <= 1) and (0 <= ymin <= 1) and (0 <= ymax <= 1):
            data.append([tag, resolution[0] * xmin, resolution[1] * ymin, resolution[0] * xmax, resolution[1] * ymax])
        else:
            data.append([tag, xmin, ymin, xmax, ymax])

    return data


def resize_image(image_path: str, boxes_data: list[list[float]],
                 desired_resolution: tuple[int, int]) -> tuple[cv2.Mat, list[list[float]]]:
    """
    Resizes the image and his boxes positions to the desired resolution
    :param image_path: The path to the image.
    :param boxes_data: The data of the boxes.
    :param desired_resolution: The desired resolution.
    """

    # Load image and get size.
    (image_width,image_height) = (cv2.imread(image_path).shape[::-1])[1::]
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Calculate scale factor and padding taking into account aspect ratio.
    scale_factor = 1.0
    x_padding = 0
    y_padding = 0
    if (float(desired_resolution[0]) / image_width) < (float(desired_resolution[1]) / image_height):
        scale_factor = float(desired_resolution[0]) / image_width
        y_padding = int((desired_resolution[1] - int(scale_factor * image_height)) / 2)
    else:
        scale_factor = float(desired_resolution[1]) / image_height
        x_padding = int((desired_resolution[0] - int(scale_factor * image_width)) / 2)

    # Calculate new height and width of the final image.
    new_height = int(image_height * scale_factor)
    new_width = int(image_width * scale_factor)

    # Resize image, pad edges and resize again to fix the resolution.
    image = cv2.resize(image, (new_width, new_height))
    x_padding_offset = int(desired_resolution[0] - new_width - (2*x_padding))
    y_padding_offset = int(desired_resolution[1] - new_height - (2*y_padding))
    # *Argument minimum establish the values used to pad the image, use "edge" to replicate the last pixel.
    image = np.pad(image, ((y_padding ,y_padding + y_padding_offset), (x_padding,x_padding + x_padding_offset),
                           (0,0)),"minimum" )  # type: ignore
    image = cv2.resize(image, (desired_resolution[0],desired_resolution[1]))

    # Scale Boxes
    scaled_boxes = boxes_data
    for entry in scaled_boxes:
        entry[1] = (scale_factor * entry[1]) + x_padding # xmin
        entry[2] = (scale_factor * entry[2]) + y_padding # ymin
        entry[3] = (scale_factor * entry[3]) + x_padding # xmax
        entry[4] = (scale_factor * entry[4]) + y_padding # ymax

    # Show image
    # print("image_width: " + str(image_width) + ", image_height: " + str(image_height))
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap="gray")
    # plt.show()

    return (image, scaled_boxes)


def draw_image_with_boxes(image: cv2.Mat, boxes:list[list[float]], border: int = 10) -> None:
    """
    Draw the boundary boxes on the image.
    """
    for box in boxes:
        cv2.rectangle(image, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), (0,255,255), border)

    plt.imshow(image, cmap="gray")
    plt.show()


class DataNormalizer():
    """_summary_
    Ths class loads all the dataset images, normalize their resolution and annotations and then allow to save rescaled
    images and generate csv annotations with specific datasets.
    """

    plates_data_array: list[DataSample] = []
    ocr_data_array: list[DataSample] = []
    plates_desired_resolution = 0
    ocr_desired_resolution = 0
    normalized_data_path = ""

    def __init__(self, dataset_locations: list[DataSetLocation]) -> None:
        print(colored("Loading the annotations...","yellow"))

        current_directory = os.getcwd() + "/"

        for dataset in dataset_locations:
            dataset_path = current_directory + dataset.directory_path
            dataset_id = dataset.dataset_id
            for image in os.listdir(dataset_path + dataset.image_subdirectory_path):
                if image.endswith(".jpg") or image.endswith(".png"):

                    # get image path and resolution
                    file_name = os.path.join(image)
                    image_path = dataset_path + dataset.image_subdirectory_path + file_name
                    annotation_path = dataset_path + dataset.annotation_subdirectory_path + file_name[:-4]
                    resolution = (cv2.imread(image_path).shape[::-1])[1::]

                    # Load the annotations
                    if dataset.annotation_format == "plate_xml_voc":
                        expected_output = load_xml_voc(annotation_path, resolution)
                        self.plates_data_array.append(DataSample(image_path, expected_output, dataset_id))

                    elif dataset.annotation_format == "plate_txt_yolo":
                        expected_output = load_txt_yolo(annotation_path, resolution)
                        self.plates_data_array.append(DataSample(image_path, expected_output, dataset_id))

                    elif dataset.annotation_format == "untagged_plate":
                        expected_output = [[0.0,0.0,0.0,0.0,0.0]]
                        self.plates_data_array.append(DataSample(image_path, expected_output, dataset_id))

                    elif dataset.annotation_format == "ocr_txt_yolo":
                        expected_output = load_txt_yolo(annotation_path, resolution)
                        self.ocr_data_array.append(DataSample(image_path, expected_output, dataset_id))

                    else:
                        raise Exception("\"" + dataset.annotation_format +"\" format not supported")

        print(colored("Annotations loaded","green"))


    def save_images(self, _plates_desired_resolution: tuple[int,int], _ocr_desired_resolution: tuple[int,int],
                    _normalized_data_path: str = "normalized_data") -> None:
        """
        This function saves all the images rescaled to an specific resolution, resulting in two folders for both plates
        dataset and ocr dataset.
        """
        self.plates_desired_resolution = _plates_desired_resolution
        self.ocr_desired_resolution = _ocr_desired_resolution
        self.normalized_data_path = _normalized_data_path

        current_path = os.getcwd() + "/"
        print(colored("Saving normalized images...","yellow"))

        # remove pre existing folder
        if os.path.isdir(current_path + self.normalized_data_path):
            shutil.rmtree(current_path + self.normalized_data_path)

        # Create sub folders
        os.makedirs(current_path + self.normalized_data_path + "/plates_images")
        os.makedirs(current_path + self.normalized_data_path + "/ocr_images")
        image_name = 0
        for sample in self.plates_data_array:
            (image_file, sample.expected_output) = resize_image(sample.image_path, sample.expected_output,
                                                                self.plates_desired_resolution)
            cv2.imwrite(current_path + self.normalized_data_path + "/plates_images/" + str(image_name) + ".jpg",
                        image_file)
            sample.image_path = current_path + self.normalized_data_path + "/plates_images/" + str(image_name) + ".jpg"
            image_name += 1

        for sample in self.ocr_data_array:
            (image_file, sample.expected_output) = resize_image(sample.image_path, sample.expected_output,
                                                                self.plates_desired_resolution)
            cv2.imwrite(current_path + self.normalized_data_path + "/ocr_images/" + str(image_name) + ".jpg", 
                        image_file)
            sample.image_path = current_path + self.normalized_data_path + "/ocr_images/" + str(image_name) + ".jpg"
            image_name += 1

        print(colored("Image saving finished","green"))


    def generate_unique_plates_csv(self, id_filters: tuple[int, ...] = (0, 1))->None:
        """
        This function generates the csv file with the annotations for the plates images with only one plate, taking a
        list of dataset ids filters allowed to be included in the the cvs file.
        """

        print(colored("Generating plates csv","yellow"))
        current_path = os.getcwd() + "/"

        # Delete the file if already exist.
        if os.path.isfile(current_path + self.normalized_data_path + "unique_plates.csv"):
            os.remove(current_path + self.normalized_data_path + "unique_plates.csv")

        # Create the table.
        output_table = []
        for sample in self.plates_data_array:
            if sample.dataset_id in id_filters and len(sample.expected_output) == 1:
                # change tag for specific dataset id
                entry = sample.expected_output[0]
                if sample.dataset_id == 1:
                    entry = [0,0.0,0.0,0.0,0.0]
                else:
                    entry[0] = 100
                output_table.append([sample.image_path] + entry)

        # Write the csv.
        with open(current_path + self.normalized_data_path + "/unique_plates.csv", 'w', encoding="utf-8") as file:
            write = csv.writer(file)
            write.writerow(["img_path", "tag", "xmin", "ymin", "xmax", "ymax"])
            write.writerows(output_table)

        print(colored("Plates csv finished","green"))


    def generate_plates_csv(self, id_filters: tuple[int, ...] = (0, 1))->None:
        """
        This function generates the csv files with the annotations for the plates images, taking a list of  dataset ids
        filters allowed to be included in the the cvs file.
        """

        print(colored("Generating plates csv","yellow"))
        current_path = os.getcwd() + "/"

        # Delete the file if already exist.
        if os.path.isfile(current_path + self.normalized_data_path + "plates.csv"):
            os.remove(current_path + self.normalized_data_path + "plates.csv")

        # Create the table.
        output_table = []
        for sample in  self.plates_data_array:
            if sample.dataset_id in id_filters:
                for entry in sample.expected_output:
                    # change annotation for specific dataset id
                    if sample.dataset_id == 1:
                        entry = [0,0.0,0.0,0.0,0.0]
                    else:
                        entry[0] = 100
                    output_table.append([sample.image_path] + entry)

        # Write the csv.
        with open(current_path + self.normalized_data_path + "/plates.csv", 'w', encoding="utf-8") as file:
            write = csv.writer(file)
            write.writerow(["img_path", "tag", "xmin", "ymin", "xmax", "ymax"])
            write.writerows(output_table)

        print(colored("Plates csv finished","green"))


    def generate_ocr_csv(self, id_filters: tuple[int, ...] = (4,))->None:
        """
        This function generates the csv files with the annotations for the ocr images, taking a list of  dataset ids
        allowed to be included in the the cvs file.
        """

        print(colored("Generating OCR csv","yellow"))
        current_path = os.getcwd() + "/"

        # Delete the file if already exist.
        if os.path.isfile(current_path + self.normalized_data_path + "ocr.csv"):
            os.remove(current_path + self.normalized_data_path + "ocr.csv")

        # Create the table.
        output_table = []
        for sample in  self.ocr_data_array:
            if sample.dataset_id in id_filters:
                for entry in sample.expected_output:
                    output_table.append([sample.image_path] + entry)

        # Write the csv.
        with open(current_path + self.normalized_data_path + "/ocr.csv", 'w', encoding="utf-8") as file:
            write = csv.writer(file)
            write.writerow(["img_path", "tag", "xmin", "ymin", "xmax", "ymax"])
            write.writerows(output_table)

        print(colored("OCR csv finished","green"))


    def generate_untagged_csv(self, id_filters: tuple[int, ...] = (2,))->None:
        """
        This function generates the csv file with the image paths of the untagged images, taking a list of dataset ids
        allowed to be included in the the cvs file.
        """

        print(colored("Generating untagged csv","yellow"))
        current_path = os.getcwd() + "/"

        # Delete the file if already exist.
        if os.path.isfile(current_path + self.normalized_data_path + "untagged.csv"):
            os.remove(current_path + self.normalized_data_path + "untagged.csv")

        # Create the table.
        output_table = []
        for sample in  self.plates_data_array:
            if sample.dataset_id in id_filters:
                output_table.append([sample.image_path])

        # Write the csv.
        with open(current_path + self.normalized_data_path + "/untagged.csv", 'w', encoding="utf-8") as file:
            write = csv.writer(file)
            write.writerow(["img_path"])
            write.writerows(output_table)

        print(colored("Untagged csv finished","green"))


if __name__ == "__main__":

    # ID description:
    # 0: Plates (single and multiple).
    # 1: Without plates.
    # 2: Plates for traditional techniques.
    # 3: Available.
    # 4: OCR.
    # 5: Available.
    # 6: Available.
    # 7: Available.
    # 8: Car images with both plates and OCR data for validation of full system.
    # 9: Available.

    dataset_paths=[
        # Plates:
        DataSetLocation(0, "data/Car_License_Plate_Detection/", "images/", "annotations/", "plate_xml_voc"),
        DataSetLocation(0, "data/license-plate-dataset/dataset/train/", "images/", "annots/", "plate_xml_voc"),
        DataSetLocation(0, "data/license-plate-dataset/dataset/valid/", "images/", "annots/", "plate_xml_voc"),
        # DataSetLocation(0, "data/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano/yolo_plate_dataset/", "", "",
        #                 "plate_txt_yolo"),
        # # Without Plates:
        # DataSetLocation(1, "data/plateless_cars/", "images/", "", "untagged_plate"),
        # # Plates for traditional techniques:
        # DataSetLocation(2, "data/unlabeled_plates/", "images/", "", "untagged_plate"),
        # # OCR:
        # DataSetLocation(4, "data/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano/yolo_plate_ocr_dataset/", "",
        #                 "", "ocr_txt_yolo")
    ]
    data_normalizer = DataNormalizer(dataset_paths)
    data_normalizer.save_images((256,256),(256,256))
    data_normalizer.generate_plates_csv()
    data_normalizer.generate_unique_plates_csv()
    data_normalizer.generate_ocr_csv()
    data_normalizer.generate_untagged_csv()

    # Test image
    # im = cv2.imread("normalized_data/plates_images/8.jpg", cv2.IMREAD_COLOR)
    # scaled_boxes = [
    #     [0,34.8,290.4,68.39999999999999,303.6],
    #     [0,201.6,302.4,237.6,316.8],
    #     [0,375.59999999999997,276.0,399.59999999999997,289.2],
    #     [0,462.0,266.4,478.79999999999995,277.2]]
    # draw_image_with_boxes(im, scaled_boxes)
