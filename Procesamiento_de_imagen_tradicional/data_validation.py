import os

import cv2
import numpy as np
import pandas as pd

from plate_detector import detect_plate

if __name__ == "__main__":



    # Fet directory of the current file
    path = os.path.dirname(os.path.abspath(__file__))

    # Load the cvs file
    data = pd.read_csv(path + "/../normalized_data/unique_plates.csv")

    # Separate data into plated and un-plated cars
    plateless_cars = data[data["tag"] == 0]
    plated_cars = data[data["tag"] == 100]

    # Sample 50 images of each type
    plateless_cars_sample = plateless_cars.sample(n=50)
    plated_cars_sample = plated_cars.sample(n=50)

    # join the samples
    samples = pd.concat([plated_cars_sample, plateless_cars_sample])

    # Confusion matrix parameters
    # Correct predictions
    true_positive = 0
    true_negative = 0
    # Wrong predictions
    false_positive = 0
    false_negative = 0

    print("Every time that an image is shown, press any key to continue and response the question in the terminal.")

    i = 0
    for ind in samples.index:
        print("sample: ", i)
        i += 1
        # load image
        input_image = cv2.imread(str(samples["img_path"][ind]))
        is_plated_label = samples["tag"][ind] == 100
        # detect plate
        (is_plated_prediction, output_image, message) = detect_plate(input_image)

        # Update confusion matrix values
        if is_plated_label and is_plated_prediction:
            # Verify if the plate is correct
            cv2.imshow(f"Image {i}",output_image)
            cv2.waitKey(0)
            while (res:=input(f"The extracted {i} image has a plate? (Enter y/n)\n").lower() ) not in {"y", "n"}: pass
            cv2.destroyAllWindows()
            if res == "y":
                true_positive += 1
            else:
                false_positive += 1

        elif not is_plated_label and is_plated_prediction:
            # Verify if the label is correct
            cv2.imshow(f"Image {i}",output_image)
            cv2.waitKey(0)
            while (res:=input(f"The extracted {i} image has a plate? (Enter y/n)\n").lower() ) not in {"y", "n"}: pass
            cv2.destroyAllWindows()
            if res == "y":
                true_positive += 1
            else:
                false_positive += 1
        elif is_plated_label and not is_plated_prediction:
            false_negative += 1
        elif not is_plated_label and not is_plated_prediction:
            true_negative += 1

        # cv2.imshow("input_image", input_image)
        # cv2.waitKey(0)
        # x = input()


    # Calculate accuracy
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    # Calculate recall
    recall = true_positive / (true_positive + false_negative)
    # Calculate precision
    precision = true_positive / (true_positive + false_positive)
    # Calculate f1 score
    f1_score = 2 * (recall * precision) / (recall + precision)

    # Print results
    print("True positive: ", true_positive)
    print("True negative: ", true_negative)
    print("False positive: ", false_positive)
    print("False negative: ", false_negative)
    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("F1 score: ", f1_score)

    # Delete txt file if it exists
    if os.path.exists(path + "/data_validation_results.txt"):
        os.remove(path + "/data_validation_results.txt")
    # Write results to a file
    with open(path + "/data_validation_results.txt", "w") as f:
        f.write(f"True positive: {true_positive}\n")
        f.write(f"True negative: {true_negative}\n")
        f.write(f"False positive: {false_positive}\n")
        f.write(f"False negative: {false_negative}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"F1 score: {f1_score}\n")


