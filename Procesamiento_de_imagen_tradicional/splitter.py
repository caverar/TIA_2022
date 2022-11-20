import cv2
import skimage
import matplotlib.pyplot as plt

def split_in_characters(img):
    """
    Splits the image plate in its characters

        Parameters:
            img (ndarray): Image in BGR

        Returns:
            list: list fo images in ndarray corresponding
                to the found characters
    """
    plate_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thres = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 13)
    clear_border = skimage.segmentation.clear_border(thres)


    inverted = cv2.bitwise_not(clear_border)

    # Get totally white columns
    black = 0
    selected_columns = []
    for column in range(inverted.shape[1]):
        if black in inverted[:, column]: continue
        selected_columns.append(column)


    limit_columns = []
    # Get limit white columns
    for i in range(len(selected_columns)-1):
        if selected_columns[i+1] - selected_columns[i] == 1: continue
        limit_columns.append(selected_columns[i])
        limit_columns.append(selected_columns[i+1])
    
    if len(limit_columns) % 2 == 0: 
        print(f"Number of characters in plate before filter: {len(limit_columns) // 2}")
    else:
        print("Not coherent number of characters found")

    plate_chars = []

    for i in range(0, len(limit_columns), 2):
        char = inverted[:, limit_columns[i]+1:limit_columns[i+1]]
        # Delete fake chars by width
        if char.shape[1] < 4: continue
        plate_chars.append(inverted[:, limit_columns[i]+1:limit_columns[i+1]]) 

    print(f"Number of characters in plate after filter: {len(plate_chars)}")

    # Clean chars horizontally
    # Get totally white rows
    black = 0

    clean_plate_chars = []
    for plate_char in plate_chars:
        selected_rows = []
        for row in range(plate_char.shape[0]):
            if black in plate_char[row, :]: continue
            selected_rows.append(row)
        limit_rows = []
        # Get limit white rows
        for i in range(len(selected_rows)-1):
            if selected_rows[i+1] - selected_rows[i] == 1: continue
            limit_rows.append(selected_rows[i])
            limit_rows.append(selected_rows[i+1])

        index_to_delete = []
        for i in range(0, len(limit_rows), 2):
            if limit_rows[i+1] - limit_rows[i] < 6:
                index_to_delete.append(i)
                index_to_delete.append(i+1)

        for i, index in enumerate(index_to_delete): limit_rows.pop(index - i)

        if len(limit_rows):
            plate_char = plate_char[limit_rows[0]+1: limit_rows[1], :]
        else:
            print("Not a character")
        clean_plate_chars.append(plate_char)

    black = 0

    final_chars = []
    for plate_char in clean_plate_chars:
        selected_rows = []
        for row in range(plate_char.shape[0]):
            if black in plate_char[row, :]: continue
            selected_rows.append(row)
        if len(selected_rows) / plate_char.shape[0] > 0.5: continue 
        final_chars.append(plate_char)

    final_chars2 = []
    # Re check if it is a character or not
    for plate in final_chars:
        if plate.shape[0] < 10 and plate.shape[1] < 10: continue
        final_chars2.append(plate)

    return final_chars2

