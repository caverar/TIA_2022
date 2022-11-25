import cv2
import matplotlib.pyplot as plt
from third_test import test_plate_method3
import numpy as np
import skimage
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

def read_plate(plate):
    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

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

    #print(limit_columns)
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

    if len(final_chars2) != 7: return "", ""

    for i, pl in enumerate(final_chars2):
        filename = f'char{i}.jpg'
        cv2.imwrite(filename, pl)

    letter_0 = cv2.imread('char0.jpg')
    letter_1 = cv2.imread('char1.jpg')
    letter_2 = cv2.imread('char5.jpg')
    letter_3= cv2.imread('char6.jpg')

    h0, w0 = letter_0.shape[:2]
    h1, w1 = letter_1.shape[:2]
    h2, w2 = letter_2.shape[:2]
    h3, w3 = letter_3.shape[:2]

    letters = np.zeros((max(h0, h1, h2, h3) + 10, w0 + w1 + w2 + w3 + 15, 3), dtype=np.uint8)
    letters[:,:] = (255,255,255)

    letters[2:h0+2, 3:w0+3,:3] = letter_0
    letters[2:h1+2, w0+3+3: w1 + w0 + 3 + 3,:3] = letter_1
    letters[2:h2+2,  w1 + w0 + 3 + 3 + 3: w1 + w0 + 3 + 3 + 3 + w2,:3] = letter_2
    letters[2:h3+2, w1 + w0 + 3 + 3 + 3 + w2 + 3: w1 + w0 + 3 + 3 + w2 + 3 + 3 + w3,:3] = letter_3

    plt.title("Letters in plate")
    plt.imshow(cv2.cvtColor(letters, cv2.COLOR_BGR2RGB))
    plt.show()

    psm = 7
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    options += " --psm {}".format(psm)

    letters_text = pytesseract.image_to_string(letters, config=options)
    print(f"Detected letters: {letters_text}")


    num_0 = cv2.imread('char2.jpg')
    num_1 = cv2.imread('char3.jpg')
    num_2 = cv2.imread('char4.jpg')


    h0, w0 = num_0.shape[:2]
    h1, w1 = num_1.shape[:2]
    h2, w2 = num_2.shape[:2]

    numbers = np.zeros((max(h0, h1, h2) + 10, w0 + w1 + w2 + 12, 3), dtype=np.uint8)
    numbers[:,:] = (255,255,255)
    numbers[2:h0+2, 3:w0+3,:3] = num_0
    numbers[2:h1+2, w0+3+3: w1 + w0 + 3 + 3,:3] = num_1
    numbers[2:h2+2,  w1 + w0 + 3 + 3 + 3: w1 + w0 + 3 + 3 + 3 + w2,:3] = num_2

    plt.title("Numbers in plate")
    plt.imshow(cv2.cvtColor(numbers, cv2.COLOR_BGR2RGB))
    plt.show()

    alphanumeric = "0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    options += " --psm {}".format(psm)

    numbers_text = pytesseract.image_to_string(numbers, config=options)
    print(f"Detected numbers: {numbers_text}")

    return letters_text.strip(), numbers_text.strip()
