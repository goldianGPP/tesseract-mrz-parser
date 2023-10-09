import math
import pytesseract
import cv2
import numpy as np
import os

def imgCanny(img) :
    img_copy = img.copy()
    img_canny = cv2.Canny(img_copy, 50, 100, apertureSize = 3)
    return img_copy, img_canny

def imgHough(img_copy, img_canny) :
    img_hough = cv2.HoughLinesP(img_canny, 1, math.pi / 180, 100, minLineLength = 100, maxLineGap = 10)

    (x, y, w, h) = (np.amin(img_hough, axis = 0)[0,0], np.amin(img_hough, axis = 0)[0,1], np.amax(img_hough, axis = 0)[0,0] - np.amin(img_hough, axis = 0)[0,0], np.amax(img_hough, axis = 0)[0,1] - np.amin(img_hough, axis = 0)[0,1])
    img_roi = img_copy[y:y+h,x:x+w]
    return img_roi

custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=<0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Load the image with OpenCV for drawing bounding boxes
img = cv2.imread("C:/Users/USER/Documents/EyeGil/_PYTHON/EyeToCr/data/input/images/MRZ_ch.jpg")

# Perform OCR on the image to get text and bounding boxes
data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

# Initialize variables to track the current sentence and its bounding box
current_sentence = ""
current_sentence_box = None

# Process the OCR results and print text and bounding boxes by sentences
for i in range(len(data['text'])):
    text = data['text'][i].strip()
    if text:
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        
        # Combine words to form sentences
        current_sentence += " " + text
        if current_sentence_box is None:
            current_sentence_box = (x, y, x + w, y + h)
        else:
            current_sentence_box = (
                min(current_sentence_box[0], x),
                min(current_sentence_box[1], y),
                max(current_sentence_box[2], x + w),
                max(current_sentence_box[3], y + h)
            )
    elif current_sentence:
        # Draw a bounding box around the sentence
        cv2.rectangle(img, (current_sentence_box[0], current_sentence_box[1]),
                      (current_sentence_box[2], current_sentence_box[3]), (0, 0, 255), 1)
        
        # Print the sentence and its bounding box
        print(f"Sentence: {current_sentence.strip()}")
        
        # Reset variables for the next sentence
        current_sentence = ""
        current_sentence_box = None

# Check if there is a sentence left after the loop (for the last sentence)
if current_sentence:
    print(current_sentence_box)
    # Draw a bounding box around the last sentence
    cv2.rectangle(img, (current_sentence_box[0], current_sentence_box[1]),
                  (current_sentence_box[2], current_sentence_box[3]), (0, 0, 255), 1)
    
    # Print the last sentence and its bounding box
    print(f"Sentence: {current_sentence.strip()}")

    
# Save the image with bounding boxes around sentences (optional)
cv2.imwrite("output_image_with_sentence_boxes.png", img)

os.system("output_image_with_sentence_boxes.png")