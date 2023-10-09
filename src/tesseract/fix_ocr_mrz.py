import pytesseract
import cv2
import re
import os

def load_image(image_path):
    return cv2.imread(image_path)

def perform_ocr(image_cv):
    # set 'mrz' language model
    return pytesseract.image_to_data(image_cv, output_type=pytesseract.Output.DICT, lang='mrz')

def initialize_counters(mrz_patterns):
    mrz_type_counts = {mrz_type: 0 for mrz_type in mrz_patterns.keys()}
    first_line_match = {mrz_type: False for mrz_type in mrz_patterns.keys()}
    return mrz_type_counts, first_line_match

def process_ocr_results(data, mrz_patterns):
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # stream on mrz_patterns items
            for mrz_type, regex_patterns in mrz_patterns.items():
                # check on each first regex of every mrz type for matching pattern with the text

                # check if the type is on it first line or not, if not it goes through for match checking
                if not first_line_match[mrz_type]:
                    first_regex_pattern = regex_patterns[0]
                    match = re.match(first_regex_pattern, text)
                    if match:
                        # if match the mrz type will be flagged @true to be included in the next lines
                        first_line_match[mrz_type] = True
                        # if match the mrz type will also be set a count on for counter check
                        mrz_type_counts[mrz_type] += 1
                        draw_bounding_box(image_cv, x, y, w, h)
                        print_matched_groups(text, first_regex_pattern, mrz_type, match)
                        break
                elif mrz_type_counts[mrz_type] < len(regex_patterns):
                    for j in range(1, len(regex_patterns)):
                        regex_pattern = regex_patterns[j]
                        match = re.match(regex_pattern, text)
                        if match:
                            mrz_type_counts[mrz_type] += 1
                            draw_bounding_box(image_cv, x, y, w, h)
                            print_matched_groups(text, regex_pattern, mrz_type, match)
                            break

def draw_bounding_box(image_cv, x, y, w, h):
    cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 0, 255), 1)

def print_matched_groups(text, regex_pattern, mrz_type, match):
    print("---------------------------------")
    print(f"value : {text}")
    print(f"pattern : {regex_pattern}")
    print(f"MRZ Type: {mrz_type}")
    print("---------------------------------")
    for group_num, group_text in enumerate(match.groups(), start=1):
        print(f"Group #{group_num}: {group_text}")
    print("")

def find_detected_mrz_type(mrz_type_counts):
    detected_mrz_type = max(mrz_type_counts, key=mrz_type_counts.get)
    return detected_mrz_type

def save_output_image(image_cv):
    cv2.imwrite("output_image_with_mrz_boxes.png", image_cv)
    os.system("output_image_with_mrz_boxes.png")

# MRZ REGEX
mrz_patterns = {
    "TD1": [
        r"([A|C|I][A-Z0-9<]{1})([A-Z]{3})([A-Z0-9<]{9})([0-9]{1})([A-Z0-9<]{15})",
        r"([0-9]{6})([0-9]{1})([M|F|X|<]{1})([0-9]{6})([0-9]{1})([A-Z]{3})([A-Z0-9<]{11})([0-9]{1})",
        r"([A-Z0-9<]{30})"
    ],
    "TD2": [
        r"([A|C|I][A-Z0-9<]{1})([A-Z]{3})([A-Z0-9<]{31})",
        r"([A-Z0-9<]{9})([0-9]{1})([A-Z]{3})([0-9]{6})([0-9]{1})([M|F|X|<]{1})([0-9]{6})([0-9]{1})([A-Z0-9<]{7})([0-9]{1})"
    ],
    "TD3": [
        r"(P[A-Z0-9<]{1})([A-Z]{3})([A-Z0-9<]{39})",
        r"([A-Z0-9<]{9})([0-9]{1})([A-Z]{3})([0-9]{6})([0-9]{1})([M|F|X|<]{1})([0-9]{6})([0-9]{1})([A-Z0-9<]{14})([0-9]{1})([0-9]{1})"
    ],
    "MRVA": [
        r"(V[A-Z0-9<]{1})([A-Z]{3})([A-Z0-9<]{39})",
        r"([A-Z0-9<]{9})([0-9]{1})([A-Z]{3})([0-9]{6})([0-9]{1})([M|F|X|<]{1})([0-9]{6})([0-9]{1})([A-Z0-9<]{16})"
    ],
    "MRVB": [
        r"(V[A-Z0-9<]{1})([A-Z]{3})([A-Z0-9<]{31})",
        r"([A-Z0-9<]{9})([0-9]{1})([A-Z]{3})([0-9]{6})([0-9]{1})([M|F|X|<]{1})([0-9]{6})([0-9]{1})([A-Z0-9<]{8})"
    ]
}

# load image with openCV
image_cv = load_image("C:/Users/USER/Documents/EyeGil/_PYTHON/EyeToCr/data/input/images/old/MRVA.jpg") # taruh gambar di sini

# Perform OCR on the image to get result (text and bounding boxes)
data = perform_ocr(image_cv)

# Initialize counters and flags (for mrz type detection)
# note: this will set up a count for any mrz types and pattern also a true/false cheker on the first line of MRZ text
mrz_type_counts, first_line_match = initialize_counters(mrz_patterns)

# Process OCR results (parsing data and bounding boxes)
process_ocr_results(data, mrz_patterns)

detected_mrz_type = find_detected_mrz_type(mrz_type_counts)

if mrz_type_counts[detected_mrz_type] > 0:
    print(f"Detected MRZ Type: {detected_mrz_type}")
else:
    print("MRZ Type not detected")


save_output_image(image_cv)