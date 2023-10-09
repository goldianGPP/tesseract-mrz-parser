import pytesseract
import cv2
import re
import os

# Define regular expressions for MRZ text for each type
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

# Load the image with OpenCV for drawing bounding boxes
image_cv = cv2.imread("C:/Users/USER/Documents/EyeGil/_PYTHON/EyeToCr/data/input/images/TD1.jpg")

# Perform OCR on the image to get text and bounding boxes
data = pytesseract.image_to_data(image_cv, output_type=pytesseract.Output.DICT, lang='mrz')

# Initialize a dictionary to count matching regex patterns for each MRZ type
mrz_type_counts = {mrz_type: 0 for mrz_type in mrz_patterns.keys()}

# Initialize a dictionary to track which MRZ types have matched the first line
first_line_match = {mrz_type: False for mrz_type in mrz_patterns.keys()}

# Process the OCR results and extract and print MRZ groups
for i in range(len(data['text'])):
    text = data['text'][i].strip()
    if text:
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        
        # Iterate through MRZ types and their regex patterns
        for mrz_type, regex_patterns in mrz_patterns.items():
            if not first_line_match[mrz_type]:
                first_regex_pattern = regex_patterns[0]
                match = re.match(first_regex_pattern, text)
                if match:
                    first_line_match[mrz_type] = True
                    # Increment the count for the matching MRZ type
                    mrz_type_counts[mrz_type] += 1
                    # Draw a bounding box around MRZ text
                    cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    # Print matched groups for the first line
                    print("---------------------------------")
                    print(f"value : {text}")
                    print(f"pattern : {first_regex_pattern}")
                    print(f"MRZ Type: {mrz_type}")
                    print("First Line Matched")
                    print("---------------------------------")
                    for group_num, group_text in enumerate(match.groups(), start=1):
                        print(f"Group #{group_num}: {group_text}")
                    print("")
                    break  # Stop checking regex patterns for this MRZ type after the first line match
            elif mrz_type_counts[mrz_type] < len(regex_patterns):
                # Check the second and third lines for MRZ types that haven't been fully matched
                for j in range(1, len(regex_patterns)):
                    regex_pattern = regex_patterns[j]
                    match = re.match(regex_pattern, text)
                    if match:
                        # Increment the count for the matching MRZ type
                        mrz_type_counts[mrz_type] += 1
                        # Draw a bounding box around MRZ text
                        cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 0, 255), 1)
                        # Print matched groups for the second and third lines
                        print("---------------------------------")
                        print(f"value : {text}")
                        print(f"pattern : {regex_pattern}")
                        print(f"MRZ Type: {mrz_type}")
                        print("---------------------------------")
                        for group_num, group_text in enumerate(match.groups(), start=1):
                            print(f"Group #{group_num}: {group_text}")
                        print("")
                        break  # Stop checking regex patterns once a match is found

# Find the MRZ type with the highest count
detected_mrz_type = max(mrz_type_counts, key=mrz_type_counts.get)

if mrz_type_counts[detected_mrz_type] > 0:
    print(f"Detected MRZ Type: {detected_mrz_type}")
else:
    print("MRZ Type not detected")

# Save the image with bounding boxes around MRZ text (optional)
cv2.imwrite("output_image_with_mrz_boxes.png", image_cv)

os.system("output_image_with_mrz_boxes.png")
