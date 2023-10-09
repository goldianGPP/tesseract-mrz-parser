import pytesseract
import cv2
import os

# Load the image with OpenCV
image_cv = cv2.imread("C:/Users/USER/Documents/EyeGil/_PYTHON/EyeToCr/data/input/images/old/TD2.jpg")

# Perform OCR on the image to get text, confidences, and bounding boxes
data = pytesseract.image_to_data(image_cv, output_type=pytesseract.Output.DICT)

# Define a list of colors to use for bounding boxes (you can add more colors)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# Initialize a color index
color_index = 0

# Process the OCR results and print text and bounding boxes with different colors
for i in range(len(data["text"])):
    text = data["text"][i]
    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
    conf = int(data["conf"][i])

    # Get the current color
    color = colors[color_index % len(colors)]

    # Draw a bounding box with the current color on the image
    cv2.rectangle(image_cv, (x, y), (x + w, y + h), color, 1)

    # Print the text, confidence, and bounding box coordinates with the current color
    print(f"Text: {text}, Confidence: {conf}, Bounding Box: x={x}, y={y}, w={w}, h={h}")

    # Increment the color index
    color_index += 1

# Save the image with colored bounding boxes (optional)
cv2.imwrite("output_image_with_colored_boxes.png", image_cv)

# Show the image with colored bounding boxes (optional)
os.system("output_image_with_colored_boxes.png")
