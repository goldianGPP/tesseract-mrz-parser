import cv2
import numpy as np
import imutils

def image_preproccessing(image_path):
    # initialize a rectangular and square structuring kernel
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    # load the image, resize it, and convert it to grayscale
    original_image = cv2.imread(image_path)
    image = imutils.resize(original_image, height=600)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # smooth the image using a 3x3 Gaussian, then apply the blackhat
    # morphological operator to find dark regions on a light background
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    # compute the Scharr gradient of the blackhat image and scale the
    # result into the range [0, 255]
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    # apply a closing operation using the rectangular kernel to close
    # gaps in between letters -- then apply Otsu's thresholding method
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # perform another closing operation, this time using the square
    # kernel to close gaps between lines of the MRZ, then perform a
    # series of erosions to break apart connected components
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=4)

    # during thresholding, it's possible that border pixels were
    # included in the thresholding, so let's set 5% of the left and
    # right borders to zero
    p = int(image.shape[1] * 0.05)
    thresh[:, 0:p] = 0
    thresh[:, image.shape[1] - p:] = 0

    # find contours in the thresholded image and sort them by their
    # size
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Initialize a list to store ROI bounding box coordinates
    roi_coordinates = []

    # Loop over the contours
    for c in cnts:
        # Compute the bounding box of the contour and use the contour to
        # compute the aspect ratio and coverage ratio of the bounding box
        # width to the width of the image
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        crWidth = w / float(gray.shape[1])
        # Check to see if the aspect ratio and coverage width are within
        # acceptable criteria
        if ar > 5 and crWidth > 0.75:
            # Pad the bounding box since we applied erosions and now need
            # to re-grow it
            pX = int((x + w) * 0.03)
            pY = int((y + h) * 0.03)
            (x, y) = (x - pX, y - pY)
            (w, h) = (w + (pX * 2), h + (pY * 2))
            # Store the ROI coordinates in the list
            roi_coordinates.append((x, y, w, h))

    # Combine all detected ROIs into a single bounding box
    if roi_coordinates:
        # Get the minimum x, minimum y, maximum width, and maximum height
        x_values, y_values, w_values, h_values = zip(*roi_coordinates)
        combined_x = min(x_values)
        combined_y = min(y_values)
        combined_w = max(x_values) + max(w_values) - combined_x
        combined_h = max(y_values) + max(h_values) - combined_y

        # Scale the coordinates back to the original image size
        original_height = cv2.imread(image_path).shape[0]  # Get the height of the original image
        original_scale = original_height / 600.0  # Assuming you resized the image to a height of 600

        combined_x_original = int(combined_x * original_scale)
        combined_y_original = int(combined_y * original_scale)
        combined_w_original = int(combined_w * original_scale)
        combined_h_original = int(combined_h * original_scale)

        # Extract the combined ROI from the original size image
        original_image = cv2.imread(image_path)  # Load the original image
        combined_roi_original = original_image[combined_y_original:combined_y_original + combined_h_original,
                                combined_x_original:combined_x_original + combined_w_original].copy()

        # Draw a bounding box around the combined ROI in the original size image
        cv2.rectangle(original_image, (combined_x_original, combined_y_original),
                    (combined_x_original + combined_w_original, combined_y_original + combined_h_original),
                    (0, 255, 0), 2)
        cv2.imshow("Combined ROI (Original Size)", combined_roi_original)
        cv2.waitKey(0)

        return combined_roi_original