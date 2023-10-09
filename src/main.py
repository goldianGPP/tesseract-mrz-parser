import cv2
import logging
import os
import argparse
import detector.mrzDetector as mrzDetector
from parser.mrzParser import mrzParser
from parser.ocrParser import ocrParser

# Configure logging
logging.basicConfig(filename='ocr.log', level=logging.INFO)
logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

def load_image(image_path):
    """Load an image from the given path."""
    return cv2.imread(image_path)

def save_output_image(image_cv, output_path="output_image_with_mrz_boxes.png"):
    """Save the image with MRZ boxes to the specified output path."""
    try:
        cv2.imwrite(output_path, image_cv)
        os.system(output_path)
        logger.info(f"Saved output image as {output_path}")
    except Exception as e:
        logger.error(f"Error saving output image: {e}")

if __name__ == "__main__":
    # Load image
    # image_path = "C:/Users/USER/Documents/EyeGil/_PYTHON/EyeToCr/data/input/images/MRV.jpg"
    image_path = "C:/Users/USER/Documents/EyeGil/_PYTHON/EyeToCr/data/input/images/TD3.jpg"

    parser = argparse.ArgumentParser(description="Perform OCR on an image and extract MRZ data.")
    parser.add_argument("--module", default="OCR_SENTENCE", help="Path to the image for OCR.")
    parser.add_argument("--language", default="eng_old", help="OCR language (default: mrz)")
    parser.add_argument("--path", default=image_path, help="Path to the image for OCR.")

    args = parser.parse_args()

    # image_cv = mrzDetector.image_preproccessing(args.path)
    image_cv = cv2.imread(args.path)

    if image_cv is not None:

        if args.module == "MRZ_PARSER" :
            mrz_parser = mrzParser(image_cv)
            mrz_parser.process_ocr_results()
            mrz_parser.print_results()
        elif args.module == "OCR_WORDS" :
            ocr_parser = ocrParser(image_cv)
            ocr_parser.process_ocr_results()
            ocr_parser.print_results()
        elif args.module == "OCR_SENTENCE" :
            ocr_parser = ocrParser(image_cv)
            ocr_parser.process_ocr_results(method=args.module, language=args.language)
            ocr_parser.print_results()

        # Save the output image with MRZ boxes
        save_output_image(image_cv, f"{args.module}.png")
    else:
        logger.error(f"Failed to load image from {args.path}")
