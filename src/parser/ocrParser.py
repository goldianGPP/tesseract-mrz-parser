import cv2
import logging
import pytesseract

# Configure logging
logging.basicConfig(filename="ocr.log", level=logging.INFO)
logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


class ocrParser:
    image_cv = None
    ocr_result = None

    def __init__(self, image_cv):
        self.image_cv = image_cv

    def perform_ocr(self, language="eng"):
        """Perform OCR on the given image and return the results."""
        try:
            return pytesseract.image_to_data(self.image_cv, output_type=pytesseract.Output.DICT, lang=language)
        except pytesseract.TesseractError as e:
            logger.error(f"OCR Error: {e}")
            return None

    def process_ocr_results(self, method="OCR_WORDS", language="eng"):
        data = self.perform_ocr(language=language)

        if data :
            self.ocr_result = {
                "text": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
                "conf": []
            }
            if method == "OCR_WORDS":
                for i in range(len(data["text"])):
                    text = data["text"][i].strip()
                    if text:
                        x, y, w, h, conf = (
                            data["left"][i],
                            data["top"][i],
                            data["width"][i],
                            data["height"][i],
                            data["conf"][i]
                        )

                        self.ocr_result['text'].append(text)
                        self.ocr_result['left'].append(x)
                        self.ocr_result['top'].append(y)
                        self.ocr_result['width'].append(w)
                        self.ocr_result['height'].append(h)
                        self.ocr_result['conf'].append(conf)
            elif method == "OCR_SENTENCE":
                current_sentence = ""
                current_sentence_box = None

                for i in range(len(data["text"])):
                    text = data["text"][i].strip()
                    if text:
                        x, y, w, h, conf = (
                            data["left"][i],
                            data["top"][i],
                            data["width"][i],
                            data["height"][i],
                            data["conf"][i]
                        )

                        # Combine words to form sentences
                        current_sentence += " " + text
                        if current_sentence_box is None:
                            current_sentence_box = (x, y, x + w, y + h, conf, 1)
                        else:
                            current_sentence_box = (
                                min(current_sentence_box[0], x),
                                min(current_sentence_box[1], y),
                                max(current_sentence_box[2], x + w),
                                max(current_sentence_box[3], y + h),
                                current_sentence_box[4] + conf,
                                current_sentence_box[5] + 1
                            )
                    elif current_sentence:
                        self.ocr_result['text'].append(current_sentence.strip())
                        self.ocr_result['left'].append(current_sentence_box[0])
                        self.ocr_result['top'].append(current_sentence_box[1])
                        self.ocr_result['width'].append(current_sentence_box[2] - current_sentence_box[0])
                        self.ocr_result['height'].append(current_sentence_box[3] - current_sentence_box[1])
                        self.ocr_result['conf'].append(current_sentence_box[4]/current_sentence_box[5])

                        # Reset variables for the next sentence
                        current_sentence = ""
                        current_sentence_box = None

                # Check if there is a sentence left after the loop (for the last sentence)
                if current_sentence:
                    self.ocr_result['text'].append(current_sentence.strip())
                    self.ocr_result['left'].append(current_sentence_box[0])
                    self.ocr_result['top'].append(current_sentence_box[1])
                    self.ocr_result['width'].append(current_sentence_box[2] - current_sentence_box[0])
                    self.ocr_result['height'].append(current_sentence_box[3] - current_sentence_box[1])
                    self.ocr_result['conf'].append(current_sentence_box[4]/current_sentence_box[5])

        return self.ocr_result

    def print_results(self):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        # Initialize a color index
        color_index = 0

        for i in range(len(self.ocr_result["text"])):
            text = self.ocr_result["text"][i]
            x, y, w, h = (
                self.ocr_result["left"][i],
                self.ocr_result["top"][i],
                self.ocr_result["width"][i],
                self.ocr_result["height"][i],
            )
            conf = int(self.ocr_result["conf"][i])

            # Get the current color
            color = colors[color_index % len(colors)]

            # Draw a bounding box with the current color on the image
            cv2.rectangle(self.image_cv, (x, y), (x + w, y + h), color, 1)

            # Print the text, confidence, and bounding box coordinates with the current color
            logger.info("--------------------------------------------")
            logger.info(f"Text {i+1}: {text}")
            logger.info(f"Confidence: {conf}")
            logger.info(f"Bounding Box: x={x}, y={y}, w={w}, h={h}")

            # Increment the color index
            color_index += 1
        logger.info("--------------------------------------------")
