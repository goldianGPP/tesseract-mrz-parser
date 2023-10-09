import cv2
import re
import logging
import pytesseract

# Configure logging
logging.basicConfig(filename="ocr.log", level=logging.INFO)
logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

class mrzParser:
    def __init__(self, image_cv):
        self.image_cv = image_cv
        self.mrz_patterns = {
            "TD1": [
                r"(?P<doc>[A|C|I][A-Z0-9<])(?P<country>[A-Z]{3})(?P<doc_number>[A-Z0-9<]{9})(?P<hash_doc_number>[0-9])(?P<optional_data1>[A-Z0-9<]{15})",
                r"(?P<birth_date>[0-9]{6})(?P<hash_birth_date>[0-9])(?P<sex>[M|F|X|<])(?P<expiry_date>[0-9]{6})(?P<hash_expiry_date>[0-9])(?P<nationality>[A-Z]{3})(?P<optional_data2>[A-Z0-9<]{11})(?P<final_hash>[0-9])",
                r"(?P<names>[A-Z0-9<]{30})",
            ],
            "TD2": [
                r"(?P<doc>[A|C|I][A-Z0-9<])(?P<country>[A-Z]{3})(?P<names>[A-Z0-9<]{31})",
                r"(?P<doc_number>[A-Z0-9<]{9})(?P<hash_doc_number>[0-9])(?P<nationality>[A-Z]{3})(?P<birth_date>[0-9]{6})(?P<hash_birth_date>[0-9])(?P<sex>[M|F|X|<])(?P<expiry_date>[0-9]{6})(?P<hash_expiry_date>[0-9])(?P<optional_data1>[A-Z0-9<]{7})(?P<final_hash>[0-9])",
            ],
            "TD3": [
                r"(?P<doc>P[A-Z0-9<])(?P<country>[A-Z]{3})(?P<names>[A-Z0-9<]{39})",
                r"(?P<doc_number>[A-Z0-9<]{9})(?P<hash_doc_number>[0-9])(?P<nationality>[A-Z]{3})(?P<birth_date>[0-9]{6})(?P<hash_birth_date>[0-9])(?P<sex>[M|F|X|<])(?P<expiry_date>[0-9]{6})(?P<hash_expiry_date>[0-9])(?P<optional_data1>[A-Z0-9<]{14})(?P<hash_optional_data1>[0-9])(?P<final_hash>[0-9])",
            ],
            "MRVA": [
                r"(?P<doc>V[A-Z0-9<])(?P<country>[A-Z]{3})(?P<names>[A-Z0-9<]{39})",
                r"(?P<doc_number>[A-Z0-9<]{9})(?P<hash_doc_number>[0-9])(?P<nationality>[A-Z]{3})(?P<birth_date>[0-9]{6})(?P<hash_birth_date>[0-9])(?P<sex>[M|F|X|<])(?P<expiry_date>[0-9]{6})(?P<hash_expiry_date>[0-9])(?P<optional_data1>[A-Z0-9<]{16})",
            ],
            "MRVB": [
                r"(?P<doc>V[A-Z0-9<])(?P<country>[A-Z]{3})(?P<names>[A-Z0-9<]{31})",
                r"(?P<doc_number>[A-Z0-9<]{9})(?P<hash_doc_number>[0-9])(?P<nationality>[A-Z]{3})(?P<birth_date>[0-9]{6})(?P<hash_birth_date>[0-9])(?P<sex>[M|F|X|<])(?P<expiry_date>[0-9]{6})(?P<hash_expiry_date>[0-9])(?P<optional_data1>[A-Z0-9<]{8})",
            ],
        }
        self.mrz_type_counts = {mrz_type: 0 for mrz_type in self.mrz_patterns.keys()}
        self.mrz_results = {mrz_type: [] for mrz_type in self.mrz_patterns.keys()}

    def perform_ocr(self, language='mrz'):
        try:
            return pytesseract.image_to_data(self.image_cv, output_type=pytesseract.Output.DICT, lang=language)
        except pytesseract.TesseractError as e:
            logger.error(f"OCR Error: {e}")
            return None

    def draw_bounding_box(self, x, y, w, h):
        cv2.rectangle(self.image_cv, (x, y), (x + w, y + h), (0, 0, 255), 1)

    def get_matched_groups(self, match):
        return match.groupdict().items()

    def find_detected_mrz_type(self):
        detected_mrz_type = max(self.mrz_type_counts, key=self.mrz_type_counts.get)
        return detected_mrz_type

    def process_ocr_results(self, language="mrz"):
        data = self.perform_ocr(language=language)

        if data:
            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                if text:
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    for mrz_type, regex_patterns in self.mrz_patterns.items():
                        if self.mrz_type_counts[mrz_type] < len(regex_patterns):
                            regex_pattern = regex_patterns[self.mrz_type_counts[mrz_type]]
                            match = re.match(regex_pattern, text)
                            if match:
                                self.mrz_type_counts[mrz_type] += 1
                                self.draw_bounding_box(x, y, w, h)
                                self.mrz_results[mrz_type].append({
                                    "group": self.get_matched_groups(match),
                                    "mrz": text,
                                    "pattern": regex_pattern,
                                })

        return self.get_result()

    def get_result(self):
        detected_mrz_type = self.find_detected_mrz_type()

        if self.mrz_type_counts[detected_mrz_type] == len(self.mrz_patterns[detected_mrz_type]):
            return {
                "mrz_type": detected_mrz_type,
                "data": self.mrz_results[detected_mrz_type],
            }
        else:
            logger.info(f"failed to extract the MRZ")
            return None

    def print_results(self):
        result = self.get_result()

        if result:
            logger.info(f"MRZ Type: {result['mrz_type']}")
            logger.info("--------------------------------------------")

            for i, data in enumerate(result['data'], start=1):
                logger.info(f"Line {i}")
                logger.info("--------------------------------------------")
                logger.info(f"MRZ: {data['mrz']}")
                logger.info(f"Pattern: {data['pattern']}")
                logger.info("--------------------------------------------")
                for name, value in data['group']:
                    logger.info(f"\t{name} : {value}")
                logger.info("--------------------------------------------")