import unittest
from PIL import Image, ImageDraw
import pytesseract

# Path to tesseract, adjust if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class TestTesseractOCR(unittest.TestCase):

    def test_tesseract_version(self):
        version = pytesseract.get_tesseract_version()
        self.assertIsNotNone(version)

    def test_simple_text_ocr(self):
        img = Image.new("RGB", (200, 60), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Hello 2025!", fill=(0, 0, 0))
        result = pytesseract.image_to_string(img).strip()
        self.assertIn("Hello", result)
        self.assertIn("2025", result)


if __name__ == "__main__":
    unittest.main()
