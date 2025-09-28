import unittest
from PIL import Image
import pyautogui

# -----------------------------
# Configuration
# -----------------------------
SCREEN_REGION = (845, 430, 575, 200)  # adjust to your screen
CARD_AREAS = [
    (0, 0, 145, 200),
    (145, 0, 145, 200),
    (290, 0, 145, 200),
    (435, 0, 145, 200),
]

# -----------------------------
# Functions
# -----------------------------
def capture_screenshot():
    """Take a screenshot of the full 4-card region."""
    return pyautogui.screenshot(region=SCREEN_REGION)

def crop_card(screenshot, area):
    """Crop a single card from the screenshot."""
    x, y, w, h = area
    return screenshot.crop((x, y, x + w, y + h))

# -----------------------------
# Unit Tests
# -----------------------------
class TestCardCapture(unittest.TestCase):

    def test_screenshot_size(self):
        """Screenshot should have the same size as SCREEN_REGION."""
        screenshot = capture_screenshot()
        expected_width = SCREEN_REGION[2]
        expected_height = SCREEN_REGION[3]
        self.assertEqual(screenshot.size, (expected_width, expected_height))

    def test_crop_sizes(self):
        """Each cropped card should match its defined width and height."""
        screenshot = capture_screenshot()
        for i, area in enumerate(CARD_AREAS):
            cropped = crop_card(screenshot, area)
            expected_w, expected_h = area[2], area[3]
            self.assertEqual(cropped.size, (expected_w, expected_h),
                             f"Card {i+1} size mismatch")

    def test_crop_not_empty(self):
        """Each cropped card should not be fully blank (checks pixel diversity)."""
        screenshot = capture_screenshot()
        for i, area in enumerate(CARD_AREAS):
            cropped = crop_card(screenshot, area)
            pixels = list(cropped.getdata())
            unique_pixels = set(pixels)
            self.assertTrue(len(unique_pixels) > 1, f"Card {i+1} seems blank")

# -----------------------------
# Run tests
# -----------------------------
if __name__ == "__main__":
    unittest.main()
