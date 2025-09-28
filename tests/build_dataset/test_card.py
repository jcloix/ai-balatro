# tests/test_dataset_card.py
from unittest.mock import patch, MagicMock
from PIL import Image
import pytest
import build_dataset.card as card
from build_dataset.build_config import CARD_AREAS, SCREEN_REGION
import os

# -------------------------
# Helper to create a real dummy image
# -------------------------
def create_dummy_image():
    return Image.new("RGB", (145, 200), color="red")

# -------------------------
# Test save_card_forced
# -------------------------
def test_save_card_forced_returns_filename():
    """save_card_forced returns a filename and calls save() on the cropped card."""
    dummy_img = create_dummy_image()
    # Patch pyautogui.screenshot to return our dummy image
    with patch("build_dataset.card.pyautogui.screenshot", return_value=dummy_img), \
         patch.object(dummy_img, "crop", return_value=dummy_img) as mock_crop, \
         patch.object(dummy_img, "save") as mock_save:
        
        unique, filename = card.save_card_forced(0, CARD_AREAS, SCREEN_REGION)

        # Ensure crop() and save() were called
        mock_crop.assert_called_once()
        mock_save.assert_called_once()
        assert unique is True
        assert filename.startswith("forced_cluster0_card0_id")
        # Optionally check that the file path is in OUTPUT_DIR
        assert os.path.dirname(os.path.join(card.OUTPUT_DIR, filename)) == card.OUTPUT_DIR

# -------------------------
# Test capture_cards calls save_card for each card
# -------------------------
def test_capture_cards_calls_save_card_for_all_cards():
    """capture_cards calls save_card for each card in CARD_AREAS."""
    # Mock screenshot and cropped card
    screenshot_mock = MagicMock(spec=Image.Image)
    cropped_card_mock = MagicMock(spec=Image.Image)
    screenshot_mock.crop.return_value = cropped_card_mock

    # Patch pyautogui.screenshot and save_card
    with patch("build_dataset.card.pyautogui.screenshot", return_value=screenshot_mock), \
         patch("build_dataset.card.save_card") as mock_save_card:
        
        # Provide a dummy return value
        mock_save_card.return_value = (True, "dummy.png")
        
        card.capture_cards(show_only=False)

        # Assert save_card was called once per card area
        assert mock_save_card.call_count == len(CARD_AREAS)

# -------------------------
# Test capture_cards show_only
# -------------------------
def test_capture_cards_show_only():
    """capture_cards with show_only=True should call screenshot.show() instead of save_card."""
    dummy_img = create_dummy_image()
    # Patch pyautogui.screenshot to return our dummy image
    with patch("build_dataset.card.pyautogui.screenshot", return_value=dummy_img), \
         patch.object(dummy_img, "show") as mock_show:
        card.capture_cards(show_only=True)
        mock_show.assert_called_once()
