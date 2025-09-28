# tests/test_automation.py
from unittest.mock import patch, MagicMock, ANY
from PIL import Image
import threading

import build_dataset.automation as automation
from build_dataset.config import CARD_AREAS, AUTO_INTERVAL


def test_start_stop_auto_sets_flag():
    """start_auto sets the flag, stop_auto clears it."""
    root_mock = MagicMock()
    automation.auto_running = False

    # Patch auto_capture so it does not run actual code
    with patch("build_dataset.automation.auto_capture"):
        automation.start_auto(root_mock)
        assert automation.auto_running is True

    automation.stop_auto()
    assert automation.auto_running is False


@patch("build_dataset.automation.save_card_forced")  # ensure forced saves don't touch files
@patch("build_dataset.automation.save_card")         # ensure normal saves don't touch files
@patch("build_dataset.automation.pyautogui.screenshot")
def test_auto_capture_new_screen(mock_screenshot, mock_save_card, mock_save_card_forced):
    """auto_capture saves cards when screen changes, but safely."""
    root_mock = MagicMock()
    automation.auto_running = True
    automation.last_screen_hash = None

    # Provide a real dummy image for phash
    dummy_img = Image.new("RGB", (64, 64))
    mock_screenshot.return_value = dummy_img

    # Mock save_card so no real files are written
    mock_save_card.return_value = (True, "dummy.png")
    mock_save_card_forced.return_value = (True, "dummy.png")

    automation.auto_capture(root_mock)

    # Check that root.after schedules next capture
    root_mock.after.assert_called_with(AUTO_INTERVAL, ANY)


@patch("build_dataset.automation.save_card_forced")
@patch("build_dataset.automation.save_card")
@patch("build_dataset.automation.pyautogui.screenshot")
def test_auto_capture_no_change(mock_screenshot, mock_save_card, mock_save_card_forced):
    """auto_capture does not save if screen hash is unchanged."""
    root_mock = MagicMock()
    automation.auto_running = True

    dummy_img = Image.new("RGB", (64, 64))
    mock_screenshot.return_value = dummy_img

    # Compute hash once so next capture sees no change
    import imagehash
    automation.last_screen_hash = imagehash.phash(dummy_img)

    automation.auto_capture(root_mock)

    # save_card should not be called
    mock_save_card.assert_not_called()
    mock_save_card_forced.assert_not_called()


def test_start_listener_starts_thread():
    """start_listener launches a daemon thread."""
    with patch("socket.socket"):  # prevent real socket binding
        thread_before = threading.enumerate()
        automation.start_listener()

        thread_after = threading.enumerate()
        assert len(thread_after) > len(thread_before)

        daemon_threads = [t for t in thread_after if t not in thread_before]
        assert all(t.daemon for t in daemon_threads)
