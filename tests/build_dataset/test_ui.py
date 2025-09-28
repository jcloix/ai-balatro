# tests/test_ui.py
from unittest.mock import MagicMock, patch
import pytest
import tkinter as tk

import build_dataset.ui as ui
from build_dataset.build_config import CARD_AREAS, SCREEN_REGION


def test_toggle_regions_shows_and_hides():
    # Ensure overlays start empty
    ui.overlays.clear()

    root_mock = MagicMock()

    with patch("build_dataset.ui.tk.Toplevel") as ToplevelMock, \
         patch("build_dataset.ui.tk.Canvas") as CanvasMock:
        
        # Setup mocks to return MagicMocks
        ToplevelMock.return_value = MagicMock()
        CanvasMock.return_value = MagicMock()

        # Show regions
        ui.toggle_regions_wrapper(root=root_mock)
        assert len(ui.overlays) == len(CARD_AREAS)

        # Hide regions
        ui.toggle_regions_wrapper(root=root_mock)
        assert len(ui.overlays) == 0


@patch("build_dataset.ui.capture_cards")
@patch("build_dataset.ui.save_card_forced")
def test_build_ui_binds_keys(mock_save_card_forced, mock_capture_cards):
    """build_ui should bind buttons and shortcuts correctly."""
    root = ui.build_ui()
    
    # Check buttons exist
    button_texts = [child.cget("text") for child in root.winfo_children() if isinstance(child, tk.Button)]
    assert "Show Regions (S)" in button_texts
    assert "Hide Regions (H)" in button_texts

    # Check key bindings exist
    assert root.bind("s") is not None
    assert root.bind("h") is not None
    assert root.bind("c") is not None
    assert root.bind("v") is not None
