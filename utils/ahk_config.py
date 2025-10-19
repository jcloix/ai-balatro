# ahk_config.py
import os
# The path of the AHK scripts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Balatro/
AHK_PATH = "C:\\Program Files\\AutoHotkey\\v2\\AutoHotkey.exe"
CLICK_SCRIPT = os.path.join(BASE_DIR, "scripts", "ahk", "click_position.ahk")

# Coordinates are (x, y)
POSITIONS = {
    "reroll_button": (700, 590),
    "buy_card_1": (500, 600),
    "buy_card_2": (600, 600),
    "buy_card_3": (700, 600),
    # add more cards if needed
}

# Optional: wait times
WAIT_TIMES = {
    "after_click": 0.5,  # seconds
}
