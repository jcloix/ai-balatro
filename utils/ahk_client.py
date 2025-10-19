import subprocess
import time
from utils.ahk_config import POSITIONS, WAIT_TIMES, AHK_PATH, CLICK_SCRIPT



class AHKClient:
    """
    Handles executing AHK actions from Python.
    Positions and timings are centralized in ahk_config.py.
    """

    def __init__(self):
        pass

    # -----------------------------
    # Core method
    # -----------------------------
    def run_ahk_click(self, x, y):
        """
        Run the generic AHK script to click a given position.
        """
        subprocess.run([AHK_PATH, CLICK_SCRIPT, str(x), str(y)])
        time.sleep(WAIT_TIMES["after_click"])

    # -----------------------------
    # High-level actions
    # -----------------------------
    def reroll(self):
        """
        Click the reroll button.
        """
        x, y = POSITIONS["reroll_button"]
        self.run_ahk_click(x, y)

    def buy_card(self, card_index):
        """
        Click a specific card slot.
        """
        key = f"buy_card_{card_index}"
        if key not in POSITIONS:
            raise ValueError(f"No position defined for card {card_index}")
        x, y = POSITIONS[key]
        self.run_ahk_click(x, y)

    def click(self, x, y):
        """
        Generic click action at any position.
        """
        self.run_ahk_click(x, y)
