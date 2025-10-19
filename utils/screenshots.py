# file: screenshots.py
import pyautogui
from config.config import CARD_AREAS, SCREEN_REGION

def capture_cards(region=SCREEN_REGION):
    screenshot = pyautogui.screenshot(region=region)
    results = []
    for _, area in enumerate(CARD_AREAS):
        ax, ay, w, h = area
        card = screenshot.crop((ax, ay, ax + w, ay + h))
        results.append(card)
    return results

def capture_card(num_card, region=SCREEN_REGION):
    """
    Capture only one specific card (1-based index) and return it as a PIL.Image object.
    """
    if num_card < 1 or num_card > len(CARD_AREAS):
        raise ValueError(f"Invalid card index {num_card}. Must be 1–{len(CARD_AREAS)}.")
    
    screenshot = pyautogui.screenshot(region=region)
    ax, ay, w, h = CARD_AREAS[num_card-1]
    card = screenshot.crop((ax, ay, ax + w, ay + h))
    return card