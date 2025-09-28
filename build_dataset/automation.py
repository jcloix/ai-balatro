# card_dataset/automation.py
import imagehash  # make sure this is imported at the top
import threading
import socket
import pyautogui
from build_dataset.card import save_card, save_card_forced, capture_cards
from build_dataset.config import SCREEN_REGION, CARD_AREAS, AUTO_INTERVAL

auto_running = False
last_screen_hash = None

def trigger_listener():
    """Listen on socket for capture / force commands."""
    HOST, PORT = "127.0.0.1", 50007
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        while True:
            conn, _ = s.accept()
            with conn:
                data = conn.recv(1024).decode("utf-8").strip()
                if not data:
                    continue
                if data == "capture":
                    capture_cards(show_only=False)
                elif data.startswith("force"):
                    try:
                        idx = int(data[-1]) - 1
                        if 0 <= idx < len(CARD_AREAS):
                            save_card_forced(idx)
                    except Exception as e:
                        print("Bad force command:", data, e)

def start_listener():
    """Start the trigger_listener in a background daemon thread."""
    threading.Thread(target=trigger_listener, daemon=True).start()

def auto_capture(root):
    """Periodic screenshot checking and saving cards automatically."""
    global auto_running, last_screen_hash
    if not auto_running:
        return

    screenshot = pyautogui.screenshot(region=SCREEN_REGION)
    current_hash = imagehash.phash(screenshot)  # cheap whole-screen hash

    if last_screen_hash is None or abs(current_hash - last_screen_hash) > 5:
        last_screen_hash = current_hash
        print("New screen detected, saving cards...")
        for i, area in enumerate(CARD_AREAS):
            ax, ay, w, h = area
            card = screenshot.crop((ax, ay, ax + w, ay + h))
            unique, info = save_card(card)
            if unique:
                print(f"Auto-saved new card {i+1}: {info}")
            else:
                print(f"Auto-skip card {i+1} duplicate of {info}")
    else:
        print("No change detected")

    # schedule next capture
    root.after(AUTO_INTERVAL, lambda: auto_capture(root))

def start_auto(root):
    """Start automatic capture loop."""
    global auto_running
    if not auto_running:
        auto_running = True
        print("Automation started")
        auto_capture(root)

def stop_auto():
    """Stop automatic capture loop."""
    global auto_running
    auto_running = False
    print("Automation stopped")
