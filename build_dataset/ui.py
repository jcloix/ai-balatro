# card_dataset/ui.py
import tkinter as tk
from build_dataset.card import capture_cards, save_card_forced
from build_dataset.automation import start_auto, stop_auto
from build_dataset.build_config import SCREEN_REGION, CARD_AREAS

overlays = []

def show_regions(root):
    global overlays
    if overlays:
        return
    x0, y0, _, _ = SCREEN_REGION
    for area in CARD_AREAS:
        ax, ay, w, h = area
        overlay = tk.Toplevel(root)
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        overlay.attributes("-alpha", 0.3)
        overlay.geometry(f"{w}x{h}+{x0+ax}+{y0+ay}")
        canvas = tk.Canvas(overlay, width=w, height=h, highlightthickness=0)
        canvas.pack()
        canvas.create_rectangle(0, 0, w, h, fill="red")
        overlays.append(overlay)

def hide_regions():
    global overlays
    for overlay in overlays:
        overlay.destroy()
    overlays = []

def toggle_regions_wrapper(event=None, root=None):
    if overlays:
        hide_regions()
    else:
        show_regions(root)

def build_ui():
    root = tk.Tk()
    root.title("Card Capture")
    root.geometry("300x180")

    # Buttons
    btn_show = tk.Button(root, text="Show Regions (S)", command=lambda: show_regions(root))
    btn_show.pack(pady=5)

    btn_hide = tk.Button(root, text="Hide Regions (H)", command=hide_regions)
    btn_hide.pack(pady=5)

    btn_capture = tk.Button(root, text="Capture Screenshot (C)", command=lambda: capture_cards(show_only=True))
    btn_capture.pack(pady=5)

    btn_save = tk.Button(root, text="Capture & Save Cards (V)", command=lambda: capture_cards(show_only=False))
    btn_save.pack(pady=5)

    btn_start_auto = tk.Button(root, text="Start Auto (A)", command=start_auto)
    btn_start_auto.pack(pady=5)

    btn_stop_auto = tk.Button(root, text="Stop Auto (X)", command=stop_auto)
    btn_stop_auto.pack(pady=5)

    # Keyboard shortcuts
    root.bind("s", lambda e: toggle_regions_wrapper(root=root))
    root.bind("h", lambda e: toggle_regions_wrapper(root=root))
    root.bind("c", lambda e: capture_cards(show_only=True))
    root.bind("v", lambda e: capture_cards(show_only=False))
    root.bind("1", lambda e: save_card_forced(0, CARD_AREAS, SCREEN_REGION))
    root.bind("2", lambda e: save_card_forced(1, CARD_AREAS, SCREEN_REGION))
    root.bind("3", lambda e: save_card_forced(2, CARD_AREAS, SCREEN_REGION))
    root.bind("4", lambda e: save_card_forced(3, CARD_AREAS, SCREEN_REGION))
    root.bind("a", lambda e: start_auto())
    root.bind("x", lambda e: stop_auto())

    return root
