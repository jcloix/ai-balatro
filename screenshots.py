# updated_card_capture.py
import os
import ctypes
import pyautogui
from PIL import Image
import tkinter as tk
import imagehash
import json
import threading
import socket

# -----------------------------
# Make Windows DPI aware (fix screenshot vs overlay mismatch)
# -----------------------------
ctypes.windll.user32.SetProcessDPIAware()

# -----------------------------
# Configuration
"""
Strategy A : Take the full card, but we see some background
SCREEN_REGION = (845, 430, 575, 200)  # adjust to your screen
CARD_AREAS = [
    (0, 0, 145, 200),
    (145, 0, 145, 200),
    (290, 0, 145, 200),
    (435, 0, 145, 200),
]
Strategy B : Take only the card, but the cards are a bit cropped
SCREEN_REGION = (860, 444, 575, 173)  # adjust to your screen
CARD_AREAS = [
    (0, 0, 112, 173),
    (145, 0, 112, 173),
    (290, 0, 112, 173),
    (435, 0, 112, 173),
]
"""
#
# -----------------------------
SCREEN_REGION = (845, 430, 575, 200)  # adjust to your screen
CARD_AREAS = [
    (0, 0, 145, 200),
    (145, 0, 145, 200),
    (290, 0, 145, 200),
    (435, 0, 145, 200),
]

OUTPUT_DIR = "cards"
HASH_FILE = os.path.join(OUTPUT_DIR, "seen_hashes.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thresholds for duplicate detection (adjustable)
STRICT_DUP_THR = 9       # very restricted (almost certain duplicate -> skip saving)
CANDIDATE_THR = 12       # restricted (likely same card -> assign same card_group_id)
CLUSTER_THR = 33         # very large group (loose cluster grouping)

# Overlay list
overlays = []

# -----------------------------
# Load previous hashes (and counters)
# -----------------------------
seen_hashes = []  # list of dicts: ph, ah, dh, cluster, card_group, filename
cluster_counter = 0
card_group_counter = 0

if os.path.exists(HASH_FILE):
    with open(HASH_FILE, "r") as f:
        saved_data = json.load(f)
        for h in saved_data:
            # Backwards-compatible: if card_group missing, fall back to cluster
            card_group = h.get("card_group", h.get("cluster"))
            # Some older JSONs may not contain filename; keep it if present
            seen_hashes.append({
                "ph": imagehash.hex_to_hash(h["ph"]),
                "ah": imagehash.hex_to_hash(h["ah"]),
                "dh": imagehash.hex_to_hash(h["dh"]),
                "cluster": h["cluster"],
                "card_group": card_group,
                "filename": h.get("filename")
            })
        if seen_hashes:
            cluster_counter = max(h.get("cluster", 0) for h in seen_hashes)
            card_group_counter = max(h.get("card_group", 0) for h in seen_hashes)

# -----------------------------
# Automation Controls
# -----------------------------
auto_running = False  # flag to control automation
AUTO_INTERVAL = 2000  # milliseconds between captures (2 seconds, adjust as needed)
last_screen_hash = None

# -----------------------------
# Helper: persist seen_hashes
# -----------------------------
def save_hashes():
    data = []
    for h in seen_hashes:
        data.append({
            "ph": str(h["ph"]),
            "ah": str(h["ah"]),
            "dh": str(h["dh"]),
            "cluster": h["cluster"],
            "card_group": h["card_group"],
            "filename": h.get("filename")
        })
    with open(HASH_FILE, "w") as f:
        json.dump(data, f, indent=2)

# -----------------------------
# Socket trigger listener (AHK integration)
# -----------------------------
def trigger_listener():
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

# Run listener in background thread
threading.Thread(target=trigger_listener, daemon=True).start()

# -----------------------------
# Automation capture (optional)
# -----------------------------
def auto_capture():
    global auto_running, last_screen_hash
    if not auto_running:
        return

    screenshot = pyautogui.screenshot(region=SCREEN_REGION)
    current_hash = imagehash.phash(screenshot)

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

    root.after(AUTO_INTERVAL, auto_capture)

def start_auto():
    global auto_running
    if not auto_running:
        auto_running = True
        print("Automation started")
        auto_capture()

def stop_auto():
    global auto_running
    auto_running = False
    print("Automation stopped")

# -----------------------------
# UI & overlay helpers
# -----------------------------
def show_regions():
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

def toggle_regions(event=None):
    if overlays:
        hide_regions()
    else:
        show_regions()

# -----------------------------
# save_card: new logic (cluster + card_group + filename)
# -----------------------------
def save_card(card_image):
    """
    Save card with hierarchical grouping:
      - strict duplicate (<= STRICT_DUP_THR): skip saving, return matched filename
      - candidate (<= CANDIDATE_THR): save and assign same card_group_id as match
      - cluster (<= CLUSTER_THR): save, assign same cluster but NEW card_group_id
      - new: new cluster AND new card_group_id
    Returns (unique_bool, info_string)
    """
    global seen_hashes, cluster_counter, card_group_counter

    hashes = {
        "ph": imagehash.phash(card_image),
        "ah": imagehash.average_hash(card_image),
        "dh": imagehash.dhash(card_image)
    }

    # find closest existing entry (by max of the three distances)
    closest_idx = None
    min_distance = None
    for idx, h in enumerate(seen_hashes):
        ph_dist = hashes["ph"] - h["ph"]
        ah_dist = hashes["ah"] - h["ah"]
        dh_dist = hashes["dh"] - h["dh"]
        max_dist = max(ph_dist, ah_dist, dh_dist)
        if min_distance is None or max_dist < min_distance:
            min_distance = max_dist
            closest_idx = idx

    # If we found a close match, decide by thresholds
    if min_distance is not None:
        if min_distance <= STRICT_DUP_THR:
            # strict duplicate: skip saving
            matched_filename = seen_hashes[closest_idx].get("filename", f"card_{closest_idx+1}")
            return False, f"strict_duplicate_of:{matched_filename}"

        elif min_distance <= CANDIDATE_THR:
            # candidate: likely same card => assign same card_group_id and same cluster
            cluster_id = seen_hashes[closest_idx]["cluster"]
            card_group_id = seen_hashes[closest_idx]["card_group"]
            label_cluster = cluster_id
            label_card_group = card_group_id

        elif min_distance <= CLUSTER_THR:
            # loose cluster: same cluster, but create a NEW card_group within cluster
            cluster_id = seen_hashes[closest_idx]["cluster"]
            cluster_counter_val = cluster_id  # keep same cluster
            # new card_group id
            card_group_counter += 1
            card_group_id = card_group_counter
            label_cluster = cluster_id
            label_card_group = card_group_id

        else:
            # far from all existing: new cluster & new card_group
            cluster_counter += 1
            card_group_counter += 1
            label_cluster = cluster_counter
            label_card_group = card_group_counter
    else:
        # no existing entries: first card ever
        cluster_counter += 1
        card_group_counter += 1
        label_cluster = cluster_counter
        label_card_group = card_group_counter

    # create filename and save
    hash_prefix = str(hashes["ph"])[:8]
    filename = f"cluster{label_cluster}_card{label_card_group}_id{len(seen_hashes)+1}.png"
    card_image.save(os.path.join(OUTPUT_DIR, filename))

    # append metadata to seen_hashes and persist
    seen_hashes.append({
        "ph": hashes["ph"],
        "ah": hashes["ah"],
        "dh": hashes["dh"],
        "cluster": label_cluster,
        "card_group": label_card_group,
        "filename": filename
    })
    save_hashes()
    return True, filename

# -----------------------------
# Forced save (no duplicate logic)
# -----------------------------
def save_card_forced(card_index):
    screenshot = pyautogui.screenshot(region=SCREEN_REGION)
    ax, ay, w, h = CARD_AREAS[card_index]
    card = screenshot.crop((ax, ay, ax + w, ay + h))
    hash_prefix = str(imagehash.phash(card))[:8]
    filename = f"forced_cluster0_card0_id{len(seen_hashes)+1}_{hash_prefix}.png"
    card.save(os.path.join(OUTPUT_DIR, filename))
    print(f"Force-saved card {card_index+1} as {filename}")

# -----------------------------
# Capture wrapper
# -----------------------------
def capture_cards(show_only=False):
    screenshot = pyautogui.screenshot(region=SCREEN_REGION)
    if show_only:
        screenshot.show()
        return

    for i, area in enumerate(CARD_AREAS):
        ax, ay, w, h = area
        card = screenshot.crop((ax, ay, ax + w, ay + h))
        unique, info = save_card(card)
        if unique:
            print(f"Saved new card {i+1}: {info}")
        else:
            print(f"Card {i+1} duplicate of {info}")

# -----------------------------
# Setup Tkinter UI
# -----------------------------
root = tk.Tk()
root.title("Card Capture")
root.geometry("300x180")

btn_show = tk.Button(root, text="Show Regions (S)", command=show_regions)
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
root.bind("s", toggle_regions)
root.bind("h", toggle_regions)
root.bind("c", lambda e: capture_cards(show_only=True))
root.bind("v", lambda e: capture_cards(show_only=False))
root.bind("1", lambda e: save_card_forced(0))
root.bind("2", lambda e: save_card_forced(1))
root.bind("3", lambda e: save_card_forced(2))
root.bind("4", lambda e: save_card_forced(3))
root.bind("a", lambda e: start_auto())
root.bind("x", lambda e: stop_auto())


root.mainloop()
