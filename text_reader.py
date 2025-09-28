import pyautogui
import pytesseract
from PIL import Image
import tkinter as tk
import time

# Make sure Tesseract path is correct
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Capture region size
CAP_WIDTH = 270
CAP_HEIGHT = 100


def capture_text(x, y, w=CAP_WIDTH, h=CAP_HEIGHT):
    """Capture a region near the mouse and extract text."""
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    text = pytesseract.image_to_string(screenshot)
    return text.strip()


def update_text():
    """Update OCR text and move windows near the mouse."""
    x, y = pyautogui.position()

    # Capture region to the left of the mouse
    scan_x = max(x - CAP_WIDTH, 0)  # prevent going off-screen
    scan_y = y

    # Capture and update text
    text = capture_text(scan_x, scan_y-100, CAP_WIDTH, CAP_HEIGHT)
    label.config(text=text if text else "(no text found)")

    # Move text window slightly offset from mouse (still on right)
    text_window.geometry(f"+{x+20}+{y+20}")

    # Move overlay rectangle to follow mouse on the left
    overlay.geometry(f"{CAP_WIDTH}x{CAP_HEIGHT}+{scan_x}+{scan_y-100}")

    # Repeat after 0.5 seconds
    text_window.after(500, update_text)


# -------------------------------
# Setup Tkinter text window
text_window = tk.Tk()
text_window.overrideredirect(True)  # Remove borders
text_window.attributes("-topmost", True)
text_window.configure(bg="black")
label = tk.Label(text_window, text="Waiting...", bg="black", fg="white", font=("Arial", 12))
label.pack()

# -------------------------------
# Setup overlay window (semi-transparent)
overlay = tk.Toplevel()
overlay.overrideredirect(True)
overlay.attributes("-topmost", True)
overlay.attributes("-alpha", 0.3)  # Transparency
overlay.configure(bg="red")

# -------------------------------
# Start updating loop
text_window.after(500, update_text)
text_window.mainloop()
