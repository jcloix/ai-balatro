# card_dataset/main.py
from build_dataset.hash_storage import load_hashes
from build_dataset.automation import start_listener
from build_dataset.ui import build_ui

# -----------------------------
# Load previous hashes
load_hashes()

# -----------------------------
# Start the socket listener in the background
start_listener()

# -----------------------------
# Build and run UI
root = build_ui()

# Optionally start auto_capture immediately
# start_auto(root)

root.mainloop()
