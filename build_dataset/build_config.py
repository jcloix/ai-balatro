# card_dataset/config.py
import os
from config.config import SCREENSHOT_DIR
# -----------------------------
# Screen and card areas (adjust to your screen)
# -----------------------------

# Output directory
OUTPUT_DIR = SCREENSHOT_DIR
HASH_FILE = os.path.join(SCREENSHOT_DIR, "seen_hashes.json")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# -----------------------------
# Thresholds for duplicate detection
# -----------------------------
STRICT_DUP_THR = 9       # very restricted (almost certain duplicate -> skip saving)
CANDIDATE_THR = 12       # restricted (likely same card -> assign same card_group_id)
CLUSTER_THR = 25         # very large group (loose cluster grouping)

# Automation interval in milliseconds
AUTO_INTERVAL = 2000
