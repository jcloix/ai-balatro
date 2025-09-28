# card_dataset/config.py
import os
from config.config import DATASET_DIR
# -----------------------------
# Screen and card areas (adjust to your screen)
# -----------------------------
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
SCREEN_REGION = (845, 430, 575, 200)
CARD_AREAS = [
    (0, 0, 145, 200),
    (145, 0, 145, 200),
    (290, 0, 145, 200),
    (435, 0, 145, 200),
]

# Output directory
OUTPUT_DIR = DATASET_DIR
HASH_FILE = os.path.join(OUTPUT_DIR, "seen_hashes.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Thresholds for duplicate detection
# -----------------------------
STRICT_DUP_THR = 9       # very restricted (almost certain duplicate -> skip saving)
CANDIDATE_THR = 12       # restricted (likely same card -> assign same card_group_id)
CLUSTER_THR = 25         # very large group (loose cluster grouping)

# Automation interval in milliseconds
AUTO_INTERVAL = 2000
