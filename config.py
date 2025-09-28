# config.py
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

OUTPUT_DIR = "cards"
HASH_FILE = "cards/seen_hashes.json"

STRICT_DUP_THR = 9
CANDIDATE_THR = 12
CLUSTER_THR = 33

AUTO_INTERVAL = 2000  # milliseconds
