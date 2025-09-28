# config.py

# --------------------------
# Dataset paths
# --------------------------
DATASET_DIR = "data/unlabeled"
LABELS_FILE = "annotate_dataset/labels.json"

# --------------------------
# Card types
# --------------------------
CARD_TYPES = ["Joker", "Planet", "Tarot", "Spectral"]

# --------------------------
# Rarity options
# --------------------------
RARITY_OPTIONS = ["Common", "Uncommon", "Rare"]

# --------------------------
# Modifier options
# --------------------------
MODIFIER_OPTIONS = ["Base", "Foil", "Holographic", "Polychrome", "Negative"]

# --------------------------
# Helper selectbox keys
# --------------------------
HELPER_KEYS = ["select_joker", "select_planet", "select_tarot", "select_spectral"]

# --------------------------
# Default central input values
# --------------------------
DEFAULT_INPUTS = {
    "card_name": "",
    "card_type": "Joker",
    "card_rarity": "Common",
    "card_modifier": "Base",
    "select_joker": "",
    "select_planet": "",
    "select_tarot": "",
    "select_spectral": ""
}
