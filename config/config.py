# config.py

# --------------------------
# Dataset paths
# --------------------------
DATA_DIR = "data"
DATASET_DIR = f"{DATA_DIR}/unlabeled"               # raw screenshots
DATASET_AUGMENTED_DIR = f"{DATA_DIR}/augmented"      # augmented images
LABELS_FILE = f"{DATA_DIR}/labels.json"              # manual annotations
AUGMENTED_LABELS_FILE = f"{DATA_DIR}/augmented.json" # metadata for augmented images
MERGED_LABELS_FILE = f"{DATA_DIR}/merged_labels.json" # optional merged JSON for training
MODELS_DIR = f"{DATA_DIR}/models"                       # directory for saving models

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
