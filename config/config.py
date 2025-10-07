# config.py

# --------------------------
# Dataset paths
# --------------------------
DATA_DIR = "data"
SCREENSHOT_DIR = f"{DATA_DIR}/screenshots"               # raw screenshots
DATASET_DIR = f"{DATA_DIR}/dataset_default"               # original picture to use as dataset
DATASET_AUGMENTED_DIR = f"{DATA_DIR}/augmented"      # augmented images
LABELS_FILE = f"{DATASET_DIR}/labels.json"              # manual annotations
AUGMENTED_LABELS_FILE = f"{DATASET_AUGMENTED_DIR}/augmented.json" # metadata for augmented images
MERGED_LABELS_FILE = f"{DATA_DIR}/merged_labels.json" # optional merged JSON for training

# --------------------------
# Models paths
# --------------------------
MODELS_DIR = f"{DATA_DIR}/models"                       # directory for saving models
BEST_MODEL_PATH = f"{DATA_DIR}/models/best_model.pth"    

# --------------------------
# Reference data
# --------------------------
CARDS_NAMES = f"{DATA_DIR}/reference_data/cards_names.json"                      

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
# Map field to list
# --------------------------
FIELD_LIST_MAP = {
    "type":CARD_TYPES,
    "rarity":RARITY_OPTIONS,
    "modifier":MODIFIER_OPTIONS,
}