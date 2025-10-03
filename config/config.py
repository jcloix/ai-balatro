# config.py

# --------------------------
# Dataset paths
# --------------------------
DATA_DIR = "data"
SCREENSHOT_DIR = f"{DATA_DIR}/unlabeled"               # raw screenshots
DATASET_DIR = f"{DATA_DIR}/cards"               # original picture to use as dataset
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
# Class options
# --------------------------
# The number of different classes (TODO: Find better way to have this number)
NB_CLASSES=170