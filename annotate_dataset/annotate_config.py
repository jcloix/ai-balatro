# config.py

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

INPUT_TO_FIELD_MAP = {
    "card_name": "name",
    "card_type": "type",
    "card_rarity": "rarity",
    "card_modifier": "modifier",
}

# --------------------------
# Inference constants (use for semi-automatic annotate)
# --------------------------

TRESHOLD_TOP2 = 0.10