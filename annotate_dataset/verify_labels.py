import streamlit as st
import os
from collections import defaultdict
from PIL import Image
from config.config import DATASET_DIR
from annotate_dataset.data_loader import load_labels

# --------------------------
# Streamlit config
# --------------------------
st.set_page_config(page_title="Card Label Verifier", layout="wide")

# --------------------------
# Load data
# --------------------------
labels = load_labels()

st.title("🃏 Verify Annotated Cards")

# Group cards by (name, modifier)
grouped_cards = defaultdict(list)
for filename, info in labels.items():
    key = (info["name"], info["modifier"])
    grouped_cards[key].append(filename)

st.write(f"Loaded {len(labels)} cards, grouped into {len(grouped_cards)} unique name/modifier pairs.")

# Display each group in a compact style
for (name, modifier), filenames in grouped_cards.items():
    # Grab type & rarity from the first card
    first_info = labels[filenames[0]]
    card_type = first_info.get("type", "?")
    rarity = first_info.get("rarity", "?")

    # Compact group header
    st.markdown(
        f"**{name} — {modifier}** "
        f"({card_type}, {rarity}, {len(filenames)} cards)"
    )

    cols = st.columns(10)
    for col, filename in zip(cols, filenames):
        img_path = os.path.join(DATASET_DIR, filename)
        if os.path.exists(img_path):
            image = Image.open(img_path)
            col.image(image, width=70)
            col.caption(filename, help=filename)
        else:
            col.text("❌ Missing")
    st.divider()
