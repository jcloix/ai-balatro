# ui_components.py
import streamlit as st
from PIL import Image
from annotate_dataset.config import CARD_TYPES, RARITY_OPTIONS, MODIFIER_OPTIONS, HELPER_KEYS

def display_card(image_path, width=200):
    st.image(Image.open(image_path), width=width)

def render_sidebar(total, labeled_count, unlabeled_count, unique_by_type):
    st.sidebar.title("📊 Labeling Progress")
    st.sidebar.write(f"✅ Labeled: {labeled_count}")
    st.sidebar.write(f"🕓 Unlabeled: {unlabeled_count}")
    st.sidebar.write(f"📦 Total: {total}")

    st.sidebar.markdown("### 🔎 Unique Cards by Type")
    st.sidebar.write(f"🃏 Joker: {len(unique_by_type['Joker'])}")
    st.sidebar.write(f"🪐 Planet: {len(unique_by_type['Planet'])}")
    st.sidebar.write(f"🔮 Tarot: {len(unique_by_type['Tarot'])}")
    st.sidebar.write(f"👻 Spectral: {len(unique_by_type['Spectral'])}")    

def helper_selectboxes(existing_by_type):
    col_j, col_p, col_t, col_s = st.columns(4)

    selected = {}
    for col, typ, key in zip(
        [col_j, col_p, col_t, col_s],
        CARD_TYPES,
        HELPER_KEYS
    ):
        with col:
            opts = [""] + existing_by_type.get(typ, [])
            selected_val = st.selectbox(typ, opts, key=key)
            selected[typ.lower()] = selected_val
    return selected

