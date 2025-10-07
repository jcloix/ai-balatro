# main.py

import os
import json
import streamlit as st
from pathlib import Path
from PIL import Image
from config.config import DATASET_DIR, LABELS_FILE, CARD_TYPES, RARITY_OPTIONS, MODIFIER_OPTIONS
from annotate_dataset.annotate_config import HELPER_KEYS
from annotate_dataset.data_loader import load_labels, list_images, build_maps, get_unlabeled_groups, compute_unique_by_type, compute_unique_by_field
from annotate_dataset.session_utils import init_session_state, init_session_state_inputs, clear_helpers, clear_fields
from annotate_dataset.ui_components import display_card, render_sidebar, helper_selectboxes

# --------------------------
# Streamlit config
# --------------------------
st.set_page_config(page_title="Card Labeler", layout="wide")

# --------------------------
# Load data
# --------------------------
labels = load_labels()
all_images = list_images()
maps_obj  = build_maps(all_images)
# Extract maps
card_group_map = maps_obj.group_map
cluster_map = maps_obj.cluster_map
file_map = maps_obj.file_map
prefill_map = getattr(maps_obj, "prefill_map", {})

unlabeled_card_groups = get_unlabeled_groups(card_group_map, labels)
unique_by_type = compute_unique_by_field(labels, "type")
unique_by_modifier = compute_unique_by_field(labels, "modifier")
total = len(all_images)
labeled_count = sum(1 for f in all_images if f in labels)
unlabeled_count = total - labeled_count

# --------------------------
# Initialize session
# --------------------------
init_session_state()

# --------------------------
# Sidebar progress
# --------------------------
render_sidebar(total, labeled_count, unlabeled_count, unique_by_type, unique_by_modifier)

# --------------------------
# Main app logic
# --------------------------

def inputs_form(labels, unique_by_type, current_file=None):
    # Existing names
    existing_by_type = {t: sorted(unique_by_type[t]) for t in CARD_TYPES}
    # Input columns
    cols_inputs = st.columns([2, 1, 1, 1])

    with cols_inputs[0]:
        typed_name = st.text_input("Card Name", key="card_name")
        typed_name = typed_name.capitalize() if typed_name else typed_name

    # Helper selectboxes
    selected_helpers = helper_selectboxes(existing_by_type)

    # Handle helper auto-fill
    for typ, widget_key in zip(["joker","planet","tarot","spectral"], HELPER_KEYS):
        sel = st.session_state.get(widget_key, "")
        prev = st.session_state.get(f"_prev_{typ}")
        if sel and sel != prev:
            st.session_state["card_name_to_fill"] = sel
            st.session_state["card_type_to_fill"] = typ.capitalize()
            if typ == "joker":
                st.session_state["card_rarity_to_fill"] = next(
                    (v["rarity"] for v in labels.values() if v.get("name") == sel and "rarity" in v),
                    "Common"
                )
            else:
                st.session_state["card_rarity_to_fill"] = "Common"
            st.session_state["card_modifier_to_fill"] = "Base"
            st.session_state[f"_prev_{typ}"] = sel
            clear_helpers(typ)
            st.rerun()

    # Type
    with cols_inputs[1]:
        card_type = st.selectbox(
            "Type", CARD_TYPES,
            key="card_type"
        )

    # Rarity
    with cols_inputs[2]:
        rarity = st.selectbox(
            "Rarity", RARITY_OPTIONS,
            key="card_rarity",
            disabled=(card_type != "Joker")
        )

    # Modifier
    with cols_inputs[3]:
        modifier = st.selectbox(
            "Modifier", MODIFIER_OPTIONS,
            key="card_modifier",
            disabled=(card_type != "Joker")
        )

def display_main_card(current_file):
    st.title(f"Label Card : {current_file}")
    st.subheader("Current Card")
    display_card(os.path.join(DATASET_DIR, current_file), width=200)

def group_section(group_files, labels):
    # Initialize section only once
    if "group_section" not in st.session_state or not st.session_state["group_section"]:
        st.session_state["group_section"] = {f: True for f in group_files if f not in labels}
        # Initialize checkbox keys directly in session_state
        for f in st.session_state["group_section"]:
            key = f"group_{f}"
            if key not in st.session_state:
                st.session_state[key] = st.session_state["group_section"][f]

    st.subheader("Cards of same group (to be labeled together)")

    # Buttons side by side (compact layout)
    col1, col2, _ = st.columns([0.25, 0.25, 1])
    with col1:
        if st.button("Unselect All"):
            for f in st.session_state["group_section"]:
                st.session_state[f"group_{f}"] = False
            st.rerun()
    with col2:
        if st.button("Select All"):
            for f in st.session_state["group_section"]:
                st.session_state[f"group_{f}"] = True
            st.rerun()

    # Display checkboxes without explicit value argument
    num_cols = 14
    cols = st.columns(num_cols)
    for idx, f in enumerate(st.session_state["group_section"].keys()):
        key = f"group_{f}"
        with cols[idx % num_cols]:
            st.image(Image.open(os.path.join(DATASET_DIR, f)), width=80)
            st.checkbox("Include", key=key)

def display_save_button(labels):
    if st.button("Save Labels for Current Group"):
        name_to_save = st.session_state["card_name"]
        type_to_save = st.session_state["card_type"]
        rarity_to_save = st.session_state["card_rarity"]
        modifier_to_save = st.session_state["card_modifier"]

        # Read checkboxes dynamically from session_state
        if "group_section" in st.session_state:
            for f in list(st.session_state["group_section"].keys()):
                checkbox_key = f"group_{f}"
                include = st.session_state.get(checkbox_key, False)
                st.session_state["group_section"][f] = include  # sync for consistency

                if include:
                    labels[f] = {
                        "name": name_to_save,
                        "type": type_to_save,
                        "rarity": rarity_to_save,
                        "modifier": modifier_to_save,
                        "full_path": str(Path(DATASET_DIR) / f).replace("\\", "/")
                    }
        with open(LABELS_FILE, "w") as fw:
            json.dump(labels, fw, indent=2)

        st.session_state["group_idx"] += 1
        st.session_state["group_section"] = {}
        st.session_state["cluster_section"] = {}
        clear_fields() 

        st.success(f"Labeled {name_to_save} ✅")
        st.rerun()

def cluster_section(cluster_id, group_files, labels):
    if not st.session_state["cluster_section"]:
        cluster_files = [f for f in cluster_map.get(cluster_id, []) if f not in group_files and f not in labels]
        st.session_state["cluster_section"] = {f: False for f in cluster_files}
    st.subheader("Cards of same cluster (not labeled together)")
    num_cols_cluster = 14
    cols = st.columns(num_cols_cluster)
    cluster_keys = list(st.session_state["cluster_section"].keys())
    for idx, f in enumerate(cluster_keys):
        with cols[idx % num_cols_cluster]:
            st.image(Image.open(os.path.join(DATASET_DIR, f)), width=80)
            if st.button("Add", key=f"add_{f}"):
                st.session_state["group_section"][f] = True
                del st.session_state["cluster_section"][f]
                st.rerun()


def display_body():
    if not unlabeled_card_groups or st.session_state["group_idx"] >= len(unlabeled_card_groups):
        st.success("🎉 All card groups labeled!")
        return # exit early
    current_group_id = unlabeled_card_groups[st.session_state["group_idx"]]
    group_files = card_group_map[current_group_id]

    # Pick first unlabeled card
    current_file = next(f for f in group_files if f not in labels)

    # Prefill the values of the inputs
    init_session_state_inputs(current_file,prefill_map)

    # Display main card
    display_main_card(current_file)
    
    # Inputs form
    inputs_form(labels, unique_by_type)
    
    # --------------------------
    # Cards of same group
    # --------------------------
    group_section(group_files, labels)
    # --------------------------
    # Save button
    # --------------------------
    display_save_button(labels)

    # --------------------------
    # Cards of same cluster
    # --------------------------
    cluster_id = file_map[current_file]["cluster_id"]
    cluster_section(cluster_id, group_files, labels)

display_body()