# main.py

import os
import json
import streamlit as st
from PIL import Image
from config.config import DATASET_DIR, LABELS_FILE, CARD_TYPES, RARITY_OPTIONS, MODIFIER_OPTIONS
from annotate_dataset.annotate_config import HELPER_KEYS
from annotate_dataset.data_loader import load_labels, list_images, build_maps, get_unlabeled_groups, parse_ids, compute_unique_by_type
from annotate_dataset.session_utils import init_session_state, clear_helpers, clear_fields
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
card_group_map, cluster_map = build_maps(all_images)
unlabeled_card_groups = get_unlabeled_groups(card_group_map, labels)
unique_by_type = compute_unique_by_type(labels)
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
render_sidebar(total, labeled_count, unlabeled_count, unique_by_type)

# --------------------------
# Main app logic
# --------------------------

def inputs_form(labels,unique_by_type):
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

def display_main_card(current_file, current_group_id, cluster_id):
    st.title(f"Label Card Group ID {current_group_id} (Cluster {cluster_id})")
    st.subheader("Current Card")
    display_card(os.path.join(DATASET_DIR, current_file), width=200)

def group_section(group_files, labels):
     # Initialize sections
    if not st.session_state["group_section"]:
        st.session_state["group_section"] = {f: True for f in group_files if f not in labels}
    st.subheader("Cards of same group (to be labeled together)")
    num_cols = 14
    cols = st.columns(num_cols)
    for idx, f in enumerate(list(st.session_state["group_section"].keys())):
        with cols[idx % num_cols]:
            st.image(Image.open(os.path.join(DATASET_DIR, f)), width=80)
            st.session_state["group_section"][f] = st.checkbox(
                "Include", key=f"group_{f}", value=st.session_state["group_section"][f]
            )

def display_save_button(cluster_id, current_group_id, labels):
    if st.button("Save Labels for Current Group"):
        name_to_save = st.session_state["card_name"]
        type_to_save = st.session_state["card_type"]
        rarity_to_save = st.session_state["card_rarity"]
        modifier_to_save = st.session_state["card_modifier"]

        if "group_section" in st.session_state:
            for f, include in st.session_state["group_section"].items():
                if include:
                    labels[f] = {
                        "name": name_to_save,
                        "type": type_to_save,
                        "rarity": rarity_to_save,
                        "modifier": modifier_to_save,
                        "cluster": cluster_id,
                        "card_group": current_group_id
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
        cluster_files = [f for f in cluster_map[cluster_id] if f not in group_files and f not in labels]
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
    cluster_id, _ = parse_ids(current_file)

    # Display main card
    display_main_card(current_file, current_group_id, cluster_id)
    
    # Inputs form
    inputs_form(labels, unique_by_type)
    
    # --------------------------
    # Cards of same group
    # --------------------------
    group_section(group_files, labels)
    # --------------------------
    # Save button
    # --------------------------
    display_save_button(cluster_id, current_group_id, labels)

    # --------------------------
    # Cards of same cluster
    # --------------------------
    cluster_section(cluster_id, group_files, labels)

display_body()