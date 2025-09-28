# session_utils.py
import streamlit as st
from annotate_dataset.config import DEFAULT_INPUTS, HELPER_KEYS

def init_session_state():
    if "group_idx" not in st.session_state:
        st.session_state.group_idx = 0
    if "group_section" not in st.session_state:
        st.session_state.group_section = {}
    if "cluster_section" not in st.session_state:
        st.session_state.cluster_section = {}

    for k in DEFAULT_INPUTS.keys():
        if f"{k}_to_fill" in st.session_state:
            st.session_state[k] = st.session_state.pop(f"{k}_to_fill")


def clear_fields():
    clear_helpers()
    st.session_state["card_name_to_fill"] = ""
    st.session_state["card_type_to_fill"] = DEFAULT_INPUTS["card_type"]
    st.session_state["card_rarity_to_fill"] = DEFAULT_INPUTS["card_rarity"]
    st.session_state["card_modifier_to_fill"] = DEFAULT_INPUTS["card_modifier"]

def clear_helpers(type=None):
    for t, k in zip(["joker", "planet", "tarot", "spectral"], HELPER_KEYS):
        if not type or t != type:
            st.session_state[f"{k}_to_fill"] = ""
            st.session_state[f"_prev_{t}"] = None