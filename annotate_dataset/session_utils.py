# session_utils.py
import streamlit as st
from annotate_dataset.annotate_config import DEFAULT_INPUTS, HELPER_KEYS, INPUT_TO_FIELD_MAP

def init_session_state():
    if "group_idx" not in st.session_state:
        st.session_state.group_idx = 0
    if "group_section" not in st.session_state:
        st.session_state.group_section = {}
    if "cluster_section" not in st.session_state:
        st.session_state.cluster_section = {}

def init_session_state_inputs(currentFile, prefill_map):
    # Clean inputs session and prefill with explicit values. 
    for k in DEFAULT_INPUTS.keys():
        if f"{k}_to_fill" in st.session_state:  
            st.session_state[k] = st.session_state.pop(f"{k}_to_fill")
    # Prefill inputs (only the first time)
    prefill_inputs(currentFile, prefill_map)


def prefill_inputs(currentFile, prefill_map):
    if "prefilled_card" not in st.session_state or st.session_state["prefilled_card"] != currentFile:
        card_info = prefill_map[currentFile] if prefill_map else None
        for k in [k for k in DEFAULT_INPUTS.keys() if k in INPUT_TO_FIELD_MAP]: 
            st.session_state[k] = card_info.get(INPUT_TO_FIELD_MAP[k],DEFAULT_INPUTS[k])
        st.session_state["prefilled_card"] = currentFile


def clear_fields():
    clear_helpers()
    st.session_state["card_name_to_fill"] = ""
    st.session_state["card_type_to_fill"] = DEFAULT_INPUTS["card_type"]
    st.session_state["card_rarity_to_fill"] = DEFAULT_INPUTS["card_rarity"]
    st.session_state["card_modifier_to_fill"] = DEFAULT_INPUTS["card_modifier"]
    st.session_state["session_cleared"] = True

def clear_helpers(type=None):
    for t, k in zip(["joker", "planet", "tarot", "spectral"], HELPER_KEYS):
        if not type or t != type:
            st.session_state[f"{k}_to_fill"] = ""
            st.session_state[f"_prev_{t}"] = None