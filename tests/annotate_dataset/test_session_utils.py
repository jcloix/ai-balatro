# tests/annotate_dataset/test_session_utils.py
import unittest
from unittest.mock import patch
import streamlit as st
from annotate_dataset import session_utils
from annotate_dataset.config import DEFAULT_INPUTS, HELPER_KEYS

class TestSessionUtils(unittest.TestCase):

    def setUp(self):
        # Clear the Streamlit session state before each test
        st.session_state.clear()

    # ---------------------------
    # Test init_session_state
    # ---------------------------
    def test_init_session_state_empty(self):
        # Nothing in st.session_state yet
        session_utils.init_session_state()
        self.assertIn("group_idx", st.session_state)
        self.assertEqual(st.session_state.group_idx, 0)
        self.assertIn("group_section", st.session_state)
        self.assertEqual(st.session_state.group_section, {})
        self.assertIn("cluster_section", st.session_state)
        self.assertEqual(st.session_state.cluster_section, {})

    def test_init_session_state_with_to_fill(self):
        # Populate _to_fill keys
        for k in DEFAULT_INPUTS.keys():
            st.session_state[f"{k}_to_fill"] = "value"
        session_utils.init_session_state()
        for k in DEFAULT_INPUTS.keys():
            self.assertEqual(st.session_state[k], "value")
            self.assertNotIn(f"{k}_to_fill", st.session_state)

    # ---------------------------
    # Test clear_helpers
    # ---------------------------
    def test_clear_helpers_all(self):
        # Populate helper keys
        for t, k in zip(["joker","planet","tarot","spectral"], HELPER_KEYS):
            st.session_state[f"{k}_to_fill"] = "value"
            st.session_state[f"_prev_{t}"] = "prev"
        session_utils.clear_helpers()
        for t, k in zip(["joker","planet","tarot","spectral"], HELPER_KEYS):
            self.assertEqual(st.session_state[f"{k}_to_fill"], "")
            self.assertIsNone(st.session_state[f"_prev_{t}"])

    def test_clear_helpers_exclude_type(self):
        # Populate helper keys
        for t, k in zip(["joker","planet","tarot","spectral"], HELPER_KEYS):
            st.session_state[f"{k}_to_fill"] = "value"
            st.session_state[f"_prev_{t}"] = "prev"
        session_utils.clear_helpers(type="joker")
        # joker should remain
        self.assertEqual(st.session_state[f"{HELPER_KEYS[0]}_to_fill"], "value")
        self.assertEqual(st.session_state[f"_prev_joker"], "prev")
        # others cleared
        for t, k in zip(["planet","tarot","spectral"], HELPER_KEYS[1:]):
            self.assertEqual(st.session_state[f"{k}_to_fill"], "")
            self.assertIsNone(st.session_state[f"_prev_{t}"])

    # ---------------------------
    # Test clear_fields
    # ---------------------------
    @patch("annotate_dataset.session_utils.clear_helpers")
    def test_clear_fields(self, mock_clear_helpers):
        # Populate session_state with arbitrary values
        for k in ["card_name_to_fill","card_type_to_fill","card_rarity_to_fill","card_modifier_to_fill"]:
            st.session_state[k] = "old"
        session_utils.clear_fields()
        # clear_helpers should be called
        mock_clear_helpers.assert_called_once()
        # Fields reset
        self.assertEqual(st.session_state["card_name_to_fill"], "")
        self.assertEqual(st.session_state["card_type_to_fill"], DEFAULT_INPUTS["card_type"])
        self.assertEqual(st.session_state["card_rarity_to_fill"], DEFAULT_INPUTS["card_rarity"])
        self.assertEqual(st.session_state["card_modifier_to_fill"], DEFAULT_INPUTS["card_modifier"])

if __name__ == "__main__":
    unittest.main()
