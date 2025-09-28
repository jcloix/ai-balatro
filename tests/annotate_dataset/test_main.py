# tests/annotate_dataset/test_main.py
import unittest
from unittest.mock import patch, MagicMock
from annotate_dataset import main
from types import SimpleNamespace

class TestMain(unittest.TestCase):

    @patch("annotate_dataset.main.st")
    @patch("annotate_dataset.main.display_card")
    @patch("annotate_dataset.main.helper_selectboxes")
    @patch("annotate_dataset.main.parse_ids")
    def test_display_main_card_and_inputs_form(self, mock_parse_ids, mock_helpers, mock_display_card, mock_st):
        # Setup
        mock_parse_ids.return_value = (1, 10)
        mock_st.session_state = {
            "card_name": "",
            "card_type": "Joker",
            "card_rarity": "Common",
            "card_modifier": "Base",
            "group_section": {},
            "cluster_section": {}
        }
        labels = {"file1.png": {"name": "Card1", "type": "Joker"}}
        unique_by_type = {"Joker": ["Card1"], "Planet": [], "Tarot": [], "Spectral": []}
        
        # Run inputs_form
        main.inputs_form(labels, unique_by_type)
        mock_helpers.assert_called_once()  # helper_selectboxes called

    @patch("annotate_dataset.main.st")
    @patch("annotate_dataset.main.display_card")
    @patch("annotate_dataset.main.parse_ids")
    def test_display_main_card(self, mock_parse_ids, mock_display_card, mock_st):
        mock_parse_ids.return_value = (1, 10)
        mock_st.session_state = {}
        main.display_main_card("file1.png", 10, 1)
        mock_display_card.assert_called_once()

    from types import SimpleNamespace

    @patch("annotate_dataset.main.st")
    @patch("builtins.open", new_callable=MagicMock)
    def test_display_save_button(self, mock_open_file, mock_st):
        # Setup session_state as a plain dict
        mock_st.session_state = {
            "card_name": "CardX",
            "card_type": "Joker",
            "card_rarity": "Rare",
            "card_modifier": "Base",
            "group_section": {"file1.png": True},
            "cluster_section": {},
            "group_idx": 0
        }

        # Patch button and rerun
        mock_st.button.return_value = True
        mock_st.rerun = lambda: None  # avoid Streamlit rerun

        labels = {}

        # Call the function
        main.display_save_button(cluster_id=1, current_group_id=10, labels=labels)

        # Assert labels were updated
        assert "file1.png" in labels
        assert labels["file1.png"]["name"] == "CardX"
        assert labels["file1.png"]["type"] == "Joker"
        assert labels["file1.png"]["rarity"] == "Rare"
        assert labels["file1.png"]["modifier"] == "Base"
        assert labels["file1.png"]["cluster"] == 1
        assert labels["file1.png"]["card_group"] == 10

        # Assert session_state reset
        assert mock_st.session_state["group_section"] == {}
        assert mock_st.session_state["cluster_section"] == {}


    @patch("annotate_dataset.main.st")
    @patch("annotate_dataset.main.Image.open", return_value=MagicMock())
    def test_group_section_runs(self, mock_image_open, mock_st):
        # Setup session_state
        mock_st.session_state = {
            "group_section": {},
            "cluster_section": {}
        }
        # Mock checkbox to always return True
        mock_st.checkbox.return_value = True

        group_files = ["f1.png", "f2.png"]
        labels = {}

        # run the function
        main.group_section(group_files, labels)

        # After function call, check keys are initialized
        assert set(mock_st.session_state["group_section"].keys()) == set(group_files)
        # Each key should be True because st.checkbox returns True
        assert all(mock_st.session_state["group_section"][f] is True for f in group_files)
        # st.image and st.checkbox should have been called
        assert mock_st.image.called
        assert mock_st.checkbox.called

    @patch("annotate_dataset.main.st")
    @patch("annotate_dataset.main.Image.open", return_value=MagicMock())
    def test_cluster_section_runs(self, mock_image_open, mock_st):
        # Setup session_state
        mock_st.session_state = {
            "group_section": {},
            "cluster_section": {}
        }

        cluster_id = 1
        group_files = ["f1.png"]
        labels = {}

        # Patch cluster_map
        main.cluster_map = {cluster_id: ["f1.png", "f2.png", "f3.png"]}

        # Mock st.button to always return False (simulate no click)
        mock_st.button.return_value = False

        main.cluster_section(cluster_id, group_files, labels)

        # Check cluster_section initialized correctly
        assert set(mock_st.session_state["cluster_section"].keys()) == {"f2.png", "f3.png"}
        # All values should be False by default
        assert all(v is False for v in mock_st.session_state["cluster_section"].values())
        # st.image should have been called for each cluster file
        assert mock_st.image.call_count == 2  