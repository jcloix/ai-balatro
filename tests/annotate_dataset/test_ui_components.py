# tests/annotate_dataset/test_ui_components.py
import unittest
from unittest.mock import patch, MagicMock
from annotate_dataset import ui_components

class TestUIComponents(unittest.TestCase):

    # ---------------------------
    # Test display_card
    # ---------------------------
    @patch("annotate_dataset.ui_components.st.image")
    @patch("annotate_dataset.ui_components.Image.open")
    def test_display_card(self, mock_open, mock_st_image):
        mock_img = MagicMock()
        mock_open.return_value = mock_img

        ui_components.display_card("dummy_path.png", width=123)
        mock_open.assert_called_with("dummy_path.png")
        mock_st_image.assert_called_with(mock_img, width=123)

    # ---------------------------
    # Test render_sidebar
    # ---------------------------
    @patch("annotate_dataset.ui_components.st.sidebar")
    def test_render_sidebar(self, mock_sidebar):
        unique_by_type = {
            "Joker": [1, 2],
            "Planet": [3],
            "Tarot": [],
            "Spectral": [4, 5, 6]
        }
        unique_by_modifier = {
            "Base": [1, 2],
            "Foil": [3],
            "Holographic": [],
            "Polychrome": [4],
            "Negative": [5, 6]
        }

        ui_components.render_sidebar(
            total=10,
            labeled_count=4,
            unlabeled_count=6,
            unique_by_type=unique_by_type,
            unique_by_modifier=unique_by_modifier
        )

        # Check sidebar methods called
        mock_sidebar.title.assert_called_with("📊 Labeling Progress")
        mock_sidebar.write.assert_any_call("✅ Labeled: 4")
        mock_sidebar.write.assert_any_call("🕓 Unlabeled: 6")
        mock_sidebar.write.assert_any_call("📦 Total: 10")

        mock_sidebar.markdown.assert_any_call("### 🔎 Unique Cards by Type")
        mock_sidebar.write.assert_any_call("🃏 Joker: 2")
        mock_sidebar.write.assert_any_call("🪐 Planet: 1")
        mock_sidebar.write.assert_any_call("🔮 Tarot: 0")
        mock_sidebar.write.assert_any_call("👻 Spectral: 3")

        mock_sidebar.markdown.assert_any_call("### ✨ Unique Cards by Modifier")
        mock_sidebar.write.assert_any_call("🎴 Base: 2")
        mock_sidebar.write.assert_any_call("🌈 Foil: 1")
        mock_sidebar.write.assert_any_call("💎 Holographic: 0")
        mock_sidebar.write.assert_any_call("🎨 Polychrome: 1")
        mock_sidebar.write.assert_any_call("🕳️ Negative: 2")


    # ---------------------------
    # Test helper_selectboxes
    # ---------------------------
    @patch("annotate_dataset.ui_components.st.selectbox")
    @patch("annotate_dataset.ui_components.st.columns")
    def test_helper_selectboxes(self, mock_columns, mock_selectbox):
        # Mock 4 columns
        col_mocks = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_columns.return_value = col_mocks

        existing_by_type = {
            "Joker": ["J1", "J2"],
            "Planet": ["P1"],
            "Tarot": ["T1", "T2", "T3"],
            "Spectral": []
        }
        # Simulate selectbox returning the first option (empty string)
        mock_selectbox.side_effect = lambda label, opts, key: opts[0]

        selected = ui_components.helper_selectboxes(existing_by_type)
        self.assertEqual(selected, {"joker": "", "planet": "", "tarot": "", "spectral": ""})

        mock_columns.assert_called_with(4)
        self.assertEqual(mock_selectbox.call_count, 4)

if __name__ == "__main__":
    unittest.main()
