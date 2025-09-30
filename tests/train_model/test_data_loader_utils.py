from unittest.mock import MagicMock, patch

# Modules to test
from train_model import data_loader_utils

@patch("train_model.data_loader_utils.load_merged_labels")
@patch("train_model.data_loader_utils.get_train_val_loaders")
def test_load_dataloaders(mock_get_loaders, mock_load_labels):
    mock_load_labels.return_value = {"img1.png": {"name": "Baron"}, "img2.png": {"name": "Mime"}}
    mock_get_loaders.return_value = ("train_loader", "val_loader")

    train_loader, val_loader = data_loader_utils.load_dataloaders(batch_size=2, val_split=0.5)
    assert train_loader == "train_loader"
    assert val_loader == "val_loader"


@patch("train_model.data_loader_utils.CardDataset.from_labels_dict")
@patch("train_model.data_loader_utils.load_merged_labels")
@patch("train_model.data_loader_utils.get_train_val_loaders")
def test_load_dataloaders_full(mock_get_loaders, mock_load_labels, mock_dataset):
    dummy_dataset = MagicMock()
    mock_dataset.return_value = dummy_dataset
    dummy_train_loader = dummy_val_loader = "loader"
    mock_get_loaders.return_value = (dummy_train_loader, dummy_val_loader)
    mock_load_labels.return_value = {"img1.png":{"name":"Joker"}}

    train_loader, val_loader = data_loader_utils.load_dataloaders(batch_size=2, val_split=0.1, no_augmented=False)
    assert train_loader == dummy_train_loader
    assert val_loader == dummy_val_loader
    mock_get_loaders.assert_called_once_with(
        dummy_dataset,
        batch_size=2,
        val_split=0.1,
        train_transform=data_loader_utils.Config.TRANSFORMS['train'],
        val_transform=data_loader_utils.Config.TRANSFORMS['test'],
        shuffle=True,
        use_weighted_sampler=False
    )
