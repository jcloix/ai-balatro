import pytest
from unittest.mock import patch
import torch
from train_model import train_setup
from train_model.train_state import TrainingState


@pytest.fixture(autouse=True)
def mock_summary_writer():
    # Patch SummaryWriter where it's imported in train_setup
    with patch("train_model.train_setup.SummaryWriter") as mock_writer_class:
        yield mock_writer_class.return_value


def test_prepare_training_returns_state():
    state = train_setup.prepare_training(num_classes=5, log_dir="runs/test")
    assert isinstance(state, TrainingState)
    assert hasattr(state, "model")
    assert hasattr(state, "optimizer")
    assert hasattr(state, "scheduler")
    assert hasattr(state, "criterion")
    assert hasattr(state, "scaler")
    assert hasattr(state, "early_stopping")
    assert hasattr(state, "writer")


def test_freeze_backbone_parameters():
    state = train_setup.prepare_training(num_classes=3, log_dir="runs/test", freeze_backbone=True)
    for name, param in state.model.named_parameters():
        if "fc" in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad


def test_scheduler_selection_step_lr_and_plateau():
    from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

    # Small dataset -> StepLR
    state_small = train_setup.prepare_training(num_classes=3, log_dir="runs/test", dataset_size=10)
    assert isinstance(state_small.scheduler, StepLR)

    # Medium dataset -> ReduceLROnPlateau
    state_medium = train_setup.prepare_training(num_classes=3, log_dir="runs/test", dataset_size=150)
    assert isinstance(state_medium.scheduler, ReduceLROnPlateau)


@patch("train_model.train_setup.torch.load")
@patch("os.path.exists", return_value=True)
def test_resume_checkpoint_calls_load_state_dict(mock_exists, mock_torch_load):
    dummy_ckpt = {"fc.weight": torch.randn(3, 512), "fc.bias": torch.randn(3)}
    mock_torch_load.return_value = dummy_ckpt

    with patch.object(train_setup, "build_model") as mock_build:
        model_mock = torch.nn.Linear(512, 3)
        with patch.object(model_mock, "load_state_dict") as mock_load_state:
            mock_build.return_value = (model_mock, torch.device("cpu"))
            state = train_setup.prepare_training(
                num_classes=3,
                log_dir="runs/test",
                resume_path="fake_path.pth"
            )

            mock_exists.assert_called_with("fake_path.pth")
            mock_torch_load.assert_called_with("fake_path.pth", map_location=state.device)
            mock_load_state.assert_called_with(dummy_ckpt)


def test_grad_scaler_initialization_mock():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.amp.GradScaler") as mock_scaler, \
         patch("train_model.train_setup.build_model") as mock_build:

        dummy_model = torch.nn.Linear(10, 3)
        dummy_device = torch.device("cuda")
        mock_build.return_value = (dummy_model, dummy_device)

        state = train_setup.prepare_training(num_classes=3, log_dir="runs/test")

        mock_scaler.assert_called_once_with(device="cuda")
        assert state.scaler == mock_scaler.return_value
        assert state.model == dummy_model
        assert state.device == dummy_device


def test_grad_scaler_none_on_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    state = train_setup.prepare_training(num_classes=3, log_dir="runs/test")
    assert state.scaler is None
