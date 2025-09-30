import os
import pytest
from unittest.mock import patch, MagicMock
from torch import nn, optim
import torch
from torchvision.models import ResNet, resnet18

# Modules to test
from train_model import models
from train_model.train_state import TrainingState

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture(autouse=True)
def mock_summary_writer():
    # Patch the SummaryWriter **where it's imported in models.py**
    with patch("train_model.models.SummaryWriter") as mock_writer_class:
        yield mock_writer_class.return_value

@pytest.fixture
def dummy_model():
    return nn.Linear(10, 2)

# -----------------------------
# Existing tests
# -----------------------------
def test_build_model_returns_resnet18():
    model, device = models.build_model(num_classes=10)
    assert isinstance(model, ResNet)
    assert model.fc.out_features == 10
    assert next(model.parameters()).device == device

def test_prepare_training_returns_state():
    state = models.prepare_training(num_classes=5, log_dir="runs/test")
    assert isinstance(state, TrainingState)
    assert hasattr(state, "model")
    assert hasattr(state, "optimizer")
    assert hasattr(state, "scheduler")
    assert hasattr(state, "criterion")
    assert hasattr(state, "scaler")
    assert hasattr(state, "early_stopping")
    assert hasattr(state, "writer")

# -----------------------------
# New tests
# -----------------------------
def test_freeze_backbone_parameters():
    state = models.prepare_training(num_classes=3, log_dir="runs/test", freeze_backbone=True)
    for name, param in state.model.named_parameters():
        if "fc" in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad

def test_scheduler_selection_step_lr_and_plateau():
    # Small dataset -> StepLR
    state_small = models.prepare_training(num_classes=3, log_dir="runs/test", dataset_size=10)
    from torch.optim.lr_scheduler import StepLR
    assert isinstance(state_small.scheduler, StepLR)

    # Medium dataset -> ReduceLROnPlateau
    state_medium = models.prepare_training(num_classes=3, log_dir="runs/test", dataset_size=150)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    assert isinstance(state_medium.scheduler, ReduceLROnPlateau)

@patch("train_model.models.torch.load")
@patch("os.path.exists", return_value=True)
def test_resume_checkpoint_calls_load_state_dict(mock_exists, mock_torch_load):
    # Mock checkpoint dict
    dummy_ckpt = {"fc.weight": torch.randn(3, 512), "fc.bias": torch.randn(3)}
    mock_torch_load.return_value = dummy_ckpt

    # Patch build_model to return a mock model
    with patch.object(models, "build_model") as mock_build:
        model_mock = nn.Linear(512, 3)
        # Patch load_state_dict on the model to track the call
        with patch.object(model_mock, "load_state_dict") as mock_load_state:
            mock_build.return_value = (model_mock, torch.device("cpu"))
            
            state = models.prepare_training(
                num_classes=3,
                log_dir="runs/test",
                resume_path="fake_path.pth"
            )
            
            # Check that os.path.exists was called
            mock_exists.assert_called_with("fake_path.pth")
            # Check that torch.load was called
            mock_torch_load.assert_called_with("fake_path.pth", map_location=state.device)
            # Check that load_state_dict was called with the checkpoint
            mock_load_state.assert_called_with(dummy_ckpt)


def test_grad_scaler_initialization_mock():
    # Mock torch.cuda.is_available to return True
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.amp.GradScaler") as mock_scaler, \
         patch("train_model.models.build_model") as mock_build:

        # Provide a dummy model and device
        dummy_model = torch.nn.Linear(10, 3)
        dummy_device = torch.device("cuda")
        mock_build.return_value = (dummy_model, dummy_device)

        # Call prepare_training
        state = models.prepare_training(num_classes=3, log_dir="runs/test")

        # Check that GradScaler was instantiated with device="cuda"
        mock_scaler.assert_called_once_with(device="cuda")

        # Check that the scaler in state is the mocked one
        assert state.scaler == mock_scaler.return_value

        # Also sanity check the model and device
        assert state.model == dummy_model
        assert state.device == dummy_device

def test_grad_scaler_none_on_cpu(monkeypatch):
    # Force CUDA unavailable
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    state = models.prepare_training(num_classes=3, log_dir="runs/test")
    assert state.scaler is None
