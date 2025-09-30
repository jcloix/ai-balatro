import os
from unittest.mock import patch
import pytest
from torch import nn
import torch

# Modules to test
from train_model import model_saving

@pytest.fixture
def dummy_model():
    """Simple Linear model for testing"""
    return nn.Linear(10, 2)

@pytest.fixture
def dummy_stateful_model():
    """Model with custom state_dict for testing mock comparison"""
    class DummyModel:
        def state_dict(self):
            return {"weight": torch.randn(2, 10), "bias": torch.randn(2)}
    return DummyModel()

# Dummy model for testing
class DummyStatefulModel:
    def state_dict(self):
        return {"weights": [1, 2, 3]}

@patch("train_model.model_saving.torch.save")
@patch("train_model.model_saving.os.makedirs")
def test_handle_checkpoints_saves(mock_makedirs, mock_torch_save, dummy_stateful_model):
    """
    Test that handle_checkpoints correctly saves the best model and periodic checkpoints.
    Patches torch.save and os.makedirs to avoid filesystem writes.
    """
    checkpoint_dir = "/fake/dir"
    best_val_loss = 1.0
    val_loss = 0.5

    # Best model should trigger save
    new_best = model_saving.handle_checkpoints(
        model=dummy_stateful_model,
        val_loss=val_loss,
        best_val_loss=best_val_loss,
        epoch=1,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=5
    )

    assert new_best == val_loss
    mock_makedirs.assert_called_with(checkpoint_dir, exist_ok=True)

    # Validate torch.save call for best model
    assert mock_torch_save.called
    saved_model_dict, saved_path = mock_torch_save.call_args[0]
    assert saved_path == os.path.join(checkpoint_dir, "best_model.pth")
    assert "weight" in saved_model_dict
    assert "bias" in saved_model_dict

    # Periodic checkpoint at interval
    mock_torch_save.reset_mock()
    model_saving.handle_checkpoints(
        model=dummy_stateful_model,
        val_loss=0.6,
        best_val_loss=new_best,
        epoch=5,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=5
    )

    # Should trigger save for periodic checkpoint
    assert mock_torch_save.called
    saved_model_dict, saved_path = mock_torch_save.call_args[0]
    assert saved_path == os.path.join(checkpoint_dir, "checkpoint_epoch5.pth")
    assert "weight" in saved_model_dict
    assert "bias" in saved_model_dict


def test_no_save_when_val_loss_worse(dummy_stateful_model):
    with patch("train_model.model_saving.torch.save") as mock_save, \
         patch("train_model.model_saving.os.makedirs") as mock_mkdir:
        best_val = 0.5
        new_best = model_saving.handle_checkpoints(
            model=dummy_stateful_model,
            val_loss=0.6,  # worse
            best_val_loss=best_val,
            epoch=1,       # not a checkpoint interval
            checkpoint_dir="/fake/dir"
        )
        assert new_best == best_val
        mock_save.assert_not_called()


def test_periodic_checkpoint_even_if_not_best():
    model = DummyStatefulModel()
    best_val_loss = 0.5
    val_loss = 1.0  # worse than best
    checkpoint_dir = "/fake/dir"

    with patch("train_model.model_saving.torch.save") as mock_save, \
         patch("train_model.model_saving.os.makedirs"):
        model_saving.handle_checkpoints(
            model=model,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            epoch=10,  # multiple of checkpoint_interval
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=5
        )

        # Confirm a checkpoint save was called
        saved_model_dict, saved_path = mock_save.call_args[0]
        assert "checkpoint_epoch10.pth" in saved_path


def test_custom_checkpoint_interval():
    model = DummyStatefulModel()
    best_val_loss = 0.5
    val_loss = 0.4
    checkpoint_dir = "/fake/dir"

    with patch("train_model.model_saving.torch.save") as mock_save, \
         patch("train_model.model_saving.os.makedirs"):
        model_saving.handle_checkpoints(
            model=model,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            epoch=7,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=7  # custom interval
        )

        # Confirm the checkpoint was saved due to custom interval
        saved_model_dict, saved_path = mock_save.call_args[0]
        assert "checkpoint_epoch7.pth" in saved_path


def test_epoch_zero_behavior():
    model = DummyStatefulModel()
    best_val_loss = 1.0
    val_loss = 0.5
    checkpoint_dir = "/fake/dir"

    with patch("train_model.model_saving.torch.save") as mock_save, \
         patch("train_model.model_saving.os.makedirs"):
        new_best = model_saving.handle_checkpoints(
            model=model,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            epoch=0,  # epoch zero
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=5
        )

        # Multiple saves may happen: best_model and checkpoint_epoch0
        save_calls = [call[0][1] for call in mock_save.call_args_list]
        assert any("best_model.pth" in path for path in save_calls)
        assert any("checkpoint_epoch0.pth" in path for path in save_calls)
        assert new_best == val_loss