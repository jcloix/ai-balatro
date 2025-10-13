import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch
import torch
from train_model import persistence

# --------------------------
# Fixtures
# --------------------------
class DummyTrainingState:
    def __init__(self):
        self.model = MagicMock()
        self.optimizer = MagicMock()
        self.scheduler = MagicMock()
        self.scaler = MagicMock()

class DummyHead:
    def __init__(self, name, class_names):
        self.name = name
        self.class_names = class_names

# --------------------------
# Tests
# --------------------------
def test_save_checkpoint_creates_file(tmp_path):
    training_state = DummyTrainingState()
    head = DummyHead("id_head", ["A", "B"])
    path = tmp_path / "checkpoint.pth"
    
    # Mock state_dict returns
    training_state.model.state_dict.return_value = {"weights": 1}
    training_state.optimizer.state_dict.return_value = {"lr": 0.01}
    training_state.scheduler.state_dict.return_value = {"step": 1}
    training_state.scaler.state_dict.return_value = {"scale": 1.0}

    persistence.save_checkpoint(training_state, [head], str(path), epoch=3)
    
    assert os.path.exists(path)
    checkpoint = torch.load(path)
    assert checkpoint["epoch"] == 3
    assert "head_states" in checkpoint
    assert checkpoint["head_states"]["id_head"]["class_names"] == ["A", "B"]

def test_handle_checkpoints_saves_best_and_interval(tmp_path):
    training_state = DummyTrainingState()
    training_state.best_val_loss = 0.5   # attach to object
    head = DummyHead("h", ["X"])

    # Patch save_checkpoint to record calls
    with patch("train_model.persistence.save_checkpoint") as mock_save:
        best_val = persistence.handle_checkpoints(
            training_state,
            [head],
            val_loss=0.4,  # lower than previous best -> should trigger save
            epoch=5,
            checkpoint_dir=str(tmp_path),
            checkpoint_interval=5
        )
        
        # Should save best model and periodic checkpoint
        assert mock_save.call_count == 2
        assert best_val == 0.4

def test_load_checkpoint_file_not_found(tmp_path):
    path = tmp_path / "missing.pth"
    with pytest.raises(FileNotFoundError):
        persistence.load_checkpoint(str(path))

def test_apply_checkpoint_loads_state_dicts():
    checkpoint = {
        "model": {"weights": 1},
        "optimizer": {
            "state": {},
            "param_groups": [{"lr": 0.01}]
        },
        "scheduler": {"step": 1},
        "scaler": {"scale": 1.0},
        "epoch": 2
    }

    # Mocks
    model = MagicMock()
    model.load_state_dict.return_value = ([], [])

    optimizer = MagicMock()
    optimizer.state_dict.return_value = checkpoint["optimizer"]  # full dict
    scheduler = MagicMock()
    scaler = MagicMock()
    scaler.load_state_dict = MagicMock()

    start_epoch = persistence.apply_checkpoint(checkpoint, model, optimizer, scheduler, scaler)

    model.load_state_dict.assert_called_once_with({"weights": 1}, strict=False)
    optimizer.load_state_dict.assert_called_once()
    scaler.load_state_dict.assert_called_once()
    assert start_epoch == 3
