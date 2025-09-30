from unittest.mock import patch
import torch
import numpy as np

from train_model import logging
import matplotlib
matplotlib.use("Agg")  # non-GUI backend

# Dummy metrics class
class DummyMetrics:
    def __init__(self, train_loss=0.1, val_loss=0.2, topk_acc=0.3, cm=None):
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.topk_acc = topk_acc
        self.cm = cm


def test_log_epoch_stats_prints(capsys):
    class DummyOptimizer:
        param_groups = [{"lr": 0.01}]

    metrics = DummyMetrics()
    logging.log_epoch_stats(1, DummyOptimizer(), metrics)
    captured = capsys.readouterr()
    assert "Epoch 1" in captured.out
    assert "Train Loss: 0.1000" in captured.out
    assert "Val Loss: 0.2000" in captured.out
    assert "Top-3 Accuracy: 30.00%" in captured.out


@patch("torch.utils.tensorboard.SummaryWriter")
def test_log_epoch_stats_tensorboard(mock_writer_class):
    """
    Test that log_epoch_stats logs to a TensorBoard SummaryWriter correctly.
    Patches the real SummaryWriter to avoid creating files.
    """
    mock_writer = mock_writer_class.return_value

    class DummyOptimizer:
        param_groups = [{"lr": 0.01}]

    # Simple 2x2 confusion matrix
    cm = torch.tensor([[2, 1], [0, 3]])
    class_names = ["A", "B"]

    metrics = DummyMetrics(train_loss=0.1, val_loss=0.2, topk_acc=0.3, cm=cm)

    logging.log_epoch_stats(epoch=1, optimizer=DummyOptimizer(), metrics=metrics,
                            writer=mock_writer, class_names=class_names)

    # Check that scalar metrics were logged
    mock_writer.add_scalar.assert_any_call("Loss/Train", 0.1, 1)
    mock_writer.add_scalar.assert_any_call("Loss/Val", 0.2, 1)
    mock_writer.add_scalar.assert_any_call("Learning_Rate", 0.01, 1)
    mock_writer.add_scalar.assert_any_call("Accuracy/Top3", 0.3, 1)

    # Check that the confusion matrix figure was logged
    mock_writer.add_figure.assert_called()

def test_log_epoch_stats_with_numpy_cm():
    """
    Test log_epoch_stats using a NumPy array as confusion matrix.
    Ensures both scalar logging and confusion matrix plotting work.
    """
    class DummyOptimizer:
        param_groups = [{"lr": 0.01}]

    # NumPy confusion matrix
    cm = np.array([[5, 2], [1, 7]])
    class_names = ["Class0", "Class1"]

    class DummyMetrics:
        def __init__(self):
            self.train_loss = 0.15
            self.val_loss = 0.25
            self.topk_acc = 0.4
            self.cm = cm

    metrics = DummyMetrics()

    with patch("torch.utils.tensorboard.SummaryWriter") as mock_writer_class:
        mock_writer = mock_writer_class.return_value

        # Call log_epoch_stats
        logging.log_epoch_stats(
            epoch=2,
            optimizer=DummyOptimizer(),
            metrics=metrics,
            writer=mock_writer,
            class_names=class_names
        )

        # Scalars should be logged
        mock_writer.add_scalar.assert_any_call("Loss/Train", 0.15, 2)
        mock_writer.add_scalar.assert_any_call("Loss/Val", 0.25, 2)
        mock_writer.add_scalar.assert_any_call("Learning_Rate", 0.01, 2)
        mock_writer.add_scalar.assert_any_call("Accuracy/Top3", 0.4, 2)

        # Confusion matrix figure should be logged
        mock_writer.add_figure.assert_called()