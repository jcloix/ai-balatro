import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from train_model.metrics import metrics


# -----------------------------
# Dummy helper classes
# -----------------------------
class DummyLoader:
    def __init__(self, classes, data):
        self.classes = classes
        self.data = data

    def __iter__(self):
        return iter(self.data)


class DummyHead:
    def __init__(self, name="test_task", num_classes=2):
        self.name = name
        self.num_classes = num_classes
        self.train_loader = DummyLoader(classes=["A", "B"], data=[])


class DummyState:
    def __init__(self):
        self.optimizer = MagicMock()
        self.optimizer.param_groups = [{"lr": 0.01}]
        self.device = torch.device("cpu")
        self.model = MagicMock()


class DummyEpochResult:
    def __init__(self):
        self.epoch = 1
        self.train_loss = 0.1
        self.val_loss = 0.2
        self.val_outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        self.val_labels = torch.tensor([1, 0])


# -----------------------------
# Tests for Metrics container
# -----------------------------
def test_metrics_container_basic(monkeypatch):
    head, state, epoch_res = DummyHead(), DummyState(), DummyEpochResult()
    dummy_metric = MagicMock(spec=metrics.Metric)
    dummy_metric.name = "dummy"
    dummy_metric.value = 42

    m = metrics.Metrics(metrics=[dummy_metric])
    m.compute_all(head, state, epoch_res)
    m.log_all(head, state, epoch_res, writer=None)

    dummy_metric.compute.assert_called_once()
    dummy_metric.log.assert_called_once()
    dummy_metric.log_tensorboard.assert_called_once()


# -----------------------------
# Tests for EpochSummary
# -----------------------------
def test_epoch_summary_compute_and_log(capsys):
    head, state, epoch_res = DummyHead(), DummyState(), DummyEpochResult()

    metric = metrics.EpochSummary()
    metric.compute(head, state, epoch_res)
    metric.log(epoch=epoch_res.epoch)
    out = capsys.readouterr().out

    assert "Epoch 1" in out
    assert "Train Loss" in out
    assert pytest.approx(metric.train_loss, 0.0001) == 0.1
    assert pytest.approx(metric.val_loss, 0.0001) == 0.2


@patch("torch.utils.tensorboard.SummaryWriter")
def test_epoch_summary_tensorboard(mock_writer_class):
    writer = mock_writer_class.return_value
    metric = metrics.EpochSummary()
    metric.train_loss, metric.val_loss = 0.1, 0.2
    metric.log_tensorboard(writer, epoch=1)

    writer.add_scalar.assert_any_call("Loss/Train", 0.1, 1)
    writer.add_scalar.assert_any_call("Loss/Val", 0.2, 1)


# -----------------------------
# Tests for TopKAccuracy
# -----------------------------
def test_topk_accuracy_compute_and_log(capsys):
    epoch_res = DummyEpochResult()
    head, state = DummyHead(), DummyState()

    metric = metrics.TopKAccuracy(k=1)
    val = metric.compute(head, state, epoch_res)
    assert 0 <= val <= 1
    metric.log(epoch=1)
    assert "Top-1 Accuracy" in capsys.readouterr().out


@patch("torch.utils.tensorboard.SummaryWriter")
def test_topk_accuracy_tensorboard(mock_writer_class):
    writer = mock_writer_class.return_value
    metric = metrics.TopKAccuracy(k=3)
    metric.value = 0.75
    metric.log_tensorboard(writer, epoch=1)
    writer.add_scalar.assert_called_with("Accuracy/Top3", 0.75, 1)


# -----------------------------
# Tests for ConfusionMatrix
# -----------------------------
def test_confusion_matrix_compute(monkeypatch):
    # Mock unwrap_model to return predictable outputs
    monkeypatch.setattr(metrics, "unwrap_model", lambda model, x, task: torch.tensor([[1.0, 0.0]]))
    head = DummyHead()
    head.num_classes = 2
    state = DummyState()
    data = [(torch.randn(1, 3, 3), torch.tensor([0]))]
    head.train_loader.data = data

    epoch_res = DummyEpochResult()
    cm_metric = metrics.ConfusionMatrix()
    value = cm_metric.compute(head, state, epoch_res)
    assert isinstance(value, np.ndarray)
    assert value.shape == (2, 2)


@patch("torch.utils.tensorboard.SummaryWriter")
def test_confusion_matrix_tensorboard(mock_writer_class):
    writer = mock_writer_class.return_value
    cm_metric = metrics.ConfusionMatrix()
    cm_metric.value = np.array([[2, 1], [0, 3]])
    cm_metric.log_tensorboard(writer, epoch=1, class_names=["A", "B"])
    writer.add_figure.assert_called_once()


# -----------------------------
# Tests for ConfusionSummary
# -----------------------------
def test_confusion_summary_compute_and_log(capsys):
    head, state, epoch_res = DummyHead(), DummyState(), DummyEpochResult()

    metric = metrics.ConfusionSummary()
    val = metric.compute(head, state, epoch_res)

    assert all(k in val for k in ["precision", "recall", "f1"])
    metric.log(epoch=1)
    out = capsys.readouterr().out
    assert "Precision" in out and "Recall" in out and "F1" in out


@patch("torch.utils.tensorboard.SummaryWriter")
def test_confusion_summary_tensorboard(mock_writer_class):
    writer = mock_writer_class.return_value
    metric = metrics.ConfusionSummary()
    metric.value = {"precision": 0.8, "recall": 0.9, "f1": 0.85}
    metric.log_tensorboard(writer, epoch=1)

    writer.add_scalar.assert_any_call("Precision/macro", 0.8, 1)
    writer.add_scalar.assert_any_call("Recall/macro", 0.9, 1)
    writer.add_scalar.assert_any_call("F1/macro", 0.85, 1)
