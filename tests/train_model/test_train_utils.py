import torch
from train_model.train_utils import TrainingState, EarlyStopping, build_model, prepare_training

def test_training_state_attributes():
    model = torch.nn.Linear(2, 2)
    state = TrainingState(model, "cpu", None, None, None, None, None, None)
    assert state.model is model
    assert state.device == "cpu"
    assert state.best_val_loss == float('inf')

def test_early_stopping_triggers():
    es = EarlyStopping(patience=2)
    es.step(1.0)
    assert not es.early_stop
    es.step(1.1)
    assert not es.early_stop
    es.step(1.2)
    assert es.early_stop

def test_build_model_returns_resnet18():
    model, device = build_model(num_classes=10)
    from torchvision.models import ResNet
    assert isinstance(model, ResNet)
    assert model.fc.out_features == 10
    assert next(model.parameters()).device == device

def test_prepare_training_returns_state():
    state = prepare_training(num_classes=5, log_dir="runs/test")
    assert hasattr(state, "model")
    assert hasattr(state, "optimizer")
    assert hasattr(state, "scheduler")
    assert hasattr(state, "criterion")
    assert hasattr(state, "scaler")
    assert hasattr(state, "early_stopping")
    assert hasattr(state, "writer")
