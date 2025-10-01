from torchvision.models import ResNet
from models.models import build_model

def test_build_model_returns_resnet18():
    model, device = build_model(num_classes=10)
    assert isinstance(model, ResNet)
    assert model.fc.out_features == 10
    assert next(model.parameters()).device == device