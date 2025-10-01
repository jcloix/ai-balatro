# models/models.py
import torch
from torch import nn
from torchvision import models

def build_model(num_classes):
    """
    Build a pretrained ResNet18 and replace the final layer
    with a linear layer of num_classes outputs.
    """
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), device

def load_checkpoint(model, checkpoint_path, device=None):
    """
    Load a saved state_dict into the model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
