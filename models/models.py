# models/models.py
import torch
from torch import nn
from torchvision import models

class MultiHeadModel(nn.Module):
    def __init__(self, head_configs=None, pretrained=True):
        """
        backbone_name: str, e.g., "resnet18"
        head_configs: dict, e.g., {"identification": 52, "modifier": 20}
        """
        super().__init__()
        backbone = models.resnet18(pretrained=pretrained)

        # Remove final fc
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # outputs [B, 512, 1, 1]
        self.backbone_out_features = backbone.fc.in_features

        # Create heads
        self.heads = nn.ModuleDict()
        for name, num_classes in head_configs.items():
            self.heads[name] = nn.Linear(self.backbone_out_features, num_classes)

    def forward(self, x):
        # Flatten backbone output
        features = self.backbone(x)            # shape [B, 512, 1, 1]
        features = torch.flatten(features, 1)  # shape [B, 512]
        out = {name: head(features) for name, head in self.heads.items()}
        return out


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
