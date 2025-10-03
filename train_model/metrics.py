#metrics.py
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from dataclasses import dataclass
from models.models import unwrap_model

@dataclass
class Metrics:
    train_loss: float = None
    val_loss: float = None
    topk_acc: float = None
    cm: object = None

    @classmethod
    def from_validation(cls, val_loss, topk_acc=None, cm=None):
        """Factory for validation-only metrics"""
        return cls(val_loss=val_loss, topk_acc=topk_acc, cm=cm)

    @classmethod
    def from_epoch(cls, train_loss, val_metrics: "Metrics"):
        """Factory combining train + validation results"""
        return cls(
            train_loss=train_loss,
            val_loss=val_metrics.val_loss,
            topk_acc=val_metrics.topk_acc,
            cm=val_metrics.cm,
        )

def top_k_accuracy(outputs, labels, k=3):
    """
    Top-K accuracy: fraction of samples where true label is in top-k predictions.
    The Top-k accuracy provide information of if the answer was in the top-k guesses. 
    """
    _, topk_preds = outputs.topk(k, dim=1) # indices of top-k predictions
    correct = topk_preds.eq(labels.view(-1,1).expand_as(topk_preds)) # compare with true labels
    return correct.sum().item() / labels.size(0)

def compute_confusion_matrix(model, loader, device, num_classes, task_name):
    """
    Compute the confusion matrix for a dataset.
    
    Instead of just average, give information of class-by-class success of answers.
    
    Returns a numpy array of shape (num_classes, num_classes).
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = torch.argmax(unwrap_model(model,images,task_name), dim=1)  # predicted labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
