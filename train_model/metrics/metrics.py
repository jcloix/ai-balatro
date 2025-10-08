# metrics.py
from abc import ABC, abstractmethod
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from sklearn.metrics import confusion_matrix
from models.models import unwrap_model
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

@dataclass
class Metrics:
    metrics: list

    def compute_all(self, head, state, epoch_res):
        for metric in self.metrics:
            metric.compute(head, state, epoch_res)

    def log_all(self, head, state, epoch_res, writer=None):
        for metric in self.metrics:
            metric.log(epoch=epoch_res.epoch, class_names=head.train_loader.classes)
            metric.log_tensorboard(writer=writer, epoch=epoch_res.epoch, class_names=head.train_loader.classes)

    def get(self, name):
        for metric in self.metrics:
            if metric.name == name:
                return metric.value
        return None

# --------------------------------
# Base Class
# --------------------------------
class Metric(ABC):
    def __init__(self, name):
        self.name = name
        self.value = None

    @abstractmethod
    def compute(self, head, state, epoch_res):
        """Compute metric value using the provided context."""
        pass

    def log(self, epoch=None, class_names=None):
        """Optional: print and/or TensorBoard log"""
        print(f"{self.name}: {self.value}")

    def log_tensorboard(self, writer, epoch=None, class_names=None):
        writer.add_scalar(self.name, self.value, epoch)

    def __str__(self):
        return f"{self.name}: {self.value}"


# --------------------------------
# Implementations
# --------------------------------
class EpochSummary(Metric):
    def __init__(self):
        super().__init__("epoch_summary")

    def compute(self, head, state, epoch_res):
        self.train_loss = epoch_res.train_loss
        self.val_loss = epoch_res.val_loss
        self.lr = state.optimizer.param_groups[0]['lr']
        return self

    def log(self, epoch=None, class_names=None):
        lr_str = f"{self.lr:.6f}" if self.lr else "N/A"
        print(f"Epoch {epoch} | Train Loss: {self.train_loss:.4f} | Val Loss: {self.val_loss:.4f} | LR: {lr_str}")

    

    def log_tensorboard(self, writer, epoch=None, class_names=None):
        writer.add_scalar("Loss/Train", self.train_loss, epoch)
        writer.add_scalar("Loss/Val", self.val_loss, epoch)



class TopKAccuracy(Metric):
    def __init__(self, k=1):
        super().__init__(f"top{k}_acc")
        self.k = k

    def compute(self, head, state, epoch_res):
        outputs, labels = epoch_res.val_outputs, epoch_res.val_labels
        _, topk_preds = outputs.topk(self.k, dim=1)
        correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
        self.value = correct.sum().item() / labels.size(0)
        return self.value

    def log(self, epoch=None, class_names=None):
        print(f"Top-{self.k} Accuracy: {self.value * 100:.2f}%")

    def log_tensorboard(self, writer, epoch=None, class_names=None):
        writer.add_scalar(f"Accuracy/Top{self.k}", self.value, epoch)


class ConfusionMatrix(Metric):
    def __init__(self):
        super().__init__("cm")

    def compute(self, head, state, epoch_res):
        model, loader, device, task_name = (
            state.model, head.train_loader, state.device, head.name
        )
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                preds = torch.argmax(unwrap_model(model, images, task_name), dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        self.value = confusion_matrix(all_labels, all_preds, labels=np.arange(head.num_classes))
        return self.value

    def log(self, writer=None, epoch=None, class_names=None):
        print("Confusion Matrix:")
        df_cm = pd.DataFrame(self.value, index=class_names, columns=class_names)
        print(df_cm)

    def log_tensorboard(self, writer, epoch=None, class_names=None):
        fig = self._plot_confusion_matrix(class_names)
        writer.add_figure("Confusion_Matrix", fig, epoch)
        plt.close(fig)

    def _plot_confusion_matrix(self, class_names):
        cm = self.value
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, f"{cm[i, j]}\n{cm_normalized[i, j]*100:.1f}%",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=8)
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        return fig


class ConfusionSummary(Metric):
    """
    Lightweight confusion summary for large-class problems.
    Reports precision, recall, and F1-score (macro average).
    """
    def __init__(self):
        super().__init__("confusion_summary")

    def compute(self, head, state, epoch_res):
        outputs, labels = epoch_res.val_outputs, epoch_res.val_labels
        preds = torch.argmax(outputs, dim=1)

        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Compute summary metrics
        self.value = {
            "precision": precision_score(labels_np, preds_np, average="macro", zero_division=0),
            "recall": recall_score(labels_np, preds_np, average="macro", zero_division=0),
            "f1": f1_score(labels_np, preds_np, average="macro", zero_division=0),
        }
        return self.value

    def log(self, epoch=None, class_names=None):
        p, r, f1 = self.value["precision"], self.value["recall"], self.value["f1"]
        print(f"Precision: {p:.3f} | Recall: {r:.3f} | F1: {f1:.3f}")

    def log_tensorboard(self, writer, epoch=None, class_names=None):
        writer.add_scalar("Precision/macro", self.value["precision"], epoch)
        writer.add_scalar("Recall/macro", self.value["recall"], epoch)
        writer.add_scalar("F1/macro", self.value["f1"], epoch)