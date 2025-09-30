from config.config import MODELS_DIR
import matplotlib.pyplot as plt
import torch
import itertools
import pandas as pd
import numpy as np


def log_confusion_matrix(confusion_matrix=None, class_names=None):
    if confusion_matrix is not None:
        print("Confusion Matrix:")
        if class_names:
            # Convert to DataFrame with labels for nicer printing
            df_cm = pd.DataFrame(confusion_matrix,
                                 index=class_names,
                                 columns=class_names)
            print(df_cm)
        else:
            # Fallback: raw tensor/array
            print(confusion_matrix)

def plot_confusion_matrix(cm, class_names=None):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Accepts either a NumPy array or a PyTorch tensor.
    """

    if isinstance(cm, torch.Tensor):
        cm = cm.cpu().numpy()  # convert tensor to NumPy array

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Normalize
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    # Text annotations
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]}\n{cm_normalized[i, j]*100:.1f}%",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=8)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure

def log_epoch_stats(epoch, optimizer, metrics, writer=None, class_names=None):
    lr = optimizer.param_groups[0]['lr']

    # Print main stats
    print(f"Epoch {epoch} | Train Loss: {metrics.train_loss:.4f} | Val Loss: {metrics.val_loss:.4f} | LR: {lr:.6f}")

    # Optionally print metrics
    if metrics.topk_acc is not None:
        print(f"Top-3 Accuracy: {metrics.topk_acc *100:.2f}%")
    log_confusion_matrix(metrics.cm,class_names)

    # Log to TensorBoard if writer is provided
    if writer:
        writer.add_scalar("Loss/Train", metrics.train_loss, epoch)
        writer.add_scalar("Loss/Val", metrics.val_loss, epoch)
        writer.add_scalar("Learning_Rate", lr, epoch)
        if metrics.topk_acc is not None:
            writer.add_scalar("Accuracy/Top3", metrics.topk_acc, epoch)
        
        # 🔥 Confusion matrix heatmap
        if metrics.cm is not None:
            fig = plot_confusion_matrix(metrics.cm, class_names=class_names)
            writer.add_figure("Confusion_Matrix", fig, epoch)
            plt.close(fig)  # cleanup

