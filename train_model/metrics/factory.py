# metrics_factory.py
from train_model.metrics.metrics import TopKAccuracy, ConfusionMatrix, EpochSummary, ConfusionSummary

def create_metrics(metric_names):
    metrics = [EpochSummary()]  # mandatory, prints train + val loss
    for name in metric_names:
        if name == "loss":
            metrics.append(LossMetric())
        elif name == "top1_acc":
            metrics.append(TopKAccuracy(k=1))
        elif name == "top3_acc":
            metrics.append(TopKAccuracy(k=3))
        elif name == "cm":
            metrics.append(ConfusionMatrix())
        elif name == "confusion_summary":
            metrics.append(ConfusionSummary())
        else:
            raise ValueError(f"Unknown metric: {name}")
    return metrics
