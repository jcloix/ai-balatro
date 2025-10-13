import torch
from torchvision import transforms
from PIL import Image
from models.models import MultiHeadModel
import argparse
import json
from collections import Counter

# Example Command
# python -m utils.analyze.evaluate_model --show-top
# python -m utils.analyze.evaluate_model --analyze-failure --analyze-bias --show-top

CHECKPOINT = "data/models/best/best_model.pth"
LABELS_JSON = "data/screenshots/labels.json"
#LABELS_JSON = "data/dataset_default/labels.json"
#LABELS_JSON = "data/screenshots/labels.json"

TASK_TO_JSON_KEY = {
    "identification": "name",
    "modifier": "modifier",
    "type": "type",
    "rarity": "rarity"
}


# -----------------------------
# Model loading
# -----------------------------
def load_model(checkpoint_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    head_configs = {}
    class_names = {}
    for task, head_state in checkpoint["head_states"].items():
        class_names[task] = head_state["class_names"]
        head_configs[task] = len(head_state["class_names"])

    model = MultiHeadModel(head_configs=head_configs, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, class_names


# -----------------------------
# Preprocessing & inference
# -----------------------------
def preprocess_image(image_path, size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)


@torch.no_grad()
def infer(model, img_tensor, topk=1, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    outputs = model(img_tensor)
    results = {}
    for task, logits in outputs.items():
        probs = torch.softmax(logits, dim=1)
        top_probs, top_idx = torch.topk(probs, k=topk, dim=1)
        results[task] = {
            "indices": top_idx[0].cpu().tolist(),
            "probs": top_probs[0].cpu().tolist(),
            "all_probs": probs[0].cpu().tolist()
        }
    return results


def map_indices_to_labels(results, class_names):
    mapped = {}
    for task, data in results.items():
        mapped[task] = [class_names[task][idx] for idx in data["indices"]]
    return mapped


# -----------------------------
# Evaluation logic
# -----------------------------
def evaluate(model, class_names, labels_json, task_name=None, topk=1):
    with open(labels_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_per_task = {}
    correct_per_task = {}
    incorrect_entries = {}

    for img_file, labels in data.items():
        img_path = labels.get("full_path", img_file)
        img_tensor = preprocess_image(img_path)
        raw_results = infer(model, img_tensor, topk=topk)
        predicted = map_indices_to_labels(raw_results, class_names)

        tasks_to_check = [task_name] if task_name else predicted.keys()
        for task in tasks_to_check:
            json_key = TASK_TO_JSON_KEY.get(task, task)
            true_label = labels.get(json_key)
            pred_labels = predicted.get(task, [])

            total_per_task[task] = total_per_task.get(task, 0) + 1
            if true_label in pred_labels:
                correct_per_task[task] = correct_per_task.get(task, 0) + 1
            else:
                probs = raw_results[task]["all_probs"]
                true_idx = class_names[task].index(true_label) if true_label in class_names[task] else None
                true_prob = probs[true_idx] if true_idx is not None else 0.0
                pred_prob = raw_results[task]["probs"][0]
                incorrect_entries.setdefault(task, []).append({
                    "file": img_file,
                    "true": true_label,
                    "predicted": pred_labels,
                    "true_prob": true_prob,
                    "pred_prob": pred_prob,
                    "confidence_gap": pred_prob - true_prob
                })

    accuracy_per_task = {task: 100 * correct_per_task.get(task, 0) / total_per_task[task]
                         for task in total_per_task}

    return accuracy_per_task, incorrect_entries, total_per_task


# -----------------------------
# Display utilities
# -----------------------------
def print_all_incorrect(incorrect_entries):
    print("\n=== All Incorrect Predictions ===")
    for task, entries in incorrect_entries.items():
        print(f"\nTask: {task} ({len(entries)} incorrect samples)")
        for e in entries:
            print(f"File: {e['file']} | True: {e['true']} | Pred: {e['predicted'][0]} | "
                  f"P(pred)={e['pred_prob']:.3f}, P(true)={e['true_prob']:.3f}, Δ={e['confidence_gap']:.3f}")


def print_top_misclassified(incorrect_entries, total_per_task, top_n=20):
    print("\n=== Top Misclassified Samples ===")
    for task, entries in incorrect_entries.items():
        if not entries:
            continue
        entries.sort(key=lambda e: e["confidence_gap"], reverse=True)
        total_errors = len(entries)
        total_samples = total_per_task[task]
        top_n_subset = entries[:top_n]
        top_n_contrib = 100 * (len(top_n_subset) / total_samples)

        print(f"\n--- Task: {task} ---")
        print(f"Total incorrect: {total_errors}/{total_samples} ({100 * total_errors / total_samples:.2f}%)")
        print(f"Top {top_n} contribute ~{top_n_contrib:.2f}% of total dataset\n")

        for e in top_n_subset:
            print(f"{e['file']} | True: {e['true']} | Pred: {e['predicted'][0]} | "
                  f"P(pred)={e['pred_prob']:.3f}, P(true)={e['true_prob']:.3f}, Δ={e['confidence_gap']:.3f}")


def print_accuracy_summary(accuracy_per_task):
    print("\n=== Accuracy per Task ===")
    for task, acc in accuracy_per_task.items():
        print(f"{task}: {acc:.2f}%")


def print_prediction_distribution(incorrect_entries, total_per_task, top_n=20):
    """Show which predictions dominate the incorrect samples."""
    print("\n=== Prediction Bias Analysis (Overpredicted classes) ===")
    for task, entries in incorrect_entries.items():
        if not entries:
            continue
        preds = [e["predicted"][0] for e in entries]
        counter = Counter(preds)
        total_errors = len(entries)
        top_preds = counter.most_common(top_n)

        print(f"\n--- Task: {task} ---")
        print(f"Total incorrect: {total_errors}/{total_per_task[task]} ({100 * total_errors / total_per_task[task]:.2f}%)")
        print(f"Top {top_n} most frequent wrong predictions:\n")
        for label, count in top_preds:
            print(f"{label:<25} | {count:4d} errors ({100 * count / total_errors:5.2f}%)")
        print("-" * 60)


def print_true_label_failure_distribution(incorrect_entries, top_n=20):
    """Show which true labels are most often misclassified."""
    print("\n=== Failure Analysis (True labels most often misclassified) ===")
    for task, entries in incorrect_entries.items():
        if not entries:
            continue
        trues = [e["true"] for e in entries]
        counter = Counter(trues)
        total_errors = len(entries)
        top_trues = counter.most_common(top_n)

        print(f"\n--- Task: {task} ---")
        print(f"Total incorrect samples: {total_errors}")
        print(f"Top {top_n} hardest-to-recognize true labels:\n")
        for label, count in top_trues:
            print(f"{label:<25} | {count:4d} times ({100 * count / total_errors:5.2f}%)")
        print("-" * 60)


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate model on labeled JSON")
    parser.add_argument("--labels-json", type=str, default=LABELS_JSON)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--show-all", action="store_true", help="Show all incorrect predictions")
    parser.add_argument("--show-top", action="store_true", help="Show top-N misclassified predictions")
    parser.add_argument("--analyze-bias", action="store_true", help="Show distribution of overpredicted classes")
    parser.add_argument("--analyze-failure", action="store_true", help="Show which true labels fail most often")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names = load_model(args.checkpoint, device)
    accuracy, incorrect, total = evaluate(model, class_names, args.labels_json, args.task_name, args.topk)

    if args.show_all:
        print_all_incorrect(incorrect)
    if args.show_top:
        print_top_misclassified(incorrect, total, args.topn)
    if args.analyze_bias:
        print_prediction_distribution(incorrect, total, args.topn)
    if args.analyze_failure:
        print_true_label_failure_distribution(incorrect, args.topn)

    print_accuracy_summary(accuracy)


if __name__ == "__main__":
    main()
