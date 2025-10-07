# evaluate_model.py
import torch
from torchvision import transforms
from PIL import Image
from models.models import MultiHeadModel
import argparse
import json
import os

CHECKPOINT = "data/models/best-model-duo-45.pth"

# Example Command
# python -m utils.evaluate_model --labels-json "data/dataset_default/labels.JSON" 

# -----------------------------
# Load trained model
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
# Preprocess image
# -----------------------------
def preprocess_image(image_path, size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)

# -----------------------------
# Inference
# -----------------------------
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
            "probs": top_probs[0].cpu().tolist()
        }
    return results

# -----------------------------
# Map indices to class names
# -----------------------------
def map_indices_to_labels(results, class_names):
    mapped = {}
    for task, data in results.items():
        mapped[task] = [class_names[task][idx] for idx in data["indices"]]
    return mapped

# -----------------------------
# Map task name to JSON key
# -----------------------------
TASK_TO_JSON_KEY = {
    "identification": "name",
    "modifier": "modifier",
    "type": "type",
    "rarity": "rarity"
}

# -----------------------------
# Evaluation
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
                incorrect_entries.setdefault(task, []).append({
                    "file": img_file,
                    "true": true_label,
                    "predicted": pred_labels
                })

    accuracy_per_task = {task: 100 * correct_per_task.get(task,0)/total_per_task[task]
                         for task in total_per_task}

    return accuracy_per_task, incorrect_entries

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate model on labeled JSON")
    parser.add_argument("--labels-json", type=str, required=True, help="Path to JSON file with labels")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT, help="Path to trained checkpoint")
    parser.add_argument("--task-name", type=str, default=None, help="Optional: task to evaluate")
    parser.add_argument("--topk", type=int, default=1, help="Top-k predictions to consider correct")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names = load_model(args.checkpoint, device)
    accuracy, incorrect = evaluate(model, class_names, args.labels_json, task_name=args.task_name, topk=3)

    if incorrect:
        print("\n=== Incorrect predictions ===")
        for task, entries in incorrect.items():
            print(f"\nTask: {task}")
            for e in entries:
                print(f"File: {e['file']}, True: {e['true']}, Predicted: {e['predicted']}")
    
    print("=== Accuracy per task ===")
    for task, acc in accuracy.items():
        print(f"{task}: {acc:.2f}%")

if __name__ == "__main__":
    main()
