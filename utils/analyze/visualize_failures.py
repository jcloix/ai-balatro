# utils/analyze/visualize_failures.py
import torch
from torchvision import transforms
from PIL import Image
from models.models import MultiHeadModel
import argparse
import json
import os
from math import ceil
import matplotlib.pyplot as plt

CHECKPOINT = "data/models/best_model.pth"
LABELS_JSON = "data/dataset_base_only/labels.json"

TASK_TO_JSON_KEY = {
    "identification": "name",
    "modifier": "modifier",
    "type": "type",
    "rarity": "rarity"
}

def load_model(checkpoint_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    head_configs = {task: len(h["class_names"]) for task, h in checkpoint["head_states"].items()}
    class_names = {task: h["class_names"] for task, h in checkpoint["head_states"].items()}
    model = MultiHeadModel(head_configs=head_configs, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, class_names

def preprocess_image(image_path, size=(224,224)):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    return transform(img).unsqueeze(0)

@torch.no_grad()
def infer(model, img_tensor, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    outputs = model(img_tensor)
    results = {}
    for task, logits in outputs.items():
        probs = torch.softmax(logits, dim=1)
        top_idx = torch.argmax(probs, dim=1)
        results[task] = {
            "pred_label": top_idx[0].item(),
            "probs": probs[0].cpu().tolist()
        }
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-json", type=str, default=LABELS_JSON)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    parser.add_argument("--predicted-label", type=str, required=True, help="Class to filter (overpredicted)")
    parser.add_argument("--task-name", type=str, default="identification")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--output-dir", type=str, default="biased_grid")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names = load_model(args.checkpoint, device)

    with open(args.labels_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Count how many examples exist per true label
    true_label_counts = {}
    json_key = TASK_TO_JSON_KEY.get(args.task_name, args.task_name)
    for labels in data.values():
        lbl = labels.get(json_key)
        if lbl:
            true_label_counts[lbl] = true_label_counts.get(lbl, 0) + 1

    selected_images = []
    selected_true_labels = []
    target_idx = class_names[args.task_name].index(args.predicted_label)

    for img_file, labels in data.items():
        img_path = labels.get("full_path", img_file)
        img_tensor = preprocess_image(img_path)
        result = infer(model, img_tensor, device)
        pred_idx = result[args.task_name]["pred_label"]
        if pred_idx == target_idx:
            selected_images.append(img_path)
            selected_true_labels.append(labels.get(json_key))
            if len(selected_images) >= args.limit:
                break

    # Plot grid with true label counts
    cols = 5
    rows = ceil(len(selected_images)/cols)
    plt.figure(figsize=(cols*3, rows*3))
    for i, (img_path, true_label) in enumerate(zip(selected_images, selected_true_labels)):
        img = Image.open(img_path).convert("RGB")
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        count = true_label_counts.get(true_label, 0)
        plt.title(f"{true_label}\n(n={count})", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{args.predicted_label}_grid_with_counts.png"))
    print(f"Saved grid to {args.output_dir}/{args.predicted_label}_grid_with_counts.png")

if __name__ == "__main__":
    main()
