# train_model/inference_utils.py
import torch
from torchvision import transforms
from PIL import Image
from models.models import MultiHeadModel  # your MultiHeadModel class
import argparse
import json

CHECKPOINT="data/models/best_model.pth"

# -----------------------------
# Load trained model
# -----------------------------
def load_model(checkpoint_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Build head_configs from the saved head states
    head_configs = {}
    class_names = {}
    for task, head_state in checkpoint["head_states"].items():
        class_names[task] = head_state["class_names"]
        head_configs[task] = len(head_state["class_names"])

    # Build model
    model = MultiHeadModel(head_configs=head_configs, pretrained=False)
    model.load_state_dict(checkpoint["model"])  # your persistence uses "model"
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
def infer(model, images, task_name=None, topk=1, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device)
    outputs = model(images)

    results = {}
    tasks = [task_name] if task_name else outputs.keys()
    for task in tasks:
        logits = outputs[task]
        probs = torch.softmax(logits, dim=1)
        top_probs, top_idx = torch.topk(probs, k=topk, dim=1)
        results[task] = {
            "indices": top_idx[0].cpu().tolist(),
            "probs": top_probs[0].cpu().tolist()
        }
    return results

# -----------------------------
# Map predicted indices to class names
# -----------------------------
def map_indices_to_labels(results, class_names):
    mapped = {}
    for task, data in results.items():
        mapped[task] = [
            class_names[task][i] for i in data["indices"]
        ]
    return mapped

# -----------------------------
# Main function
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--img-path", type=str, required=True, help="Path to image")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT, help="Path to trained checkpoint")
    parser.add_argument("--task-name", type=str, default=None, help="Optional: task to infer")
    parser.add_argument("--topk", type=int, default=1, help="Return top-k predictions")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names = load_model(args.checkpoint, device)
    img_tensor = preprocess_image(args.img_path)

    results = infer(model, img_tensor, task_name=args.task_name, topk=args.topk, device=device)
    mapped = map_indices_to_labels(results, class_names)

    print(json.dumps({"raw": results, "labels": mapped}, indent=4))


if __name__ == "__main__":
    main()
# python -m utils.inference_utils --img-path "data/unlabeled/cluster1_card36_id178.png" --task-name identification