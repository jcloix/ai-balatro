# utils/inference_utils.py
import torch
from torchvision import transforms
from PIL import Image
from models.models import MultiHeadModel  # your MultiHeadModel class
import argparse
import json
import os
import glob

# Example Command
# python -m utils.inference_utils --img-dir "data/dataset_default" --task-name identification --topk 2 --out-json "data/dataset_default/inference.json"
# python -m utils.inference_utils --img-path "data/dataset_default/cluster1_card36_id610.png" --topk 3 

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
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model, class_names

# -----------------------------
# Preprocess image
# -----------------------------
def preprocess_image(image, size=(224, 224)):
    """
    Preprocess an image for the model.
    Accepts either a PIL.Image or a file path.
    """
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        raise ValueError(f"Invalid input type for preprocess_image: {type(image)}")

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
def map_indices_to_labels(results, class_names, threshold=None):
    mapped = {}
    for task, data in results.items():
        labels = []
        for idx, prob in zip(data["indices"], data["probs"]):
            if threshold is None or prob >= threshold:
                labels.append({"label": class_names[task][idx], "probability": float(prob)})
        mapped[task] = labels
    return mapped

# -----------------------------
# Run inference on folder
# -----------------------------
def infer_folder(model, folder_path, class_names, task_name=None, topk=3, threshold=None):
    results_dict = {}
    image_paths = glob.glob(os.path.join(folder_path, "*.png"))
    for img_path in image_paths:
        img_tensor = preprocess_image(img_path)
        raw_results = infer(model, img_tensor, task_name=task_name, topk=topk)
        mapped = map_indices_to_labels(raw_results, class_names, threshold)
        results_dict[os.path.basename(img_path)] = {"raw": raw_results, "labels": mapped}
    return results_dict

# -----------------------------
# Main function
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run inference on images")
    parser.add_argument("--img-path", type=str, help="Path to single image")
    parser.add_argument("--img-dir", type=str, help="Path to folder of images")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT, help="Path to trained checkpoint")
    parser.add_argument("--task-name", type=str, default=None, help="Optional: task to infer")
    parser.add_argument("--topk", type=int, default=3, help="Return top-k predictions")
    parser.add_argument("--threshold", type=float, default=None, help="Probability threshold to keep predictions")
    parser.add_argument("--out-json", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names = load_model(args.checkpoint, device)

    if args.img_path:
        img_tensor = preprocess_image(args.img_path)
        results = infer(model, img_tensor, task_name=args.task_name, topk=args.topk)
        mapped = map_indices_to_labels(results, class_names, threshold=args.threshold)
        output = {os.path.basename(args.img_path): {"raw": results, "labels": mapped}}
    elif args.img_dir:
        output = infer_folder(model, args.img_dir, class_names, task_name=args.task_name,
                              topk=args.topk, threshold=args.threshold)
    else:
        raise ValueError("Either --img-path or --img-dir must be provided")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[✓] Saved inference results to {args.out_json}")
    else:
        print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
