# utils/analyze/extract_embeddings_stats.py
import torch
from torchvision import transforms
from PIL import Image
from models.models import MultiHeadModel
import argparse
import json
import os
import numpy as np
from sklearn.manifold import TSNE

CHECKPOINT = "data/models/best_model.pth"
LABELS_JSON = "data/dataset_base_only/labels.json"
OUTPUT_JSON = "data/embedding_bias.json"

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
def extract_embedding_and_pred(model, img_tensor, task_name, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    features = model.backbone(img_tensor)
    features = torch.flatten(features, 1)  # [B, 512]
    logits = model.heads[task_name](features)
    probs = torch.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_idx].item()
    return features.cpu().numpy()[0], pred_idx, confidence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-json", type=str, default=LABELS_JSON)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    parser.add_argument("--task-name", type=str, default="identification")
    parser.add_argument("--output-json", type=str, default=OUTPUT_JSON)
    parser.add_argument("--max-samples", type=int, default=2000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names = load_model(args.checkpoint, device)
    task = args.task_name

    with open(args.labels_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = []
    true_labels = []
    pred_labels = []
    confidences = []

    for i, (img_file, label_info) in enumerate(data.items()):
        if i >= args.max_samples:
            break
        img_path = label_info.get("full_path", img_file)
        img_tensor = preprocess_image(img_path)
        emb, pred_idx, conf = extract_embedding_and_pred(model, img_tensor, task, device)
        embeddings.append(emb)
        true_labels.append(label_info[TASK_TO_JSON_KEY[task]])
        pred_labels.append(class_names[task][pred_idx])
        confidences.append(conf)

    embeddings = np.array(embeddings)
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)

    output_data = []
    for i in range(len(emb_2d)):
        output_data.append({
            "true_label": true_labels[i],
            "pred_label": pred_labels[i],
            "confidence": confidences[i],
            "embedding_2d": emb_2d[i].tolist()
        })

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved embedding + prediction data to {args.output_json}")

if __name__ == "__main__":
    main()
