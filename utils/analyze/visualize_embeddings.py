# utils/analyze/visualize_embeddings.py
import torch
from torchvision import transforms
from PIL import Image
from models.models import MultiHeadModel
import argparse
import json
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

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
def extract_embedding(model, img_tensor):
    """Extract the backbone embedding before the head."""
    features = model.backbone(img_tensor)           # [B, 512, 1, 1]
    features = torch.flatten(features, 1)           # [B, 512]
    return features.cpu().numpy()[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-json", type=str, default=LABELS_JSON)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    parser.add_argument("--task-name", type=str, default="identification")
    parser.add_argument("--output-dir", type=str, default="embedding_viz")
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names = load_model(args.checkpoint, device)

    with open(args.labels_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = []
    labels = []

    for i, (img_file, label_info) in enumerate(data.items()):
        if i >= args.max_samples:
            break
        img_path = label_info.get("full_path", img_file)
        img_tensor = preprocess_image(img_path).to(device)
        emb = extract_embedding(model, img_tensor)
        embeddings.append(emb)
        labels.append(label_info[TASK_TO_JSON_KEY[args.task_name]])

    embeddings = np.array(embeddings)
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 12))
    unique_labels = list(set(labels))
    colors = plt.cm.get_cmap("tab20", len(unique_labels))

    for idx, lbl in enumerate(unique_labels):
        pts = emb_2d[np.array(labels) == lbl]
        plt.scatter(pts[:,0], pts[:,1], label=lbl, alpha=0.6, s=30, color=colors(idx))
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.title(f"t-SNE Embedding Visualization ({args.task_name})")
    plt.tight_layout()
    out_path = os.path.join(args.output_dir, f"{args.task_name}_tsne.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved embedding visualization to {out_path}")

if __name__ == "__main__":
    main()
