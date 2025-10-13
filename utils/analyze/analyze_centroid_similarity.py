# utils/analyze/analyze_centroid_similarity.py
import torch
from torchvision import transforms
from PIL import Image
from models.models import MultiHeadModel
import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# python -m utils.analyze.analyze_centroid_similarity --focus "Invisible joker" --top-n 10

CHECKPOINT = "data/models/best_model.pth"
LABELS_JSON = "data/dataset_base_only/labels.json"
OUTPUT = "data/centroid_similarity.json"

TASK_TO_JSON_KEY = {
    "identification": "name",
    "modifier": "modifier",
    "type": "type",
    "rarity": "rarity"
}


def load_model(checkpoint_path, device=None):
    """Load the trained MultiHeadModel without reinitializing heads."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    head_configs = {task: len(h["class_names"]) for task, h in checkpoint["head_states"].items()}
    class_names = {task: h["class_names"] for task, h in checkpoint["head_states"].items()}
    model = MultiHeadModel(head_configs=head_configs, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, class_names


def preprocess_image(image_path, size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    return transform(img).unsqueeze(0)


@torch.no_grad()
def extract_embedding(model, img_tensor, device):
    """Extract 512-dim backbone embedding."""
    img_tensor = img_tensor.to(device)
    features = model.backbone(img_tensor)  # [B, 512, 1, 1]
    features = torch.flatten(features, 1)  # [B, 512]
    return features.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-json", type=str, default=LABELS_JSON)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    parser.add_argument("--task-name", type=str, default="identification")
    parser.add_argument("--output-json", type=str, default=OUTPUT)
    parser.add_argument("--max-per-class", type=int, default=50,
                        help="Limit number of samples per class for efficiency.")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Save only the top-N most similar classes per label.")
    parser.add_argument("--focus", type=str, default=None,
                        help="If set, only output similarities for this specific label.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names = load_model(args.checkpoint, device)
    key = TASK_TO_JSON_KEY[args.task_name]

    with open(args.labels_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect embeddings per class
    class_embeddings = {name: [] for name in class_names[args.task_name]}

    print("Extracting embeddings per class...")
    for img_file, label_info in tqdm(data.items(), total=len(data)):
        label = label_info.get(key)
        if label not in class_embeddings:
            continue

        if len(class_embeddings[label]) >= args.max_per_class:
            continue

        img_path = label_info.get("full_path", img_file)
        if not os.path.exists(img_path):
            continue

        try:
            img_tensor = preprocess_image(img_path)
            emb = extract_embedding(model, img_tensor, device)
            class_embeddings[label].append(emb)
        except Exception:
            continue

    # Compute centroids and counts
    centroids = {}
    counts = {}
    for label, embs in class_embeddings.items():
        if len(embs) > 0:
            centroids[label] = np.mean(embs, axis=0)
            counts[label] = len(embs)

    labels = list(centroids.keys())
    centroid_matrix = np.array([centroids[lbl] for lbl in labels])

    print("Computing cosine similarities...")
    sim_matrix = cosine_similarity(centroid_matrix)

    output = {}
    if args.focus:
        # Focus mode: only output one label
        if args.focus not in labels:
            raise ValueError(f"Focus label '{args.focus}' not found in centroids.")
        idx = labels.index(args.focus)
        sims = sim_matrix[idx]
        sorted_idx = np.argsort(sims)[::-1]  # descending
        top_idx = [i for i in sorted_idx if labels[i] != args.focus][:args.top_n]
        output = {
            "focus": args.focus,
            "samples_used": counts[args.focus],
            "top_similarities": {
                labels[i]: {
                    "similarity": float(sims[i]),
                    "samples_used": counts[labels[i]]
                }
                for i in top_idx
            }
        }
    else:
        # Output for all labels
        for i, label in enumerate(labels):
            sims = sim_matrix[i]
            sorted_idx = np.argsort(sims)[::-1]
            top_idx = [j for j in sorted_idx if j != i][:args.top_n]
            output[label] = {
                "samples_used": counts[label],
                "top_similarities": {
                    labels[j]: {
                        "similarity": float(sims[j]),
                        "samples_used": counts[labels[j]]
                    }
                    for j in top_idx
                }
            }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
