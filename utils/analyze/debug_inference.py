# utils/analyze/debug_inference.py
import torch
from torchvision import transforms
from PIL import Image
from models.models import MultiHeadModel
import argparse
import json
import os
import numpy as np

CHECKPOINT = "data/models/best_model.pth"
LABELS_JSON = "data/dataset_base_only/labels.json"
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    head_configs = {task: len(h["class_names"]) for task,h in checkpoint["head_states"].items()}
    class_names = {task: h["class_names"] for task,h in checkpoint["head_states"].items()}
    model = MultiHeadModel(head_configs=head_configs, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, class_names

def make_preproc(kind="eval", size=224):
    if kind == "train":
        # approximate train-time transform (RandomResizedCrop + Normalize)
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.7,1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    elif kind == "val":
        # common validation transform: Resize + CenterCrop + Normalize
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    else:
        # simple eval (what you had): Resize + ToTensor (no Normalize)
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

def infer_topk(model, img_tensor, topk=5, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        outs = model(img_tensor)
    # assume 'identification' head exists
    logits = outs['identification'] if 'identification' in outs else list(outs.values())[0]
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    topk_idx = probs.argsort()[::-1][:topk]
    topk_probs = probs[topk_idx]
    return topk_idx.tolist(), topk_probs.tolist(), probs

def load_image(path):
    return Image.open(path).convert("RGB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a single image to debug")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    parser.add_argument("--labels-json", type=str, default=LABELS_JSON)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names = load_model(args.checkpoint, device)
    with open(args.labels_json, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # find image entry in labels.json (optional)
    found = None
    for k, v in labels.items():
        full = v.get("full_path", k)
        if os.path.normpath(full) == os.path.normpath(args.image) or os.path.normpath(k) == os.path.normpath(args.image):
            found = v
            break

    print("Image:", args.image)
    if found:
        print("True label (from JSON):", found.get("name"))

    preprocess_eval = make_preproc("eval", size=args.size)
    preprocess_val  = make_preproc("val", size=args.size)
    preprocess_train= make_preproc("train", size=args.size)

    img = load_image(args.image)

    def run_and_print(prep, tag):
        tensor = prep(img).unsqueeze(0)
        idxs, probs, allp = infer_topk(model, tensor, topk=args.topk, device=device)
        names = [class_names['identification'][i] for i in idxs]
        print(f"\n--- {tag} preprocess ---")
        for rank, (n, p, i) in enumerate(zip(names, probs, idxs), start=1):
            print(f"{rank:02d}. {n} (idx={i})  prob={p:.4f}")
        # also report top-1 confidence and true_idx prob if available
        top1_name = class_names['identification'][idxs[0]]
        top1_conf = probs[0]
        print("Top1:", top1_name, f"{top1_conf:.4f}")

    run_and_print(preprocess_eval, "EVAL (no normalize)")
    run_and_print(preprocess_val,  "VAL  (Resize+CenterCrop+Normalize)")
    run_and_print(preprocess_train,"TRAIN (RandomResizedCrop+Normalize)")

if __name__ == "__main__":
    main()
