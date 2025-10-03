import argparse
import torch
from PIL import Image
from torchvision import transforms
from models.models import build_model  
from config.config import BEST_MODEL_PATH, NB_CLASSES

# same preprocessing as training
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# cache model so it’s not reloaded every call
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def _load_model():
    global _model
    if _model is None:
        model, _ = build_model(num_classes=NB_CLASSES)
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=_device)
        model.load_state_dict(checkpoint)
        model.eval()
        _model = model.to(_device)
    return _model

def identify_card(img_path, topk=1):
    """Run inference on a card and return predicted class (or top-K)."""
    model = _load_model()

    img = Image.open(img_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_idxs = probs.topk(topk, dim=1)

    if topk == 1:
        return top_idxs[0][0].item()
    else:
        return [(i.item(), p.item()) for i, p in zip(top_idxs[0], top_probs[0])]
    

def main():
    parser = argparse.ArgumentParser(description="Predict the card from an image")
    parser.add_argument("img_path", type=str, help="Path to the card image")
    parser.add_argument("--topk", type=int, default=1, help="Return top-K predictions")
    args = parser.parse_args()

    preds = identify_card(args.img_path, topk=args.topk)

    if args.topk == 1:
        idx = preds
        print(f"Predicted card: {idx} (index {idx})")
    else:
        print("Top-K predictions:")
        for idx, prob in preds:
            print(f"{idx}: {prob:.2f}")

if __name__ == "__main__":
    main()