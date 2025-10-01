import torch
from torchvision import transforms
import json
from train_model.dataset import CardDataset
from train_model.train_setup import prepare_training
from train_model.train_config import Config

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "data/models/best_model.pth"  # path to your trained model
LABELS_PATH = "data/merged_labels.json"
VAL_SPLIT = 0.1
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load dataset
# -----------------------------
with open(LABELS_PATH, 'r') as f:
    labels_dict = json.load(f)

dataset = CardDataset.from_labels_dict(labels_dict)

# Split dataset
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
_, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

val_dataset.dataset.transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Load model
# -----------------------------
num_classes = len(dataset.class_names)
state = prepare_training(num_classes=num_classes, log_dir=None, lr=Config.LEARNING_RATE)
state.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
state.model.to(DEVICE)
state.model.eval()

# -----------------------------
# Check predictions
# -----------------------------
print(f"{'Filename':40} | {'GT':15} | {'Pred':15} | {'Correct'}")
print("-" * 80)

# Track per-class stats
class_correct = {cls_name: 0 for cls_name in dataset.class_names}
class_total = {cls_name: 0 for cls_name in dataset.class_names}

not_correct = []
with torch.no_grad():
    for idx, (images, labels) in enumerate(val_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = state.model(images)
        _, preds = torch.max(outputs, 1)

        filename = val_dataset.dataset.filenames[val_dataset.indices[idx]]
        gt_name = dataset.idx_to_class[labels.item()]
        pred_name = dataset.idx_to_class[preds.item()]
        correct = gt_name == pred_name

        # Print per-sample
        print(f"{filename:40} | {gt_name:15} | {pred_name:15} | {correct}")

        # Update per-class stats
        class_total[gt_name] += 1
        if correct:
            class_correct[gt_name] += 1
        else:
            not_correct.append(f"{filename:40} | {gt_name:15} | {pred_name:15} | {correct}")

# -----------------------------
# Print per-class accuracy
# -----------------------------
print("\nPer-class Accuracy:")
print(f"{'Class':20} | {'Accuracy':10} | {'Samples'}")
print("-" * 45)
for cls_name in dataset.class_names:
    total = class_total[cls_name]
    correct = class_correct[cls_name]
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"{cls_name:20} | {accuracy:9.2f}% | {total}")

for s in not_correct:
    print(s)