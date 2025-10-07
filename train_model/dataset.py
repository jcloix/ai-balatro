# train_model/dataset.py
from torch.utils.data import Dataset
from PIL import Image

class CardDataset(Dataset):
    """
    Dataset constructed from a labels dict (filename -> metadata).
    """
    def __init__(self, labels_dict, field="name", transform=None, class_names=None):
        self.data = labels_dict
        self.filenames = list(self.data.keys())
        self.transform = transform  # store transform

        # Use provided class_names (resume/inference), else infer
        if class_names is None:
            class_names = sorted({v[field] for v in self.data.values()})
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.classes = self.class_names   

        # Map class name → integer label
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        print(self.class_to_idx)

        # Store integer labels for each file
        self.labels_dict = {fname: self.class_to_idx[self.data[fname][field]] for fname in self.filenames}


    @classmethod
    def from_labels_dict(cls, labels_dict, field, transform, class_names):
        return cls(labels_dict=labels_dict, field=field, transform=transform, class_names=class_names)
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        label = self.labels_dict[fname]

        # Use full_path if available, fallback to fname
        meta = self.data[fname]
        img_path = meta.get("full_path", fname)

        image = Image.open(img_path).convert('RGB')

        if self.transform:  # apply transform if provided
            image = self.transform(image)

        return image, label



