# train_model/train_config.py
from torchvision import transforms

class TransformConfig:
    """
    Predefined image transforms for different modes.
    """
    # Training transforms: augmentations applied on-the-fly
    TRAIN = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet
                             std=[0.229, 0.224, 0.225])
    ])

    # Test / validation transforms: no random augmentations
    TEST = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Heavy augmentations: more aggressive augmentations for robustness
    HEAVY = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Light augmentations: minimal augmentation, mostly normalization
    LIGHT = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Minimal transforms: resize + tensor only
    NONE = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


class Config:
    """
    Training hyperparameters and constants
    """
    # Training hyperparameters
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    VAL_SPLIT = 0.2
    PATIENCE = 5
    CHECKPOINT_INTERVAL = 5
    OUTPUT_SPLIT_TRAIN_VAL = "data/split_train_val.json"

    # Number of classes (can be overridden at runtime)
    NUM_CLASSES = 200

    # Map transform modes to objects for easy lookup
    TRANSFORMS = {
        "train": TransformConfig.TRAIN,
        "test": TransformConfig.TEST,
        "none": TransformConfig.NONE,
        "heavy": TransformConfig.HEAVY,  # can add more aggressive augmentations
        "light": TransformConfig.LIGHT    # lighter augmentations
    }
