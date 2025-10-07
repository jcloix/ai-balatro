import os
import json
import random
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# ============================================================
# 1. Standalone augmentation functions
# ============================================================

def rotate_image(image: Image.Image, max_angle: int = 15):
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle, expand=True)

def flip_image(image: Image.Image):
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image

def adjust_brightness_contrast(image: Image.Image,
                               brightness_range=(0.8, 1.2),
                               contrast_range=(0.8, 1.2)):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(*brightness_range))
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(*contrast_range))
    return image

def add_blur_noise(image: Image.Image, blur_prob=0.3, noise_prob=0.3):
    if random.random() < blur_prob:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    if random.random() < noise_prob:
        pixels = image.load()
        mode = image.mode
        for _ in range(int(0.01 * image.size[0] * image.size[1])):
            x = random.randint(0, image.size[0]-1)
            y = random.randint(0, image.size[1]-1)
            if mode == "RGBA":
                r, g, b, a = pixels[x, y]
                pixels[x, y] = (random.randint(0,255), random.randint(0,255), random.randint(0,255), a)
            else:
                pixels[x, y] = tuple(random.randint(0,255) for _ in range(3))
    return image

def apply_negative(image: Image.Image):
    if image.mode == "RGBA":
        r, g, b, a = image.split()
        rgb_image = Image.merge("RGB", (r, g, b))
        inverted = ImageOps.invert(rgb_image)
        r2, g2, b2 = inverted.split()
        return Image.merge("RGBA", (r2, g2, b2, a))
    else:
        return ImageOps.invert(image)

def augment_image(image: Image.Image, rotate=True, flip=True,
                  brightness_contrast=True, blur_noise=True, negative=False):
    """Apply all selected augmentations to a single image"""
    if rotate:
        image = rotate_image(image)
    if flip:
        image = flip_image(image)
    if brightness_contrast:
        image = adjust_brightness_contrast(image)
    if blur_noise:
        image = add_blur_noise(image)
    if negative:
        image = apply_negative(image)
    return image


# ============================================================
# 2. Configuration class (just holds options)
# ============================================================
class AugmentConfig:
    def __init__(self, rotate=True, flip=True, brightness_contrast=True,
                 blur_noise=True, negative=False, seed=None):
        self.rotate = rotate
        self.flip = flip
        self.brightness_contrast = brightness_contrast
        self.blur_noise = blur_noise
        self.negative = negative

        if seed is not None:
            random.seed(seed)

    def apply(self, image: Image.Image):
        return augment_image(image,
                             rotate=self.rotate,
                             flip=self.flip,
                             brightness_contrast=self.brightness_contrast,
                             blur_noise=self.blur_noise,
                             negative=self.negative)


# ============================================================
# 3. DatasetAugmentor class
# ============================================================
class DatasetAugmentor:
    def __init__(self, input_dir, output_dir, labels_path, config: AugmentConfig):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = config

        if not labels_path or not os.path.exists(labels_path):
            raise ValueError(f"Labels file missing {self.labels_path}.")

        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

        os.makedirs(self.output_dir, exist_ok=True)

    def _load_image(self, filename):
        path = os.path.join(self.input_dir, filename)
        try:
            return Image.open(path).convert("RGBA")
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            return None

    def compute_n_augs(self, count, all_counts, balance_by):
        """Heuristic: fewer samples → more augmentations"""
        if balance_by == "name":
            return self._compute_n_abs_aug(count)
        return self._compute_n_relative_aug(count, all_counts)

    def _compute_n_abs_aug(self, count, min_aug=2, max_aug=20):
        if count >= 15:
            return min_aug
        elif count >= 5:
            return 8
        else:
            return max_aug
    
    def _compute_n_relative_aug(self, count, all_counts, min_aug=2, mid1=5, mid2=10, mid3=15, max_aug=20):
        largest = max(all_counts)
        ratio = count / largest
        if count >= 0.66 * largest:  # top ~1/3
            return min_aug
        elif count >= 0.5 * largest:  
            return mid1
        elif count >= 0.2 * largest:  
            return mid2
        elif count >= 0.1 * largest:  
            return mid3
        else:  # small groups
            return max_aug

    # --- Fixed augmentation ---
    def augment_fixed(self, n_variations=5):
        for filename in tqdm(os.listdir(self.input_dir), desc="Fixed augmentation"):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            image = self._load_image(filename)
            if image is None:
                continue
            base, ext = os.path.splitext(filename)
            for i in range(n_variations):
                aug_img = self.config.apply(image)
                out_path = os.path.join(self.output_dir, f"{base}_aug{i}{ext}")
                aug_img.save(out_path)

    # --- Balanced augmentation by a selected field ---
    def augment_balanced(self, balance_by="name"):
        """
        balance_by: "name" or "modifier"
        """
        card_to_files = {}
        for fname, info in self.labels.items():
            key = info.get(balance_by)
            if not key:
                continue
            card_to_files.setdefault(key, []).append(fname)

        print(f"Found {len(card_to_files)} groups by '{balance_by}'")

        all_counts = [len(files) for files in card_to_files.values()]
        for key, files in tqdm(card_to_files.items(), desc=f"Balanced by {balance_by}"):
            count = len(files)
            n_augs = self.compute_n_augs(count, all_counts, balance_by)

            for filename in files:
                image = self._load_image(filename)
                if image is None:
                    continue
                base, ext = os.path.splitext(filename)
                for i in range(n_augs):
                    aug_img = self.config.apply(image)
                    out_path = os.path.join(self.output_dir, f"{base}_aug{i}{ext}")
                    aug_img.save(out_path)