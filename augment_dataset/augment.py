import os
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
from tqdm import tqdm

def rotate_image(image: Image.Image, max_angle: int = 15):
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle, expand=True)

def flip_image(image: Image.Image):
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image

def adjust_brightness_contrast(image: Image.Image, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
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
    """Invert colors to simulate a negative filter"""
    if image.mode == "RGBA":
        r, g, b, a = image.split()
        rgb_image = Image.merge("RGB", (r, g, b))
        inverted = ImageOps.invert(rgb_image)
        r2, g2, b2 = inverted.split()
        return Image.merge("RGBA", (r2, g2, b2, a))
    else:
        return ImageOps.invert(image)

def augment_image(image: Image.Image, rotate=True, flip=True, brightness_contrast=True, blur_noise=True, negative=True):
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

def augment_dataset(input_dir: str, output_dir: str, n_variations: int = 5,
                    rotate=True, flip=True, brightness_contrast=True, blur_noise=True, negative=True, seed: int = None):
    """
    Augment each image in input_dir with n_variations.
    Output images are named <original_name>_aug<N>.<ext> to preserve card identity.
    """
    if seed is not None:
        random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir), desc="Augmenting images"):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(input_dir, filename)
        try:
            image = Image.open(img_path).convert("RGBA")
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        name, ext = os.path.splitext(filename)
        for i in range(n_variations):
            aug_img = augment_image(image, rotate, flip, brightness_contrast, blur_noise, negative)
            out_path = os.path.join(output_dir, f"{name}_aug{i}{ext}")
            aug_img.save(out_path)
