import os
import random
from PIL import Image, ImageEnhance, ImageFilter

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
        for _ in range(int(0.01 * image.size[0] * image.size[1])):
            x = random.randint(0, image.size[0]-1)
            y = random.randint(0, image.size[1]-1)
            pixels[x, y] = tuple(random.randint(0, 255) for _ in range(3))
    return image

def augment_image(image: Image.Image):
    image = rotate_image(image)
    image = flip_image(image)
    image = adjust_brightness_contrast(image)
    image = add_blur_noise(image)
    return image

def augment_dataset(input_dir: str, output_dir: str, n_variations: int = 20):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(input_dir, filename)
        image = Image.open(img_path)
        name, ext = os.path.splitext(filename)
        for i in range(n_variations):
            aug_img = augment_image(image)
            out_path = os.path.join(output_dir, f"{name}_aug{i}{ext}")
            aug_img.save(out_path)
