import argparse
from augment_dataset.augment import augment_dataset
from config.config import DATASET_DIR, DATASET_AUGMENTED_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Augment labeled card images (instance-level)")
    parser.add_argument("--input", type=str, default=DATASET_DIR, help="Input folder with labeled images")
    parser.add_argument("--output", type=str, default=DATASET_AUGMENTED_DIR, help="Output folder for augmented images")
    parser.add_argument("--n", type=int, default=20, help="Number of variations per image")
    parser.add_argument("--no-rotate", action="store_true", help="Disable rotation")
    parser.add_argument("--no-flip", action="store_true", help="Disable flipping")
    parser.add_argument("--no-bc", action="store_true", help="Disable brightness/contrast adjustment")
    parser.add_argument("--no-blur", action="store_true", help="Disable blur/noise addition")
    parser.add_argument("--no-negative", action="store_true", help="Disable negative addition")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Augmenting images from {args.input} → {args.output} ({args.n} variations per image)")

    augment_dataset(
        input_dir=args.input,
        output_dir=args.output,
        n_variations=args.n,
        rotate=not args.no_rotate,
        flip=not args.no_flip,
        brightness_contrast=not args.no_bc,
        blur_noise=not args.no_blur,
        negative=not args.no_negative,
        seed=args.seed
    )

    print("Augmentation complete!")

if __name__ == "__main__":
    main()
