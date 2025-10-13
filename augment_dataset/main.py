import argparse
from augment_dataset.augment import AugmentConfig, DatasetAugmentor
from config.config import DATASET_DIR, DATASET_AUGMENTED_DIR, LABELS_FILE

# ============================================================
# 4. CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Augment labeled card images")
    parser.add_argument("--input", type=str, default=DATASET_DIR)
    parser.add_argument("--output", type=str, default=DATASET_AUGMENTED_DIR)
    parser.add_argument("--labels", type=str, default=LABELS_FILE)
    parser.add_argument("--n", type=int, default=20, help="Fixed augmentations per image")
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--no-bc", action="store_true")
    parser.add_argument("--no-blur", action="store_true")
    parser.add_argument("--with-negative", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--balance-by", type=str, choices=["name", "modifier"],
                        help="Field to balance by if using --balanced")
    return parser.parse_args()


# ============================================================
# 5. Main
# ============================================================
def main():
    args = parse_args()

    config = AugmentConfig(
        rotate=not args.no_rotate,
        flip=not args.no_flip,
        brightness_contrast=not args.no_bc,
        blur_noise=not args.no_blur,
        negative=args.with_negative,
        seed=args.seed
    )

    augmentor = DatasetAugmentor(
        input_dir=args.input,
        output_dir=args.output,
        labels_path=args.labels,
        config=config
    )

    if args.balance_by:
        print(f"Running balanced augmentation by '{args.balance_by}'")
        augmentor.augment_balanced(balance_by=args.balance_by)
    else:
        print(f"Running fixed augmentation ({args.n} per image)")
        augmentor.augment_fixed(n_variations=args.n)

    print("✅ Augmentation complete!")


if __name__ == "__main__":
    main()