import argparse
from augment import augment_dataset

# Default directories
INPUT_DIR = "dataset/labeled"
OUTPUT_DIR = "dataset/augmented"

def parse_args():
    parser = argparse.ArgumentParser(description="Augment labeled card images")
    parser.add_argument("--input", type=str, default=INPUT_DIR, help="Input folder with labeled images")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output folder for augmented images")
    parser.add_argument("--n", type=int, default=20, help="Number of variations per image")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Augmenting images from {args.input} → {args.output} ({args.n} variations per image)")
    augment_dataset(args.input, args.output, args.n)
    print("Augmentation complete!")

if __name__ == "__main__":
    main()
