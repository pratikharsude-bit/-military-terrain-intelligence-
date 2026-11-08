
from pathlib import Path
from shutil import copy2
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "archive (8)" / "aerial_landscapes"
DEST = ROOT / "data" / "processed" / "SkyView"

SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def main():
    if not SOURCE.is_dir():
        print(f"ERROR: Dataset folder not found:\n{SOURCE}")
        print("Check the actual dataset folder name under data/raw.")
        sys.exit(1)

    classes = sorted(p for p in SOURCE.iterdir() if p.is_dir())

    if len(classes) != 15:
        print(f"ERROR: Expected 15 class folders, found {len(classes)}.")
        print("Check the source folder before continuing.")
        sys.exit(1)

    if DEST.exists() and any(DEST.iterdir()):
        print(f"ERROR: Destination already contains files:\n{DEST}")
        print("Nothing was overwritten. Check the existing data first.")
        sys.exit(1)

    # Split each class separately to preserve class balance.
    rng = random.Random(SEED)
    plan = []

    for class_dir in classes:
        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            print(f"ERROR: No images found in {class_dir.name}")
            sys.exit(1)

        rng.shuffle(images)

        n = len(images)
        n_train = int(n * 0.80)
        n_val = int(n * 0.10)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, files in splits.items():
            for image_path in files:
                plan.append((image_path, split_name, class_dir.name))

    print(f"Source:      {SOURCE}")
    print(f"Destination: {DEST}")
    print(f"Images to copy: {len(plan)}")
    print("Split: approximately 80% train / 10% val / 10% test")
    print("\nStarting copy...")

    for index, (image_path, split_name, class_name) in enumerate(plan, 1):
        target_dir = DEST / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        copy2(image_path, target_dir / image_path.name)

        if index % 500 == 0:
            print(f"Copied {index}/{len(plan)} images")

    print("\nSplit complete!")
    for split_name in ("train", "val", "test"):
        count = sum(
            1 for p in (DEST / split_name).rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"{split_name}: {count} images")

    print(f"\nOutput folder: {DEST}")

if __name__ == "__main__":
    main()
