
from pathlib import Path
from collections import Counter
import csv

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "archive (5)"

def main():
    class_file = DATA_DIR / "class_dict.csv"
    metadata_file = DATA_DIR / "metadata.csv"

    if not class_file.exists() or not metadata_file.exists():
        raise FileNotFoundError("class_dict.csv or metadata.csv is missing.")

    # Read the official RGB-to-class mapping
    class_map = {}
    with class_file.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rgb = (int(row["r"]), int(row["g"]), int(row["b"]))
            class_map[rgb] = row["name"]

    print("=" * 55)
    print("DEEPGLOBE MASK VERIFICATION")
    print("=" * 55)

    print("\nOfficial class mapping:")
    for rgb, name in class_map.items():
        print(f"  RGB {rgb}: {name}")

    # Check image-mask pairs using metadata
    with metadata_file.open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))

    train_records = [r for r in records if r["split"].lower() == "train"]
    valid_records = [r for r in records if r["split"].lower() == "valid"]
    test_records = [r for r in records if r["split"].lower() == "test"]

    print("\nMetadata records:")
    print(f"  Train: {len(train_records)}")
    print(f"  Valid: {len(valid_records)}")
    print(f"  Test:  {len(test_records)}")

    missing_images = 0
    missing_masks = 0
    dimension_mismatches = 0
    pixel_counts = Counter()
    checked_masks = 0

    for record in train_records:
        image_path = DATA_DIR / record["sat_image_path"]
        mask_path = DATA_DIR / record["mask_path"]

        if not image_path.is_file():
            missing_images += 1
            continue

        if not mask_path.is_file():
            missing_masks += 1
            continue

        with Image.open(image_path) as img, Image.open(mask_path) as mask:
            image_size = img.size
            mask_rgb = mask.convert("RGB")
            mask_size = mask_rgb.size

            if image_size != mask_size:
                dimension_mismatches += 1

            if checked_masks == 0:
                print(f"\nSample image size: {image_size}")
                print(f"Sample mask size:  {mask_size}")
                print(f"Original mask mode: {mask.mode}")

            # Count RGB colours on the first 20 masks only
            if checked_masks < 20:
                pixel_counts.update(mask_rgb.getdata())

            checked_masks += 1

    print("\nPairing checks:")
    print(f"  Checked training pairs: {checked_masks}")
    print(f"  Missing satellite images: {missing_images}")
    print(f"  Missing masks: {missing_masks}")
    print(f"  Image/mask size mismatches: {dimension_mismatches}")

    print("\nRGB colours found in the first 20 masks:")
    for rgb, count in pixel_counts.most_common():
        label = class_map.get(rgb, "UNKNOWN RGB")
        print(f"  {rgb} | {label:<20} | {count:,} pixels")

    unknown_colours = set(pixel_counts) - set(class_map)
    print(f"\nUnmapped RGB colours: {len(unknown_colours)}")
    if unknown_colours:
        print("Examples:", list(unknown_colours)[:10])

    print("\nVerification finished. Dataset files were not modified.")

if __name__ == "__main__":
    main()
