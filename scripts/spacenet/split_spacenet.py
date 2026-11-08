import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/processed/SpaceNet7"
SPLIT_DIR = DATA_DIR / "splits"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

# Group image-mask pairs by geographical tile
tile_records = defaultdict(list)

image_files = sorted(DATA_DIR.rglob("*_image.npy"))

for image_path in image_files:

    mask_path = image_path.with_name(
        image_path.name.replace("_image.npy", "_mask.npy")
    )

    if not mask_path.exists():
        continue

    filename = image_path.stem.removesuffix("_image")

    if "_mosaic_" not in filename:
        raise ValueError(f"Unexpected filename: {image_path.name}")

    tile_id = filename.split("_mosaic_", 1)[1]

    tile_records[tile_id].append({
        "image": str(image_path.relative_to(DATA_DIR)),
        "mask": str(mask_path.relative_to(DATA_DIR)),
        "tile_id": tile_id
    })

# Shuffle geographical tile IDs
tile_ids = sorted(tile_records.keys())
random.shuffle(tile_ids)

total_tiles = len(tile_ids)

if total_tiles < 3:
    raise RuntimeError("Not enough geographical tiles for splitting.")

# Split tiles: approximately 70% train, 15% validation, 15% test
test_count = round(total_tiles * 0.15)
val_count = round(total_tiles * 0.15)

test_tiles = set(tile_ids[:test_count])
val_tiles = set(tile_ids[test_count:test_count + val_count])
train_tiles = set(tile_ids[test_count + val_count:])

# Assign every monthly image of a tile to the same split
train_records = []
val_records = []
test_records = []

for tile_id, records in tile_records.items():

    if tile_id in train_tiles:
        train_records.extend(records)

    elif tile_id in val_tiles:
        val_records.extend(records)

    elif tile_id in test_tiles:
        test_records.extend(records)


def save_manifest(name, records):

    output_path = SPLIT_DIR / f"{name}.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["image", "mask", "tile_id"]
        )

        writer.writeheader()
        writer.writerows(records)

    return output_path


save_manifest("train", train_records)
save_manifest("val", val_records)
save_manifest("test", test_records)

# Verify tile separation
assert train_tiles.isdisjoint(val_tiles)
assert train_tiles.isdisjoint(test_tiles)
assert val_tiles.isdisjoint(test_tiles)

# Verify every pair is assigned exactly once
assert (
    len(train_records) + len(val_records) + len(test_records)
    == sum(len(v) for v in tile_records.values())
)

print("\n===== SPACENET DATASET SPLIT =====")

total = len(train_records) + len(val_records) + len(test_records)

for name, records in [
    ("Train", train_records),
    ("Validation", val_records),
    ("Test", test_records)
]:
    print(
        f"{name}: {len(records)} pairs "
        f"({len(records) / total * 100:.2f}%)"
    )

print("\nTotal pairs:", total)
print("Total geographical tiles:", total_tiles)

print("Train tiles:", len(train_tiles))
print("Validation tiles:", len(val_tiles))
print("Test tiles:", len(test_tiles))

print("\nTile leakage check: PASSED")
print("All pairs assigned exactly once: PASSED")
print("Manifests saved at:", SPLIT_DIR)