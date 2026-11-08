
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "archive (5)"

print("=" * 55)
print("DEEPGLOBE DATASET INSPECTION")
print("=" * 55)
print(f"Dataset path: {DATA_DIR}")

if not DATA_DIR.exists():
    print("\nERROR: Dataset folder not found.")
    print("Check the actual folder name inside data/raw.")
    raise SystemExit(1)

# Show first two directory levels
print("\nFolder structure:")
for path in sorted(DATA_DIR.iterdir()):
    print(f"\n[{path.name}]")
    if path.is_dir():
        for child in sorted(path.iterdir())[:8]:
            if child.is_dir():
                count = sum(1 for p in child.iterdir() if p.is_file())
                print(f"  {child.name}/ ({count} direct files)")
            else:
                print(f"  {child.name}")

# Inspect CSV files and their column names
print("\nCSV files:")
for csv_path in DATA_DIR.rglob("*.csv"):
    print(f"\nFile: {csv_path.relative_to(DATA_DIR)}")
    try:
        with csv_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.reader(file)
            for row in list(reader)[:6]:
                print("  ", row)
    except Exception as exc:
        print(f"  Could not read: {exc}")

# Count common image formats
extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
print("\nImage file counts:")
for folder in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
    count = sum(
        1 for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )
    print(f"{folder.name}: {count}")
