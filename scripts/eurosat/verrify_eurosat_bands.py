from pathlib import Path
import pandas as pd
import rasterio

# Project and dataset paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "archive (7)" / "EuroSATallBands"

CSV_FILES = {
    "train": DATASET_DIR / "train.csv",
    "validation": DATASET_DIR / "validation.csv",
    "test": DATASET_DIR / "test.csv",
}

CLASS_MAP = {
    0: "AnnualCrop",
    1: "Forest",
    2: "HerbaceousVegetation",
    3: "Highway",
    4: "Industrial",
    5: "Pasture",
    6: "PermanentCrop",
    7: "Residential",
    8: "River",
    9: "SeaLake",
}


def main():
    print("\n========== EUROSAT ALL BANDS VERIFICATION ==========\n")

    if not DATASET_DIR.exists():
        print(f"ERROR: Dataset folder not found:\n{DATASET_DIR}")
        return

    datasets = {}
    all_paths = {}

    # 1. Check CSV files and their contents
    for split, csv_path in CSV_FILES.items():
        print(f"\n--- {split.upper()} ---")

        if not csv_path.exists():
            print(f"ERROR: Missing CSV: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        datasets[split] = df

        required_columns = {"Filename", "Label", "ClassName"}

        if not required_columns.issubset(df.columns):
            print("ERROR: Required columns are missing.")
            return

        print(f"Total entries: {len(df)}")
        print("\nClass distribution:")
        print(df["ClassName"].value_counts().sort_index())

        # 2. Check labels against class names
        invalid_labels = df[
            df.apply(
                lambda row: CLASS_MAP.get(row["Label"]) != row["ClassName"],
                axis=1,
            )
        ]

        print(f"\nIncorrect labels: {len(invalid_labels)}")

        # 3. Check duplicate paths inside the CSV
        duplicate_count = df["Filename"].duplicated().sum()
        print(f"Duplicate paths: {duplicate_count}")

        # 4. Check whether files exist
        paths = df["Filename"].astype(str).tolist()
        all_paths[split] = set(paths)

        missing = [
            filename
            for filename in paths
            if not (DATASET_DIR / filename).is_file()
        ]

        print(f"Missing TIFF files: {len(missing)}")

        if missing:
            print("First missing files:", missing[:5])
        
        # Check all TIFF files against CSV entries
    all_csv_paths = set().union(*all_paths.values())

    all_tiff_files = {
        path.relative_to(DATASET_DIR).as_posix()
        for path in DATASET_DIR.rglob("*.tif")
    }

    unlisted_files = all_tiff_files - all_csv_paths

    print("\n========== UNLISTED TIFF FILES ==========")
    print(f"Total TIFF files found: {len(all_tiff_files)}")
    print(f"Total CSV entries: {len(all_csv_paths)}")
    print(f"Unlisted TIFF files: {len(unlisted_files)}")

    for filename in sorted(unlisted_files):
        print(filename)
    # 5. Check overlap between splits
    
    print("\n========== SPLIT OVERLAP CHECK ==========")

    split_names = list(all_paths.keys())

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            first = split_names[i]
            second = split_names[j]

            overlap = all_paths[first] & all_paths[second]

            print(f"{first} vs {second}: {len(overlap)} overlapping paths")

    # 6. Inspect a few TIFF files using rasterio
    print("\n========== TIFF READ TEST ==========")

    for split, df in datasets.items():
        print(f"\n{split.upper()} samples:")

        for filename in df["Filename"].head(3):
            image_path = DATASET_DIR / filename

            if not image_path.is_file():
                continue

            try:
                with rasterio.open(image_path) as src:
                    image = src.read()

                    print(
                        f"{filename} | "
                        f"Bands: {src.count} | "
                        f"Size: {src.width}x{src.height} | "
                        f"Dtype: {src.dtypes[0]} | "
                        f"Shape: {image.shape}"
                    )

            except Exception as error:
                print(f"READ ERROR: {filename} | {error}")

    print("\n========== VERIFICATION COMPLETE ==========")


if __name__ == "__main__":
    main()