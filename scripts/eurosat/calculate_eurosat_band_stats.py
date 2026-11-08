from pathlib import Path
import json

import numpy as np
import pandas as pd
import rasterio


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "archive (7)"
    / "EuroSATallBands"
)

TRAIN_CSV = DATASET_DIR / "train.csv"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "metrics"
    / "eurosat_allbands_train_stats.json"
)


def main():

    print("\n========== EUROSAT BAND STATISTICS ==========\n")

    df = pd.read_csv(TRAIN_CSV)

    total_images = len(df)

    band_count = 13

    pixel_sums = np.zeros(band_count, dtype=np.float64)
    pixel_squared_sums = np.zeros(band_count, dtype=np.float64)
    pixel_counts = np.zeros(band_count, dtype=np.int64)

    print(f"Total training images: {total_images}")
    print(f"Processing {band_count} spectral bands...\n")

    for index, row in df.iterrows():

        image_path = DATASET_DIR / row["Filename"]

        with rasterio.open(image_path) as src:

            image = src.read(masked=True).astype(np.float64)

            for band_index in range(band_count):

                band = image[band_index]
                valid_pixels = band.compressed()

                pixel_sums[band_index] += valid_pixels.sum()
                pixel_squared_sums[band_index] += np.square(
                    valid_pixels
                ).sum()

                pixel_counts[band_index] += valid_pixels.size

        if (index + 1) % 1000 == 0:
            print(f"Processed: {index + 1}/{total_images}")

    # Calculate mean and standard deviation
    means = pixel_sums / pixel_counts

    variances = (
        pixel_squared_sums / pixel_counts
    ) - np.square(means)

    standard_deviations = np.sqrt(
        np.maximum(variances, 0)
    )

    results = {
        "total_training_images": total_images,
        "total_bands": band_count,
        "statistics_method": "Per-band global pixel statistics",
        "bands": {}
    }

    print("\n========== FINAL BAND STATISTICS ==========\n")

    for i in range(band_count):

        band_number = i + 1

        mean = float(means[i])
        std = float(standard_deviations[i])

        results["bands"][f"band_{band_number}"] = {
            "mean": mean,
            "std": std,
            "valid_pixels": int(pixel_counts[i])
        }

        print(
            f"Band {band_number:02d} | "
            f"Mean: {mean:.4f} | "
            f"Std: {std:.4f}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w") as file:
        json.dump(results, file, indent=4)

    print("\n========== COMPLETE ==========")
    print(f"Statistics saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()