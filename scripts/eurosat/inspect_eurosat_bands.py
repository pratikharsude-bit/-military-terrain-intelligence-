from pathlib import Path
import rasterio
import numpy as np

# Project and dataset paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "archive (7)"
    / "EuroSATallBands"
)

# Sample TIFF image
SAMPLE_IMAGE = (
    DATASET_DIR
    / "Forest"
    / "Forest_1001.tif"
)


def main():
    print("\n========== EUROSAT SPECTRAL BAND INSPECTION ==========\n")

    if not SAMPLE_IMAGE.exists():
        print(f"ERROR: Sample image not found:\n{SAMPLE_IMAGE}")
        return

    with rasterio.open(SAMPLE_IMAGE) as src:

        print(f"Image: {SAMPLE_IMAGE.name}")
        print(f"Number of bands: {src.count}")
        print(f"Width: {src.width}")
        print(f"Height: {src.height}")
        print(f"Data types: {src.dtypes}")

        print("\n========== TIFF METADATA ==========")
        print("Descriptions:", src.descriptions)
        print("Band units:", src.units)
        print("Nodata value:", src.nodata)
        print("Tags:", src.tags())

        print("\n========== INDIVIDUAL BAND STATISTICS ==========")

        for band_number in range(1, src.count + 1):

            band = src.read(band_number, masked=True)

            valid_pixels = band.compressed()

            print(f"\nBand {band_number}")

            if valid_pixels.size == 0:
                print("No valid pixels found.")
                continue

            print(f"Minimum: {valid_pixels.min()}")
            print(f"Maximum: {valid_pixels.max()}")
            print(f"Mean: {valid_pixels.mean():.4f}")
            print(f"Standard Deviation: {valid_pixels.std():.4f}")
            print(f"Valid Pixels: {valid_pixels.size}")
            print(f"Masked Pixels: {band.mask.sum()}")

    print("\n========== INSPECTION COMPLETE ==========")


if __name__ == "__main__":
    main()