
from pathlib import Path

import numpy as np
import rasterio


# Project root
ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    ROOT / "data" / "raw" / "archive (7)" / "EuroSATallBands"
)


def main():
    print("=" * 60)
    print("EUROSAT ALL BANDS - TIFF INSPECTION")
    print("=" * 60)

    if not DATASET_DIR.exists():
        print(f"Dataset folder not found: {DATASET_DIR}")
        return

    # Find one sample TIFF
    tif_files = sorted(DATASET_DIR.rglob("*.tif"))

    if not tif_files:
        tif_files = sorted(DATASET_DIR.rglob("*.tiff"))

    if not tif_files:
        print("No TIFF files found!")
        return

    sample_path = tif_files[0]

    print(f"\nSample image: {sample_path.name}")
    print(f"Total TIFF files found: {len(tif_files)}")

    try:
        with rasterio.open(sample_path) as src:

            print("\n--- IMAGE INFORMATION ---")
            print(f"Width: {src.width}")
            print(f"Height: {src.height}")
            print(f"Number of bands: {src.count}")
            print(f"Data types: {src.dtypes}")
            print(f"CRS: {src.crs}")
            print(f"NoData value: {src.nodata}")
            print(f"Bounds: {src.bounds}")

            print("\n--- BAND INFORMATION ---")

            for band_index in range(1, src.count + 1):
                band = src.read(band_index, masked=True)
                values = band.compressed()

                print(f"\nBand {band_index}")

                if values.size == 0:
                    print("No valid pixel values found.")
                    continue

                print(f"Minimum: {values.min()}")
                print(f"Maximum: {values.max()}")
                print(f"Mean: {values.mean():.4f}")
                print(f"Data type: {values.dtype}")
                print(f"Valid pixels: {values.size}")
                print(f"Masked pixels: {band.mask.sum()}")

            print("\n--- INSPECTION COMPLETE ---")

    except Exception as error:
        print("\nTIFF READ ERROR:")
        print(repr(error))


if __name__ == "__main__":
    main()
