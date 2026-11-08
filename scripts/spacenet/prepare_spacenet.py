import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/raw/archive (4)/SN7_buildings_train/train"
OUTPUT_DIR = ROOT / "data/processed/SpaceNet7"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_files = sorted(DATA_DIR.rglob("*.tif"))

processed = 0
skipped = 0

for image_path in image_files:

    base = image_path.stem

    geojson_path = (
        image_path.parent.parent
        / "labels"
        / f"{base}_Buildings.geojson"
    )

    if not geojson_path.exists():
        skipped += 1
        continue

    with rasterio.open(image_path) as src:

        image = src.read([1, 2, 3])
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

        with open(geojson_path, encoding="utf-8") as f:
            geojson = json.load(f)

        geometries = []

        for feature in geojson.get("features", []):

            geometry = feature.get("geometry")

            if not geometry:
                continue

            projected = transform_geom(
                "OGC:CRS84",
                crs,
                geometry
            )

            geometries.append((projected, 1))

        mask = rasterize(
            geometries,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        relative_path = image_path.relative_to(DATA_DIR)
        output_folder = OUTPUT_DIR / relative_path.parent.name

        output_folder.mkdir(parents=True, exist_ok=True)

        image_output = output_folder / f"{base}_image.npy"
        mask_output = output_folder / f"{base}_mask.npy"

        np.save(image_output, image)
        np.save(mask_output, mask)

        processed += 1

    if processed % 100 == 0:
        print(f"Processed: {processed}")

print("\n===== PREPARATION SUMMARY =====")
print("Processed image-mask pairs:", processed)
print("Skipped images:", skipped)
print("Output directory:", OUTPUT_DIR)