
import csv
import json
import random
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from shapely.geometry import shape, box, mapping
from shapely.ops import unary_union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# 1. PATHS AND CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = (
    ROOT / "data/raw/archive (4)/SN7_buildings_train/train"
)

OUTPUT_DIR = ROOT / "data/processed/SpaceNet7_UDM"

REPORT_DIR = (
    ROOT / "experiments/metrics/spacenet_udm_audit"
)

MASK_DIR = OUTPUT_DIR / "ignore_masks"
VISUAL_DIR = REPORT_DIR / "visual_samples"

MASK_DIR.mkdir(parents=True, exist_ok=True)
VISUAL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
VISUAL_SAMPLE_COUNT = 5


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def get_tile_id(path):
    """Extract geographical tile folder name."""
    relative = path.relative_to(TRAIN_DIR)
    return relative.parts[0]


def get_matching_image(udm_path):
    """Find exact corresponding TIFF image."""
    image_name = udm_path.name.replace(
        "_UDM.geojson", ".tif"
    )

    return udm_path.parent.parent / "images" / image_name


def repair_geometry(geometry):
    """Validate and attempt to repair invalid geometry."""

    if geometry is None or geometry.is_empty:
        return None, "EMPTY"

    if geometry.is_valid:
        return geometry, "VALID"

    repaired = geometry.buffer(0)

    if repaired.is_empty or not repaired.is_valid:
        return None, "REPAIR_FAILED"

    return repaired, "REPAIRED"


def polygon_only(geometry):
    """Keep only polygonal components."""

    if geometry is None or geometry.is_empty:
        return None

    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        return geometry

    if geometry.geom_type == "GeometryCollection":
        polygons = [
            item for item in geometry.geoms
            if item.geom_type in ("Polygon", "MultiPolygon")
            and not item.is_empty
        ]

        if polygons:
            return unary_union(polygons)

    return None


# =========================================================
# 3. FIND UDM FILES
# =========================================================

udm_files = sorted(TRAIN_DIR.rglob("*_UDM.geojson"))

if not udm_files:
    raise FileNotFoundError(
        f"No UDM files found in {TRAIN_DIR}"
    )

print("\n========== SPACENET-7 UDM AUDIT ==========")
print("UDM files found:", len(udm_files))


# =========================================================
# 4. COUNTERS AND REPORT STORAGE
# =========================================================

counts = {
    "matched_images": 0,
    "missing_images": 0,
    "empty_udm_files": 0,
    "non_empty_udm_files": 0,
    "total_features": 0,
    "valid_features": 0,
    "invalid_features": 0,
    "repaired_features": 0,
    "failed_geometries": 0,
    "overlapping_polygons": 0,
    "outside_polygons": 0,
    "processing_errors": 0,
}

file_reports = []
visual_candidates = []
errors = []

total_ignore_pixels = 0
total_image_pixels = 0


# =========================================================
# 5. PROCESS UDM FILES
# =========================================================

for index, udm_path in enumerate(udm_files, start=1):

    relative_path = udm_path.relative_to(TRAIN_DIR)
    image_path = get_matching_image(udm_path)
    tile_id = get_tile_id(udm_path)

    row = {
        "udm_file": str(relative_path),
        "image_file": "",
        "status": "",
        "feature_count": 0,
        "valid_features": 0,
        "invalid_features": 0,
        "repaired_features": 0,
        "failed_geometries": 0,
        "overlapping_polygons": 0,
        "outside_polygons": 0,
        "ignore_pixels": 0,
        "image_pixels": 0,
        "coverage_percent": 0.0,
        "ignore_mask_path": "",
    }

    try:

        # ---------------------------------------------
        # A. IMAGE MATCHING
        # ---------------------------------------------

        if not image_path.exists():
            counts["missing_images"] += 1
            row["status"] = "MISSING_IMAGE"
            file_reports.append(row)
            continue

        counts["matched_images"] += 1

        row["image_file"] = str(
            image_path.relative_to(TRAIN_DIR)
        )

        # ---------------------------------------------
        # B. READ GEOJSON
        # ---------------------------------------------

        with open(udm_path, "r", encoding="utf-8") as f:
            geojson = json.load(f)

        features = geojson.get("features", [])
        feature_count = len(features)

        row["feature_count"] = feature_count
        counts["total_features"] += feature_count

        if feature_count == 0:
            counts["empty_udm_files"] += 1
        else:
            counts["non_empty_udm_files"] += 1

        # ---------------------------------------------
        # C. OPEN IMAGE AND READ METADATA
        # ---------------------------------------------

        with rasterio.open(image_path) as src:

            height = src.height
            width = src.width

            image_bounds = box(*src.bounds)

            transformed_geometries = []

            # -----------------------------------------
            # D. GEOMETRY VALIDITY AND CRS TRANSFORMATION
            # -----------------------------------------

            for feature in features:

                geom_json = feature.get("geometry")

                if not geom_json:
                    counts["failed_geometries"] += 1
                    row["failed_geometries"] += 1
                    continue

                try:

                    geometry = shape(geom_json)

                    geometry, validity = repair_geometry(
                        geometry
                    )

                    if validity == "EMPTY":
                        counts["failed_geometries"] += 1
                        row["failed_geometries"] += 1
                        continue

                    if validity == "VALID":
                        counts["valid_features"] += 1
                        row["valid_features"] += 1

                    elif validity == "REPAIRED":
                        counts["invalid_features"] += 1
                        counts["repaired_features"] += 1

                        row["invalid_features"] += 1
                        row["repaired_features"] += 1

                    elif validity == "REPAIR_FAILED":
                        counts["invalid_features"] += 1
                        counts["failed_geometries"] += 1

                        row["invalid_features"] += 1
                        row["failed_geometries"] += 1
                        continue

                    geometry = polygon_only(geometry)

                    if geometry is None:
                        counts["failed_geometries"] += 1
                        row["failed_geometries"] += 1
                        continue

                    projected_json = transform_geom(
                        "OGC:CRS84",
                        src.crs,
                        mapping(geometry)
                    )

                    projected = shape(projected_json)

                    projected, projected_status = repair_geometry(
                        projected
                    )

                    projected = polygon_only(projected)

                    if projected is None:
                        counts["failed_geometries"] += 1
                        row["failed_geometries"] += 1
                        continue

                    # ---------------------------------
                    # E. IMAGE OVERLAP CHECK
                    # ---------------------------------

                    intersection = projected.intersection(
                        image_bounds
                    )

                    if (
                        intersection.is_empty
                        or intersection.area <= 0
                    ):
                        counts["outside_polygons"] += 1
                        row["outside_polygons"] += 1
                        continue

                    counts["overlapping_polygons"] += 1
                    row["overlapping_polygons"] += 1

                    transformed_geometries.append(
                        intersection
                    )

                except Exception as exc:
                    counts["failed_geometries"] += 1
                    row["failed_geometries"] += 1

                    errors.append({
                        "file": str(udm_path),
                        "error": str(exc),
                    })

            # -----------------------------------------
            # F. GENERATE FULL-RESOLUTION IGNORE MASK
            # -----------------------------------------

            ignore_mask = np.zeros(
                (height, width),
                dtype=np.uint8
            )

            if transformed_geometries:

                merged = unary_union(
                    transformed_geometries
                )

                ignore_mask = rasterize(
                    [(merged, 1)],
                    out_shape=(height, width),
                    transform=src.transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=False
                )

            ignore_pixels = int(ignore_mask.sum())
            image_pixels = height * width

            coverage = (
                ignore_pixels / image_pixels * 100
                if image_pixels else 0.0
            )

            total_ignore_pixels += ignore_pixels
            total_image_pixels += image_pixels

            row["ignore_pixels"] = ignore_pixels
            row["image_pixels"] = image_pixels
            row["coverage_percent"] = round(
                coverage, 4
            )

            # -----------------------------------------
            # G. SAVE IGNORE MASK
            # -----------------------------------------

            mask_folder = MASK_DIR / tile_id
            mask_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            mask_path = (
                mask_folder
                / f"{image_path.stem}_ignore.npy"
            )

            np.save(mask_path, ignore_mask)

            row["ignore_mask_path"] = str(
                mask_path.relative_to(ROOT)
            )

            row["status"] = "PROCESSED"

            # -----------------------------------------
            # H. COLLECT VISUAL SAMPLES
            # -----------------------------------------

            if ignore_pixels > 0:

                visual_candidates.append({
                    "image_path": image_path,
                    "mask": ignore_mask.copy(),
                    "tile_id": tile_id,
                    "coverage": coverage,
                })

        file_reports.append(row)

    except Exception as exc:

        counts["processing_errors"] += 1
        row["status"] = "PROCESSING_ERROR"

        errors.append({
            "file": str(udm_path),
            "error": str(exc),
        })

        file_reports.append(row)

    if index % 100 == 0:
        print(
            f"Processed {index}/{len(udm_files)} files"
        )


# =========================================================
# 6. SAVE VISUAL SAMPLES
# =========================================================

random.seed(RANDOM_SEED)
random.shuffle(visual_candidates)

selected_samples = visual_candidates[
    :VISUAL_SAMPLE_COUNT
]

for sample_number, sample in enumerate(
    selected_samples, start=1
):

    with rasterio.open(sample["image_path"]) as src:

        rgb = src.read([1, 2, 3])
        rgb = np.transpose(rgb, (1, 2, 0))

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    mask = sample["mask"]

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5)
    )

    axes[0].imshow(rgb)
    axes[0].set_title("Original Image")

    axes[1].imshow(
        mask, cmap="gray", vmin=0, vmax=1
    )
    axes[1].set_title("UDM Ignore Mask")

    axes[2].imshow(rgb)
    axes[2].imshow(
        np.ma.masked_where(mask == 0, mask),
        cmap="autumn",
        alpha=0.7,
        vmin=0,
        vmax=1
    )
    axes[2].set_title("UDM Overlay")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Tile: {sample['tile_id']} | "
        f"Coverage: {sample['coverage']:.2f}%"
    )

    plt.tight_layout()

    plt.savefig(
        VISUAL_DIR / f"udm_sample_{sample_number}.png",
        dpi=150
    )

    plt.close(fig)


# =========================================================
# 7. SAVE CSV REPORT
# =========================================================

csv_path = REPORT_DIR / "udm_file_report.csv"

if file_reports:

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(file_reports[0].keys())
        )

        writer.writeheader()
        writer.writerows(file_reports)


# =========================================================
# 8. SAVE JSON SUMMARY
# =========================================================

overall_coverage = (
    total_ignore_pixels / total_image_pixels * 100
    if total_image_pixels else 0.0
)

summary = {
    "total_udm_files": len(udm_files),
    **counts,
    "total_ignore_pixels": total_ignore_pixels,
    "total_image_pixels": total_image_pixels,
    "overall_ignore_coverage_percent": round(
        overall_coverage, 4
    ),
    "visual_samples_saved": len(selected_samples),
    "errors": errors,
}

json_path = REPORT_DIR / "udm_audit_summary.json"

with open(
    json_path, "w", encoding="utf-8"
) as f:

    json.dump(summary, f, indent=4)


# =========================================================
# 9. FINAL REPORT
# =========================================================

print("\n========== FINAL UDM AUDIT ==========")

for key, value in summary.items():

    if key != "errors":
        print(f"{key}: {value}")

print("\nCSV report:", csv_path)
print("JSON report:", json_path)
print("Ignore masks:", MASK_DIR)
print("Visual samples:", VISUAL_DIR)

if errors:
    print("\nWARNING: Some geometry/file errors occurred.")
    print("Check udm_audit_summary.json")
else:
    print("\nNo processing errors recorded.")

print("\n========== AUDIT FINISHED ==========")
