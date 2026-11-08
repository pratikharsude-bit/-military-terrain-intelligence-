import json
from pathlib import Path
from rasterio.warp import transform_geom
from shapely.geometry import shape, box

import rasterio

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/raw/archive (4)/SN7_buildings_train/train"

image_path = next(DATA.rglob("*.tif"))
base = image_path.stem

geojson_path = image_path.parent.parent / "labels" / f"{base}_Buildings.geojson"
udm_path = image_path.parent.parent / "labels" / f"{base}_UDM.geojson"

print("\n===== IMAGE DETAILS =====")

with rasterio.open(image_path) as src:
    print("Image:", image_path.name)
    print("Dimensions:", src.width, "x", src.height)
    print("Bands:", src.count)
    print("Data types:", src.dtypes)
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Transform:", src.transform)

print("\n===== BUILDING GEOJSON =====")

with open(geojson_path, encoding="utf-8") as f:
    buildings = json.load(f)

print("File:", geojson_path.name)
print("Type:", buildings.get("type"))
print("Features:", len(buildings.get("features", [])))

if buildings.get("features"):
    feature = buildings["features"][0]
    print("Geometry type:", feature["geometry"]["type"])
    print("Properties:", feature.get("properties"))

print("\n===== UDM GEOJSON =====")

with open(udm_path, encoding="utf-8") as f:
    udm = json.load(f)

print("Features:", len(udm.get("features", [])))

print("\nInspection completed.")

print("\n===== GEOJSON COORDINATE CHECK =====")

print("GeoJSON CRS:", buildings.get("crs"))

all_x = []
all_y = []

for feature in buildings.get("features", []):
    geometry = feature.get("geometry")

    if not geometry:
        continue

    def collect_coordinates(coords):
        if isinstance(coords[0], (int, float)):
            all_x.append(coords[0])
            all_y.append(coords[1])
        else:
            for item in coords:
                collect_coordinates(item)

    collect_coordinates(geometry["coordinates"])

if all_x and all_y:
    print("Polygon X range:", min(all_x), "to", max(all_x))
    print("Polygon Y range:", min(all_y), "to", max(all_y))

    with rasterio.open(image_path) as src:
        bounds = src.bounds

        print(
            "Inside image bounds:",
            min(all_x) >= bounds.left
            and max(all_x) <= bounds.right
            and min(all_y) >= bounds.bottom
            and max(all_y) <= bounds.top
        )

        print("\n===== CRS TRANSFORMATION CHECK =====")

with rasterio.open(image_path) as src:
    image_bounds = box(
        src.bounds.left,
        src.bounds.bottom,
        src.bounds.right,
        src.bounds.top
    )

    transformed_count = 0
    intersecting_count = 0

    for feature in buildings.get("features", []):
        geometry = feature.get("geometry")

        if not geometry:
            continue

        projected = transform_geom(
            "OGC:CRS84",
            src.crs,
            geometry
        )

        projected_shape = shape(projected)
        transformed_count += 1

        if projected_shape.intersects(image_bounds):
            intersecting_count += 1

    print("Transformed polygons:", transformed_count)
    print("Polygons intersecting image:", intersecting_count)
    print(
        "Intersection percentage:",
        round(intersecting_count / max(transformed_count, 1) * 100, 2),
        "%"
    )