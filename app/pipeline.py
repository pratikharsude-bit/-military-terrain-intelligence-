from pathlib import Path
from PIL import Image
import io
import numpy as np
import rasterio
from rasterio.io import MemoryFile

from app.inference.skyview_inference import predict_skyview
from app.inference.deepglobe_inference import predict_deepglobe
from app.inference.eurosat_inference import predict_eurosat
from app.inference.spacenet_inference import predict_spacenet


def get_pipeline_status():
    root = Path(__file__).resolve().parents[1]

    paths = {
        "skyview": root / "models/checkpoints/skyview_resnet18_best.pth",
        "deepglobe": root / "models/checkpoints/deepglobe_unet_best.pth",
        "eurosat": root / "models/checkpoints/eurosat_allbands_resnet18_best.pth",
        "spacenet": root / "models/checkpoints/spacenet_unet_best.pth",
    }

    return {
        "models": {
            name: {
                "available": path.exists(),
                "path": str(path)
            }
            for name, path in paths.items()
        },
        "all_models_available": all(
            path.exists() for path in paths.values()
        )
    }


def analyze_image(uploaded_file, run_skyview=True):

    filename = uploaded_file.name.lower()
    image_bytes = uploaded_file.getvalue()

    results = {
        "filename": uploaded_file.name,
        "input_type": None,
        "preview": None,
        "skyview": None,
        "deepglobe": None,
        "eurosat": None,
        "spacenet": None,
        "warnings": []
    }

    if filename.endswith((".tif", ".tiff")):

        with MemoryFile(image_bytes) as memfile:
            with memfile.open() as dataset:

                band_count = dataset.count

                if band_count == 13:

                    results["input_type"] = "13-band multispectral TIFF"
                    results["eurosat"] = predict_eurosat(image_bytes)

                    preview_array = dataset.read([4, 3, 2])
                    preview_array = np.moveaxis(
                        preview_array, 0, -1
                    ).astype(np.float32)

                    preview_array = np.nan_to_num(preview_array)

                    for i in range(3):
                        band = preview_array[:, :, i]
                        low, high = np.percentile(band, [2, 98])

                        preview_array[:, :, i] = np.clip(
                            (band - low) / (high - low + 1e-8),
                            0,
                            1
                        )

                    results["preview"] = Image.fromarray(
                        (preview_array * 255).astype(np.uint8)
                    )

                elif band_count >= 3:

                    results["input_type"] = (
                        f"{band_count}-band RGB-compatible TIFF"
                    )

                    rgb = dataset.read([1, 2, 3])
                    rgb = np.moveaxis(rgb, 0, -1).astype(np.float32)
                    rgb = np.nan_to_num(rgb)

                    for i in range(3):
                        band = rgb[:, :, i]
                        low, high = np.percentile(band, [2, 98])

                        rgb[:, :, i] = np.clip(
                            (band - low) / (high - low + 1e-8),
                            0,
                            1
                        )

                    preview = Image.fromarray(
                        (rgb * 255).astype(np.uint8)
                    )

                    results["preview"] = preview

                    if run_skyview:
                        results["skyview"] = predict_skyview(preview)

                    results["deepglobe"] = predict_deepglobe(preview)
                    results["spacenet"] = predict_spacenet(preview)

                    results["warnings"].append(
                        "SpaceNet building output is experimental."
                    )

                else:
                    raise ValueError(
                        f"Unsupported TIFF with {band_count} band(s)."
                    )

    else:

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        results["input_type"] = "RGB image"
        results["preview"] = image

        if run_skyview:
            results["skyview"] = predict_skyview(image)

        results["deepglobe"] = predict_deepglobe(image)
        results["spacenet"] = predict_spacenet(image)

        results["warnings"].append(
            "SpaceNet building output is experimental."
        )

    return results