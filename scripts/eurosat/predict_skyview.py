
"""
Military Terrain Analysis
Central Model Pipeline

This module will connect:
1. SkyView - aerial landscape classification
2. DeepGlobe - land-cover segmentation
3. EuroSAT All-Bands - multispectral classification
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATHS = {
    "skyview": PROJECT_ROOT / "models/checkpoints/skyview_resnet18_best.pth",
    "deepglobe": PROJECT_ROOT / "models/checkpoints/deepglobe_unet_best.pth",
    "eurosat": PROJECT_ROOT / "models/checkpoints/eurosat_allbands_resnet18_best.pth",
}


def verify_models():
    """Check whether all trained model checkpoints exist."""

    status = {}

    for model_name, model_path in MODEL_PATHS.items():
        status[model_name] = {
            "available": model_path.exists(),
            "path": str(model_path)
        }

    return status


def get_pipeline_status():
    """Return the availability of all model pipelines."""

    status = verify_models()

    return {
        "project": "Military Terrain Analysis",
        "models": status,
        "all_models_available": all(
            model["available"] for model in status.values()
        )
    }


if __name__ == "__main__":
    import json

    print(json.dumps(get_pipeline_status(), indent=4))
