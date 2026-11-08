
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from eurosat_model import EuroSATResNet18


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models/checkpoints/eurosat_allbands_resnet18_best.pth"
STATS_PATH = PROJECT_ROOT / "experiments/metrics/eurosat_allbands_train_stats.json"

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


# --------------------------------------------------
# LOAD TRAINING STATISTICS
# --------------------------------------------------

def load_band_statistics(stats_path):
    with open(stats_path, "r") as file:
        stats = json.load(file)

    # Supports the common structure:
    # {"mean": [...], "std": [...]}
    if "mean" in stats and "std" in stats:
        means = stats["mean"]
        stds = stats["std"]

    # Supports the structure:
    # {"bands": {"Band 01": {"mean": ..., "std": ...}, ...}}
    elif "bands" in stats:
        band_data = stats["bands"]
        means = [band_data[k]["mean"] for k in band_data]
        stds = [band_data[k]["std"] for k in band_data]

    else:
        raise ValueError(
            "Unknown statistics JSON structure. "
            "Please inspect eurosat_allbands_train_stats.json."
        )

    means = np.asarray(means, dtype=np.float32).reshape(13, 1, 1)
    stds = np.asarray(stds, dtype=np.float32).reshape(13, 1, 1)

    if len(means) != 13 or len(stds) != 13:
        raise ValueError("Expected statistics for exactly 13 bands.")

    stds[stds == 0] = 1.0

    return means, stds


# --------------------------------------------------
# LOAD TIFF IMAGE
# --------------------------------------------------

def load_tiff_image(image_path, means, stds):
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)

    if image.shape[0] != 13:
        raise ValueError(
            f"Expected 13 bands, but found {image.shape[0]}."
        )

    image = (image - means) / stds

    image = torch.from_numpy(image).unsqueeze(0)

    return image


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

def load_model(model_path, device):
    model = EuroSATResNet18(num_classes=10)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    # Supports either a raw state_dict or checkpoint dictionary
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict(image_path):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nUsing device: {device}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not STATS_PATH.exists():
        raise FileNotFoundError(f"Statistics not found: {STATS_PATH}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    means, stds = load_band_statistics(STATS_PATH)

    image = load_tiff_image(image_path, means, stds)
    image = image.to(device)

    model = load_model(MODEL_PATH, device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)

    confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_class = CLASS_NAMES[predicted_index.item()]
    confidence_value = confidence.item() * 100

    print("\n" + "=" * 45)
    print("       EUROSAT ALL-BANDS PREDICTION")
    print("=" * 45)
    print(f"Image: {image_path.name}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence_value:.2f}%")
    print("=" * 45)

    print("\nTop 3 Predictions:")

    top_values, top_indices = torch.topk(probabilities, 3, dim=1)

    for value, index in zip(
        top_values[0].tolist(),
        top_indices[0].tolist()
    ):
        print(f"{CLASS_NAMES[index]:25s} {value * 100:.2f}%")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict EuroSAT 13-band satellite imagery."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to a 13-band TIFF image."
    )

    args = parser.parse_args()

    predict(Path(args.image))
