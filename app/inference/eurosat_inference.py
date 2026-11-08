from pathlib import Path
from functools import lru_cache
import json

import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.io import MemoryFile
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "checkpoints"
    / "eurosat_allbands_resnet18_best.pth"
)

STATS_PATH = (
    PROJECT_ROOT / "experiments" / "metrics"
    / "eurosat_allbands_train_stats.json"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class EuroSATResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = models.resnet18(weights=None)

        self.model.conv1 = nn.Conv2d(
            13, 64, kernel_size=7,
            stride=1, padding=3, bias=False
        )

        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(
            self.model.fc.in_features, num_classes
        )

    def forward(self, x):
        return self.model(x)


@lru_cache(maxsize=1)
def load_eurosat_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not STATS_PATH.exists():
        raise FileNotFoundError(STATS_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model = EuroSATResNet18(num_classes=10)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    means = np.array(
        [stats["bands"][f"band_{i}"]["mean"] for i in range(1, 14)],
        dtype=np.float32
    )

    stds = np.array(
        [stats["bands"][f"band_{i}"]["std"] for i in range(1, 14)],
        dtype=np.float32
    )

    return model, means, stds


def predict_eurosat(image_bytes):
    model, means, stds = load_eurosat_model()

    with MemoryFile(image_bytes) as memfile:
        with memfile.open() as dataset:

            if dataset.count != 13:
                raise ValueError(
                    f"Expected 13 bands, received {dataset.count}. "
                    "Please upload a compatible 13-band GeoTIFF."
                )

            image = dataset.read().astype(np.float32)

    image = (image - means[:, None, None]) / stds[:, None, None]

    tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        values, indices = torch.topk(
            probabilities, k=3
        )

    top_predictions = [
        {
            "class": CLASS_NAMES[index.item()],
            "confidence": round(value.item() * 100, 2)
        }
        for value, index in zip(values, indices)
    ]

    return {
        "predicted_class": top_predictions[0]["class"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions
    }