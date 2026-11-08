from pathlib import Path
from functools import lru_cache

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "checkpoints" /
    "skyview_resnet18_best.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@lru_cache(maxsize=1)
def load_skyview_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"SkyView checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


def predict_skyview(image):

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")

    model, class_names = load_skyview_model()

    input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

        top_values, top_indices = torch.topk(
            probabilities,
            k=min(3, len(class_names))
        )

    top_predictions = []

    for value, index in zip(top_values, top_indices):

        top_predictions.append({
            "class": class_names[index.item()],
            "confidence": round(value.item() * 100, 2)
        })

    return {
        "predicted_class": top_predictions[0]["class"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions
    }