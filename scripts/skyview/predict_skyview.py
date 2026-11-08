
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = ROOT / "models" / "checkpoints" / "skyview_resnet18_best.pth"


def main():
    parser = argparse.ArgumentParser(
        description="Predict the landscape class of an aerial image."
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to an image, e.g. data/raw/example.jpg"
    )
    args = parser.parse_args()

    image_path = args.image or input("Enter image path: ").strip().strip('"')
    image_path = Path(image_path).expanduser()

    if not image_path.is_absolute():
        image_path = ROOT / image_path

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {CHECKPOINT_PATH}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load our saved checkpoint (no pretrained download required).
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    with Image.open(image_path) as img:
        image = img.convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        top_values, top_indices = torch.topk(
            probabilities, k=min(3, len(class_names))
        )

    print("\n" + "=" * 45)
    print("SKYVIEW PREDICTION")
    print("=" * 45)
    print(f"Image: {image_path.name}")
    print(f"Predicted class: {class_names[top_indices[0].item()]}")
    print(f"Model confidence: {top_values[0].item() * 100:.2f}%")
    print("\nTop predictions:")

    for value, index in zip(top_values, top_indices):
        print(
            f"  {class_names[index.item()]:<18} "
            f"{value.item() * 100:.2f}%"
        )

    print("\nNote: Confidence is not a guarantee of correctness.")
    print("This model predicts one of the 15 SkyView scene classes.")


if __name__ == "__main__":
    main()
