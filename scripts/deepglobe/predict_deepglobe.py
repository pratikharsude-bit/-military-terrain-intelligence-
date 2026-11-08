from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "models" / "checkpoints" / "deepglobe_unet_best.pth"
OUTPUT_DIR = ROOT / "outputs" / "predictions"

CLASS_NAMES = [
    "urban_land",
    "agriculture_land",
    "rangeland",
    "forest_land",
    "water",
    "barren_land",
]

COLORS = [
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 255),
]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DeepGlobeUNet(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet18(weights=None)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.dec3 = ConvBlock(768, 256)
        self.dec2 = ConvBlock(384, 128)
        self.dec1 = ConvBlock(192, 64)
        self.dec0 = ConvBlock(128, 64)

        self.classifier = nn.Conv2d(64, 6, 1)

    def forward(self, x):
        original_size = x.shape[-2:]

        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d3 = F.interpolate(
            x4, size=x3.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = F.interpolate(
            d3, size=x2.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = F.interpolate(
            d2, size=x1.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        d0 = F.interpolate(
            d1, size=x0.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d0 = self.dec0(torch.cat([d0, x0], dim=1))

        d0 = F.interpolate(
            d0, size=original_size,
            mode="bilinear", align_corners=False
        )

        return self.classifier(d0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Input satellite image path")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()

    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True
    )

    model = DeepGlobeUNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    original = Image.open(image_path).convert("RGB")
    width, height = original.size

    resized = original.resize(
        (256, 256),
        Image.Resampling.BILINEAR
    )

    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, MEAN, STD)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    pred_image = Image.fromarray(prediction)
    pred_image = pred_image.resize(
        (width, height),
        Image.Resampling.NEAREST
    )

    prediction = np.asarray(pred_image)

    color_map = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in enumerate(COLORS):
        color_map[prediction == class_id] = color

    overlay = (
        0.55 * np.asarray(original, dtype=np.float32)
        + 0.45 * color_map.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    mask_path = OUTPUT_DIR / f"{stem}_segmentation.png"
    overlay_path = OUTPUT_DIR / f"{stem}_overlay.png"
    report_path = OUTPUT_DIR / f"{stem}_percentages.json"

    Image.fromarray(color_map).save(mask_path)
    Image.fromarray(overlay).save(overlay_path)

    total = prediction.size

    percentages = {
        CLASS_NAMES[i]: round(
            float(np.count_nonzero(prediction == i) / total * 100),
            2
        )
        for i in range(len(CLASS_NAMES))
    }

    report = {
        "input_image": str(image_path),
        "image_width": width,
        "image_height": height,
        "model": "DeepGlobe U-Net with ResNet18 encoder",
        "best_validation_mIoU": checkpoint.get("best_val_mean_iou"),
        "class_percentages": percentages,
        "warning": "Estimated segmentation; validation metric is not test accuracy."
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nPrediction complete!")
    print(f"Device: {device}")
    print(f"Segmentation: {mask_path}")
    print(f"Overlay: {overlay_path}")
    print(f"Percentages: {report_path}")

    print("\nEstimated class percentages:")
    for name, value in percentages.items():
        print(f"  {name}: {value}%")


if __name__ == "__main__":
    print("DeepGlobe prediction script started!")
    main()