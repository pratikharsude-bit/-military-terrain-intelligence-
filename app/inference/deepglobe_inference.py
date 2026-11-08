from pathlib import Path
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "checkpoints" /
    "deepglobe_unet_best.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

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

        d3 = F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = F.interpolate(d3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = F.interpolate(d2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        d0 = F.interpolate(d1, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        d0 = self.dec0(torch.cat([d0, x0], dim=1))

        d0 = F.interpolate(d0, size=original_size, mode="bilinear", align_corners=False)

        return self.classifier(d0)

    @staticmethod
    def decoder_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        input_size = x.shape[2:]

        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d3 = F.interpolate(
            x4, size=x3.shape[2:],
            mode="bilinear", align_corners=False
        )
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = F.interpolate(
            d3, size=x2.shape[2:],
            mode="bilinear", align_corners=False
        )
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = F.interpolate(
            d2, size=x1.shape[2:],
            mode="bilinear", align_corners=False
        )
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        d0 = F.interpolate(
            d1, size=x0.shape[2:],
            mode="bilinear", align_corners=False
        )
        d0 = self.dec0(torch.cat([d0, x0], dim=1))

        output = self.classifier(d0)

        return F.interpolate(
            output,
            size=input_size,
            mode="bilinear",
            align_corners=False
        )


@lru_cache(maxsize=1)
def load_deepglobe_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model = DeepGlobeUNet()

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    return model


def predict_deepglobe(image):

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")
    original_size = image.size

    input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    model = load_deepglobe_model()

    with torch.inference_mode():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1)

        prediction = F.interpolate(
            prediction.unsqueeze(1).float(),
            size=(original_size[1], original_size[0]),
            mode="nearest"
        ).squeeze().long()

    mask = prediction.cpu().numpy().astype(np.uint8)

    color_mask = np.zeros(
        (original_size[1], original_size[0], 3),
        dtype=np.uint8
    )

    percentages = {}

    for class_id, class_name in enumerate(CLASS_NAMES):

        class_pixels = mask == class_id

        percentages[class_name] = round(
            float(class_pixels.mean() * 100), 2
        )

        color_mask[class_pixels] = COLORS[class_id]

    overlay = (
        np.array(image).astype(np.float32) * 0.55
        + color_mask.astype(np.float32) * 0.45
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return {
        "mask": Image.fromarray(color_mask),
        "overlay": Image.fromarray(overlay),
        "percentages": percentages,
        "mask_array": mask
    }