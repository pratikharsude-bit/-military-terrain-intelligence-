
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "spacenet_unet_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SpaceNetUNet(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet18(weights=None)

        self.encoder1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )

        self.pool = backbone.maxpool
        self.encoder2 = backbone.layer1
        self.encoder3 = backbone.layer2
        self.encoder4 = backbone.layer3
        self.encoder5 = backbone.layer4

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ConvBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.final_up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        d4 = self.dec4(torch.cat([self.up4(e5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.final_up(d1)
        return self.output(out)


def load_spacenet_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"SpaceNet checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = SpaceNetUNet().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


def predict_spacenet(image):
    """
    Accepts a PIL RGB image.
    Returns building mask, overlay, building percentage and probability map.
    """

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    original = image.convert("RGB")

    resized = original.resize((256, 256), Image.Resampling.BILINEAR)

    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    image_array = (image_array - MEAN) / STD

    tensor = torch.from_numpy(
        image_array.transpose(2, 0, 1)
    ).unsqueeze(0).float().to(DEVICE)

    model = load_spacenet_model()

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

    binary_mask = (probabilities >= 0.5).astype(np.uint8)

    building_percentage = float(binary_mask.mean() * 100)

    mask_image = Image.fromarray(binary_mask * 255).resize(
        original.size, Image.Resampling.NEAREST
    )

    mask_array = np.asarray(mask_image) > 0
    original_array = np.asarray(original).copy()

    overlay_array = original_array.copy()
    overlay_array[mask_array] = (
        0.4 * original_array[mask_array] +
        0.6 * np.array([255, 60, 40])
    ).astype(np.uint8)

    overlay = Image.fromarray(overlay_array)

    return {
        "mask": mask_image,
        "overlay": overlay,
        "building_percentage": building_percentage,
        "probability_map": probabilities
    }
