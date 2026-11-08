
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import functional as TF


# -------------------- CONFIG --------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw" / "archive (5)"
METADATA = DATA_ROOT / "metadata.csv"

CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
METRICS_DIR = ROOT / "experiments" / "metrics"

IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.0001
SEED = 42
VAL_FRACTION = 0.15
NUM_CLASSES = 6
IGNORE_INDEX = 255

CLASS_NAMES = [
    "urban_land",
    "agriculture_land",
    "rangeland",
    "forest_land",
    "water",
    "barren_land",
]

# DeepGlobe official RGB mask colours.
CLASS_COLORS = {
    "urban_land": (0, 255, 255),
    "agriculture_land": (255, 255, 0),
    "rangeland": (255, 0, 255),
    "forest_land": (0, 255, 0),
    "water": (0, 0, 255),
    "barren_land": (255, 255, 255),
}
UNKNOWN_COLOR = (0, 0, 0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- REPRODUCIBILITY --------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


# -------------------- PATH HELPERS --------------------

def resolve_path(value):
    """Resolve a metadata path without modifying the original dataset."""
    value = str(value).strip().replace("\\", "/")
    p = Path(value)

    candidates = [
        p if p.is_absolute() else DATA_ROOT / p,
        DATA_ROOT / p.name,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Fallback: search the dataset for the exact filename.
    matches = list(DATA_ROOT.rglob(p.name))
    if len(matches) == 1:
        return matches[0]

    return None


def load_training_records():
    if not METADATA.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA}")

    records = []

    with METADATA.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []

        required = {"sat_image_path", "mask_path"}
        if not required.issubset(set(columns)):
            raise ValueError(
                f"metadata.csv needs columns {required}. "
                f"Found: {columns}"
            )

        for row in reader:
            split = str(row.get("split", "")).strip().lower()
            image_value = row.get("sat_image_path", "")
            mask_value = row.get("mask_path", "")

            # Use only training rows; validation is created below.
            if split and split not in {"train", "training"}:
                continue

            image_path = resolve_path(image_value)
            mask_path = resolve_path(mask_value)

            if image_path is None or mask_path is None:
                continue

            records.append((image_path, mask_path))

    if not records:
        raise RuntimeError(
            "No image-mask pairs found. Check metadata.csv paths "
            "and its split column."
        )

    # Avoid accidental duplicate pairs.
    records = list(dict.fromkeys(records))
    return records


# -------------------- DATASET --------------------

class DeepGlobeDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        self.std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

    def __len__(self):
        return len(self.records)

    def encode_mask(self, mask_image):
        mask_image = mask_image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST,
        )
        mask = np.asarray(mask_image.convert("RGB"))
        labels = np.full(
            (IMAGE_SIZE, IMAGE_SIZE),
            IGNORE_INDEX,
            dtype=np.uint8,
        )

        for class_id, class_name in enumerate(CLASS_NAMES):
            color = np.array(CLASS_COLORS[class_name], dtype=np.uint8)
            matches = np.all(mask == color, axis=-1)
            labels[matches] = class_id

        # Black/unknown pixels and any unmapped colours remain ignored.
        return torch.from_numpy(labels.astype(np.int64))

    def __getitem__(self, index):
        image_path, mask_path = self.records[index]

        with Image.open(image_path) as im:
            image = im.convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.BILINEAR,
            )

        with Image.open(mask_path) as m:
            mask_image = m.convert("RGB")

        # Apply identical spatial augmentation to image and mask.
        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            mask_image = TF.hflip(mask_image)

        if self.augment and random.random() < 0.5:
            image = TF.vflip(image)
            mask_image = TF.vflip(mask_image)

        if self.augment and random.random() < 0.3:
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(0.9, 1.1)
            )
            image = ImageEnhance.Contrast(image).enhance(
                random.uniform(0.9, 1.1)
            )

        image_tensor = TF.to_tensor(image)
        image_tensor = (image_tensor - self.mean) / self.std
        mask_tensor = self.encode_mask(mask_image)

        return image_tensor, mask_tensor


# -------------------- U-NET WITH RESNET18 ENCODER --------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DeepGlobeUNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=NUM_CLASSES):
        super().__init__()

        weights = (
            models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        backbone = models.resnet18(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.dec3 = ConvBlock(512 + 256, 256)
        self.dec2 = ConvBlock(256 + 128, 128)
        self.dec1 = ConvBlock(128 + 64, 64)
        self.dec0 = ConvBlock(64 + 64, 64)

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        original_size = x.shape[-2:]

        x0 = self.relu(self.bn1(self.conv1(x)))  # 64, 128x128
        x1 = self.layer1(self.maxpool(x0))       # 64, 64x64
        x2 = self.layer2(x1)                     # 128, 32x32
        x3 = self.layer3(x2)                     # 256, 16x16
        x4 = self.layer4(x3)                     # 512, 8x8

        d3 = F.interpolate(
            x4, size=x3.shape[-2:], mode="bilinear",
            align_corners=False
        )
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = F.interpolate(
            d3, size=x2.shape[-2:], mode="bilinear",
            align_corners=False
        )
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = F.interpolate(
            d2, size=x1.shape[-2:], mode="bilinear",
            align_corners=False
        )
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        d0 = F.interpolate(
            d1, size=x0.shape[-2:], mode="bilinear",
            align_corners=False
        )
        d0 = self.dec0(torch.cat([d0, x0], dim=1))

        d0 = F.interpolate(
            d0, size=original_size, mode="bilinear",
            align_corners=False
        )
        return self.classifier(d0)


# -------------------- METRICS --------------------

def update_confusion_matrix(confusion, logits, targets):
    predictions = logits.argmax(dim=1)
    valid = targets != IGNORE_INDEX

    if valid.any():
        target_values = targets[valid]
        pred_values = predictions[valid]

        bins = torch.bincount(
            NUM_CLASSES * target_values + pred_values,
            minlength=NUM_CLASSES * NUM_CLASSES,
        )
        confusion += bins.reshape(NUM_CLASSES, NUM_CLASSES)


def calculate_metrics(confusion):
    confusion = confusion.float()
    true_positive = torch.diag(confusion)
    union = (
        confusion.sum(dim=1)
        + confusion.sum(dim=0)
        - true_positive
    )

    present = union > 0
    iou = torch.full_like(union, float("nan"))
    iou[present] = true_positive[present] / union[present]

    total = confusion.sum()
    pixel_accuracy = (
        true_positive.sum() / total
    ).item() if total > 0 else 0.0

    mean_iou = (
        iou[present].mean().item()
    ) if present.any() else 0.0

    per_class_iou = {
        CLASS_NAMES[i]: (
            float(iou[i].item()) if present[i] else None
        )
        for i in range(NUM_CLASSES)
    }

    return pixel_accuracy, mean_iou, per_class_iou


# -------------------- TRAINING --------------------

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    confusion = torch.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=torch.int64,
    device=DEVICE,
)

    for images, masks in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        update_confusion_matrix(
            confusion, logits.detach(), masks
        )

    average_loss = total_loss / len(loader.dataset)
    pixel_accuracy, mean_iou, per_class_iou = calculate_metrics(
        confusion
    )

    return average_loss, pixel_accuracy, mean_iou, per_class_iou


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {ROOT}")
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    records = load_training_records()
    random.Random(SEED).shuffle(records)

    val_count = max(1, int(len(records) * VAL_FRACTION))
    val_records = records[:val_count]
    train_records = records[val_count:]

    print(f"Verified pairs: {len(records)}")
    print(f"Training pairs: {len(train_records)}")
    print(f"Validation pairs: {len(val_records)}")
    print("Image size:", IMAGE_SIZE, "x", IMAGE_SIZE)
    print("Classes:", ", ".join(CLASS_NAMES))
    print("Unknown mask pixels are ignored.")

    train_dataset = DeepGlobeDataset(
        train_records, augment=True
    )
    val_dataset = DeepGlobeDataset(
        val_records, augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Pretrained ResNet18 weights may download on first run.
    model = DeepGlobeUNet(pretrained=True).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_miou = -1.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_iou, _ = run_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_acc, val_iou, val_class_iou = run_epoch(
            model, val_loader, criterion
        )

        scheduler.step(val_iou)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_pixel_accuracy": train_acc,
            "train_mean_iou": train_iou,
            "val_loss": val_loss,
            "val_pixel_accuracy": val_acc,
            "val_mean_iou": val_iou,
            "val_per_class_iou": val_class_iou,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val pixel acc {val_acc:.4f} | "
            f"val mIoU {val_iou:.4f}"
        )

        if val_iou > best_miou:
            best_miou = val_iou
            checkpoint_path = (
                CHECKPOINT_DIR / "deepglobe_unet_best.pth"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "class_colors": CLASS_COLORS,
                    "image_size": IMAGE_SIZE,
                    "ignore_index": IGNORE_INDEX,
                    "best_val_mean_iou": best_miou,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            print(f"  Saved best checkpoint: {checkpoint_path}")

        metrics_path = (
            METRICS_DIR / "deepglobe_training_metrics.json"
        )
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_pairs": len(records),
                    "train_pairs": len(train_records),
                    "validation_pairs": len(val_records),
                    "seed": SEED,
                    "image_size": IMAGE_SIZE,
                    "class_names": CLASS_NAMES,
                    "best_val_mean_iou": best_miou,
                    "history": history,
                },
                f,
                indent=2,
            )

    print("\nTraining complete.")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(
        "Checkpoint:",
        CHECKPOINT_DIR / "deepglobe_unet_best.pth"
    )
    print(
        "Metrics:",
        METRICS_DIR / "deepglobe_training_metrics.json"
    )
    print("Note: this is validation performance, not test performance.")


if __name__ == "__main__":
    main()
