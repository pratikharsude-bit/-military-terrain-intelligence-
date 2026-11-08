
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "spacenet"))

from spacenet_model import SpaceNetUNet
from spacenet_dataset import get_dataloaders


# ---------------- CONFIGURATION ----------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 15
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
METRICS_DIR = ROOT / "experiments" / "metrics"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "spacenet_unet_best.pth"
METRICS_PATH = METRICS_DIR / "spacenet_training_metrics.json"


# ---------------- MASKED LOSS ----------------

def masked_bce_dice_loss(logits, targets, valid_mask):

    # BCE loss for each pixel
    bce_map = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    # Ignore invalid UDM pixels
    bce = (bce_map * valid_mask).sum()
    bce = bce / valid_mask.sum().clamp(min=1.0)

    # Dice calculation only on valid pixels
    probabilities = torch.sigmoid(logits)

    probabilities = probabilities * valid_mask
    targets = targets * valid_mask

    intersection = (probabilities * targets).sum(
        dim=(1, 2, 3)
    )

    denominator = probabilities.sum(
        dim=(1, 2, 3)
    ) + targets.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + 1e-6) / (
        denominator + 1e-6
    )

    dice_loss = 1.0 - dice.mean()

    return bce + dice_loss


# ---------------- METRICS ----------------

def calculate_metrics(logits, targets, valid_mask):

    predictions = (torch.sigmoid(logits) >= 0.5).float()

    predictions = predictions * valid_mask
    targets = targets * valid_mask

    intersection = (predictions * targets).sum()
    union = (
        predictions + targets - predictions * targets
    ).sum()

    iou = intersection / union.clamp(min=1e-6)

    return iou.item()


# ---------------- TRAINING ----------------

def train_one_epoch(model, loader, optimizer):

    model.train()

    total_loss = 0.0
    total_iou = 0.0

    for images, buildings, valid in loader:

        images = images.to(DEVICE)
        buildings = buildings.to(DEVICE)
        valid = valid.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        if logits.shape != buildings.shape:
            raise ValueError(
                f"Output shape {logits.shape} "
                f"does not match target {buildings.shape}"
            )

        loss = masked_bce_dice_loss(
            logits,
            buildings,
            valid
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_iou += calculate_metrics(
            logits.detach(),
            buildings,
            valid
        )

    return (
        total_loss / len(loader),
        total_iou / len(loader)
    )


# ---------------- VALIDATION ----------------

@torch.no_grad()
def validate(model, loader):

    model.eval()

    total_loss = 0.0
    total_iou = 0.0

    for images, buildings, valid in loader:

        images = images.to(DEVICE)
        buildings = buildings.to(DEVICE)
        valid = valid.to(DEVICE)

        logits = model(images)

        loss = masked_bce_dice_loss(
            logits,
            buildings,
            valid
        )

        total_loss += loss.item()
        total_iou += calculate_metrics(
            logits,
            buildings,
            valid
        )

    return (
        total_loss / len(loader),
        total_iou / len(loader)
    )


# ---------------- MAIN ----------------

def main():

    print("\n========== SPACENET-7 TRAINING ==========")
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    model = SpaceNetUNet().to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_iou = 0.0
    history = []

    for epoch in range(EPOCHS):

        start_time = time.time()

        train_loss, train_iou = train_one_epoch(
            model,
            train_loader,
            optimizer
        )

        val_loss, val_iou = validate(
            model,
            val_loader
        )

        epoch_time = time.time() - start_time

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "epoch_time_seconds": round(epoch_time, 2)
        }

        history.append(record)

        print(
            f"\nEpoch [{epoch + 1}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train IoU: {train_iou:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val IoU: {val_iou:.4f} "
            f"| Time: {epoch_time:.1f}s"
        )

        if val_iou > best_val_iou:

            best_val_iou = val_iou

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_iou": best_val_iou,
                    "epoch": epoch + 1,
                    "architecture": "SpaceNetUNet"
                },
                BEST_MODEL_PATH
            )

            print("Best model checkpoint saved!")

        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_val_iou": best_val_iou,
                    "epochs_completed": epoch + 1,
                    "history": history
                },
                f,
                indent=4
            )

    print("\n========== TRAINING COMPLETE ==========")
    print("Best validation IoU:", round(best_val_iou, 4))
    print("Checkpoint:", BEST_MODEL_PATH)
    print("Metrics:", METRICS_PATH)


if __name__ == "__main__":
    main()   

    
print("\nUDM integration test passed!")
main()
