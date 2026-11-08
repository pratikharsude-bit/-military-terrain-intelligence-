
from pathlib import Path
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eurosat_dataset import EuroSATAllBandsDataset
from eurosat_model import EuroSATResNet18


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT / "data" / "raw" / "archive (7)" / "EuroSATallBands"
)

TRAIN_CSV = DATASET_DIR / "train.csv"
VAL_CSV = DATASET_DIR / "validation.csv"
STATS_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "metrics"
    / "eurosat_allbands_train_stats.json"
)

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
METRICS_DIR = PROJECT_ROOT / "experiments" / "metrics"

BEST_MODEL_PATH = CHECKPOINT_DIR / "eurosat_allbands_resnet18_best.pth"
METRICS_PATH = METRICS_DIR / "eurosat_allbands_training_metrics.json"


# --------------------------------------------------
# TRAINING CONFIGURATION
# --------------------------------------------------

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 5

NUM_CLASSES = 10


# --------------------------------------------------
# TRAINING FUNCTION
# --------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# --------------------------------------------------
# VALIDATION FUNCTION
# --------------------------------------------------

def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = total_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# --------------------------------------------------
# MAIN TRAINING PIPELINE
# --------------------------------------------------

def main():

    print("\n========== EUROSAT ALL BANDS TRAINING ==========\n")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Create datasets
    train_dataset = EuroSATAllBandsDataset(
        TRAIN_CSV, DATASET_DIR, STATS_FILE
    )

    val_dataset = EuroSATAllBandsDataset(
        VAL_CSV, DATASET_DIR, STATS_FILE
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Batch size:", BATCH_SIZE)

    # Create model
    model = EuroSATResNet18(
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    history = {
        "epochs": [],
        "best_validation_accuracy": 0.0,
        "configuration": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "loss": "CrossEntropyLoss",
            "architecture": "ResNet18",
            "input_channels": 13,
            "num_classes": NUM_CLASSES
        }
    }

    best_accuracy = 0.0
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(EPOCHS):

        start_time = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - start_time

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Time: {epoch_time:.2f} seconds")

        history["epochs"].append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "learning_rate": current_lr,
            "time_seconds": epoch_time
        })

        # Save best model
        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": best_accuracy
                },
                BEST_MODEL_PATH
            )

            print("Best model checkpoint saved!")

        else:
            epochs_without_improvement += 1

        history["best_validation_accuracy"] = best_accuracy

        with open(METRICS_PATH, "w") as file:
            json.dump(history, file, indent=4)

        # Early stopping
        if epochs_without_improvement >= PATIENCE:

            print("\nEarly stopping triggered.")
            break

    print("\n========== TRAINING COMPLETE ==========")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"Best Model: {BEST_MODEL_PATH}")
    print(f"Training Metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
