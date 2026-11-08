
from pathlib import Path
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "SkyView"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
METRICS_DIR = ROOT / "experiments" / "metrics"

BEST_MODEL = CHECKPOINT_DIR / "skyview_resnet18_best.pth"
METRICS_FILE = METRICS_DIR / "skyview_training_metrics.json"

# ---------- Training configuration ----------
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
SEED = 42
NUM_WORKERS = 0  # Safe default for Windows

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("=" * 55)
    print("SKYVIEW TERRAIN / LANDSCAPE CLASSIFIER")
    print("=" * 55)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---------- Verify dataset ----------
    for split in ("train", "val", "test"):
        if not (DATA_DIR / split).is_dir():
            raise FileNotFoundError(
                f"Missing {split} folder: {DATA_DIR / split}"
            )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Image transformations ----------
    weights = ResNet18_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ---------- Load datasets ----------
    train_dataset = datasets.ImageFolder(
        DATA_DIR / "train", transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        DATA_DIR / "val", transform=eval_transform
    )
    test_dataset = datasets.ImageFolder(
        DATA_DIR / "test", transform=eval_transform
    )

    if not (
        train_dataset.class_to_idx
        == val_dataset.class_to_idx
        == test_dataset.class_to_idx
    ):
        raise ValueError("Class folders differ between dataset splits.")

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    # ---------- Load pretrained ResNet18 ----------
    print("\nLoading pretrained ResNet18 weights...")
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_accuracy = 0.0
    history = []

    # ---------- Training loop ----------
    for epoch in range(EPOCHS):
        start = time.time()

        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            train_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_accuracy = 100.0 * train_correct / train_total

        # ---------- Validation ----------
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                val_correct += (
                    outputs.argmax(dim=1) == labels
                ).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_accuracy = 100.0 * val_correct / val_total
        scheduler.step(val_accuracy)

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 5),
            "train_accuracy": round(train_accuracy, 3),
            "val_loss": round(val_loss, 5),
            "val_accuracy": round(val_accuracy, 3),
        }
        history.append(epoch_info)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.2f}% | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_accuracy:.2f}% | "
            f"Time: {time.time() - start:.1f}s"
        )

        # Save only when validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "class_to_idx": train_dataset.class_to_idx,
                "num_classes": num_classes,
                "best_val_accuracy": best_val_accuracy,
                "epoch": epoch + 1,
                "architecture": "resnet18",
            }, BEST_MODEL)
            print(f"  Best checkpoint saved: {BEST_MODEL.name}")

        with open(METRICS_FILE, "w", encoding="utf-8") as file:
            json.dump({
                "architecture": "ResNet18",
                "device": str(device),
                "classes": class_names,
                "best_validation_accuracy": best_val_accuracy,
                "history": history,
            }, file, indent=2)

    # ---------- Evaluate best checkpoint on test set ----------
    print("\nEvaluating best checkpoint on the test set...")

    checkpoint = torch.load(
        BEST_MODEL, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_correct = 0
    test_total = 0
    test_loss_sum = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss_sum += loss.item() * images.size(0)
            test_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            test_total += labels.size(0)

    test_loss = test_loss_sum / test_total
    test_accuracy = 100.0 * test_correct / test_total

    results = {
        "architecture": "ResNet18",
        "device": str(device),
        "classes": class_names,
        "best_epoch": checkpoint["epoch"],
        "best_validation_accuracy": round(
            checkpoint["best_val_accuracy"], 3
        ),
        "test_accuracy": round(test_accuracy, 3),
        "test_loss": round(test_loss, 5),
        "history": history,
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\n" + "=" * 55)
    print("TRAINING COMPLETE")
    print(f"Best validation accuracy: {checkpoint['best_val_accuracy']:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Model: {BEST_MODEL}")
    print(f"Metrics: {METRICS_FILE}")
    print("=" * 55)


if __name__ == "__main__":
    main()
