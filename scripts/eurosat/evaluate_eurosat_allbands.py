
from pathlib import Path
import json

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from eurosat_dataset import EuroSATAllBandsDataset
from eurosat_model import EuroSATResNet18


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT / "data" / "raw" / "archive (7)" / "EuroSATallBands"
)

TEST_CSV = DATASET_DIR / "test.csv"

STATS_FILE = (
    PROJECT_ROOT / "experiments" / "metrics"
    / "eurosat_allbands_train_stats.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "checkpoints"
    / "eurosat_allbands_resnet18_best.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "metrics"

RESULTS_FILE = OUTPUT_DIR / "eurosat_allbands_test_results.json"
CONFUSION_FILE = OUTPUT_DIR / "eurosat_allbands_confusion_matrix.csv"


# ==========================================
# CONFIGURATION
# ==========================================

BATCH_SIZE = 32

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
    "SeaLake"
]


# ==========================================
# EVALUATION
# ==========================================

def main():

    print("\n========== EUROSAT TEST EVALUATION ==========\n")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Load test dataset
    test_dataset = EuroSATAllBandsDataset(
        TEST_CSV,
        DATASET_DIR,
        STATS_FILE
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    print("Test images:", len(test_dataset))

    # Load model
    model = EuroSATResNet18(num_classes=10).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint from epoch:", checkpoint["epoch"])
    print(
        "Best validation accuracy:",
        checkpoint["validation_accuracy"]
    )

    # Predictions
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch_index, (images, labels) in enumerate(test_loader):

            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

            if (batch_index + 1) % 20 == 0:
                print(
                    f"Processed batches: "
                    f"{batch_index + 1}/{len(test_loader)}"
                )

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(10)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(10))
    )

    print("\n========== TEST RESULTS ==========\n")

    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:\n")

    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(10)),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    # Save confusion matrix
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    confusion_df = pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    confusion_df.to_csv(CONFUSION_FILE)

    # Save complete results
    results = {
        "test_images": len(test_dataset),
        "checkpoint_epoch": checkpoint["epoch"],
        "validation_accuracy": checkpoint["validation_accuracy"],
        "test_accuracy": accuracy,
        "test_accuracy_percent": accuracy * 100,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_names": CLASS_NAMES
    }

    with open(RESULTS_FILE, "w") as file:
        json.dump(results, file, indent=4)

    print("\nResults saved:")
    print(RESULTS_FILE)

    print("\nConfusion matrix saved:")
    print(CONFUSION_FILE)

    print("\n========== EVALUATION COMPLETE ==========")


if __name__ == "__main__":
    main()
