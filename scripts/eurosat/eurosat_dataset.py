
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT / "data" / "raw" / "archive (7)" / "EuroSATallBands"
)

TRAIN_CSV = DATASET_DIR / "train.csv"
STATS_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "metrics"
    / "eurosat_allbands_train_stats.json"
)


class EuroSATAllBandsDataset(Dataset):

    def __init__(self, csv_file, dataset_dir, stats_file):

        self.data = pd.read_csv(csv_file)
        self.dataset_dir = dataset_dir

        # Load training-set statistics
        import json

        with open(stats_file, "r") as file:
            stats = json.load(file)

        self.means = np.array(
            [stats["bands"][f"band_{i}"]["mean"] for i in range(1, 14)],
            dtype=np.float32
        )

        self.stds = np.array(
            [stats["bands"][f"band_{i}"]["std"] for i in range(1, 14)],
            dtype=np.float32
        )

        self.means = self.means[:, None, None]
        self.stds = self.stds[:, None, None]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = self.dataset_dir / row["Filename"]
        label = int(row["Label"])

        # Read all 13 spectral bands
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)

        # Normalize each band using training statistics
        image = (image - self.means) / self.stds

        # Convert NumPy array into PyTorch tensor
        image = torch.from_numpy(image)

        return image, label


def main():

    print("\n========== EUROSAT DATALOADER TEST ==========\n")

    dataset = EuroSATAllBandsDataset(
        csv_file=TRAIN_CSV,
        dataset_dir=DATASET_DIR,
        stats_file=STATS_FILE
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    images, labels = next(iter(dataloader))

    print("Total training images:", len(dataset))
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    print("Image data type:", images.dtype)
    print("Label data type:", labels.dtype)

    print("\nFirst 10 labels:", labels[:10].tolist())

    print("Normalized batch mean:", images.mean().item())
    print("Normalized batch std:", images.std().item())

    print("\n========== DATALOADER TEST COMPLETE ==========")


if __name__ == "__main__":
    main()
