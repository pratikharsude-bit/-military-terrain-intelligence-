
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    ROOT / "data" / "raw" / "archive (7)" / "EuroSATallBands"
)


class EuroSATBandsDataset(Dataset):

    def __init__(self, split="train", transform=None):

        if split not in {"train", "validation", "test"}:
            raise ValueError(
                "split must be train, validation or test"
            )

        self.csv_path = DATASET_DIR / f"{split}.csv"
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {self.csv_path}"
            )

        self.data = pd.read_csv(self.csv_path)

        required_columns = {"Filename", "Label", "ClassName"}

        if not required_columns.issubset(self.data.columns):
            raise ValueError(
                f"CSV must contain columns: {required_columns}"
            )

        self.data["Label"] = self.data["Label"].astype(int)

        # Verify that CSV labels and class names agree.
        for _, row in self.data.iterrows():
            expected_name = row["ClassName"]

            if row["Filename"].split("/")[0] != expected_name:
                raise ValueError(
                    f"Class/path mismatch: {row['Filename']}"
                )

        self.image_paths = [
            DATASET_DIR / filename
            for filename in self.data["Filename"]
        ]

        self.labels = self.data["Label"].tolist()

        # Verify image files exist.
        missing = [
            str(path)
            for path in self.image_paths
            if not path.is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} image files are missing. "
                f"First missing file: {missing[0]}"
            )

        print(f"{split.upper()} dataset loaded")
        print(f"Images: {len(self.data)}")
        print(f"Classes: {self.data['ClassName'].nunique()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        label = self.labels[index]

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)

        # Expected layout: channels, height, width.
        if image.ndim != 3:
            raise ValueError(
                f"Unexpected image shape: {image.shape}"
            )

        if not np.isfinite(image).all():
            raise ValueError(
                f"Invalid pixel values in: {image_path}"
            )

        image = torch.from_numpy(image)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

