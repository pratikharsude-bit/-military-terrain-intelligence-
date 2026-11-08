
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/processed/SpaceNet7"
SPLIT_DIR = DATA_DIR / "splits"
UDM_DIR = ROOT / "data/processed/SpaceNet7_UDM/ignore_masks"

IMAGE_SIZE = 256

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SpaceNetDataset(Dataset):

    def __init__(self, split="train"):

        if split not in ["train", "val", "test"]:
            raise ValueError("Invalid dataset split")

        csv_path = SPLIT_DIR / f"{split}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        if not UDM_DIR.exists():
            raise FileNotFoundError(f"UDM directory not found: {UDM_DIR}")

        with open(csv_path, newline="", encoding="utf-8") as f:
            self.records = list(csv.DictReader(f))

        # Build filename -> UDM mask mapping
        self.udm_index = {}

        for path in UDM_DIR.rglob("*_ignore.npy"):
            key = path.name.replace("_ignore.npy", "")

            if key in self.udm_index:
                raise ValueError(f"Duplicate UDM mask found for: {key}")

            self.udm_index[key] = path

        # Verify every image has a corresponding UDM mask
        for record in self.records:
            image_name = Path(record["image"]).name
            key = image_name.replace("_image.npy", "")

            if key not in self.udm_index:
                raise FileNotFoundError(
                    f"UDM mask missing for image: {image_name}"
                )

        print(
            f"{split} dataset loaded: {len(self.records)} pairs "
            f"with verified UDM masks"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image_path = DATA_DIR / record["image"]
        building_path = DATA_DIR / record["mask"]

        image_name = Path(record["image"]).name
        key = image_name.replace("_image.npy", "")
        udm_path = self.udm_index[key]

        # Load RGB satellite image
        image = np.load(image_path).copy()
        image = torch.from_numpy(image).float() / 255.0

        # Load building mask
        building_mask = np.load(building_path).copy()
        building_mask = torch.from_numpy(
            building_mask
        ).float().unsqueeze(0)

        # Load UDM ignore mask: 1 = ignore, 0 = valid
        ignore_mask = np.load(udm_path).copy()
        ignore_mask = torch.from_numpy(
            ignore_mask
        ).float().unsqueeze(0)

        # Check original dimensions
        if image.shape[-2:] != building_mask.shape[-2:]:
            raise ValueError(
                f"Image/building mask size mismatch: {image_name}"
            )

        if image.shape[-2:] != ignore_mask.shape[-2:]:
            raise ValueError(
                f"Image/UDM mask size mismatch: {image_name}"
            )

        # Convert ignore mask into valid mask
        valid_mask = (ignore_mask < 0.5).float()

        # Resize image
        image = F.interpolate(
            image.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # Resize building mask
        building_mask = F.interpolate(
            building_mask.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="nearest"
        ).squeeze(0)

        # Resize valid mask
        valid_mask = F.interpolate(
            valid_mask.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="nearest"
        ).squeeze(0)

        # Normalize image
        image = (image - MEAN) / STD

        return image, building_mask, valid_mask


def get_dataloaders(batch_size=4):

    train_dataset = SpaceNetDataset("train")
    val_dataset = SpaceNetDataset("val")
    test_dataset = SpaceNetDataset("test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


# Run tests only when this file is executed directly.
# Importing this module from train_spacenet.py will NOT run these tests.

if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_dataloaders()

    images, buildings, valid = next(iter(train_loader))

    print("\n===== DATALOADER + UDM TEST =====")
    print("Image shape:", images.shape)
    print("Building mask shape:", buildings.shape)
    print("Valid mask shape:", valid.shape)

    print("Image dtype:", images.dtype)
    print("Building mask values:", torch.unique(buildings))
    print("Valid mask values:", torch.unique(valid))

    print(
        "Valid pixel percentage:",
        round(valid.mean().item() * 100, 2), "%"
    )

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Testing batches:", len(test_loader))

    # Shape checks
    assert images.shape[0] == buildings.shape[0] == valid.shape[0]
    assert images.shape[2:] == buildings.shape[2:] == valid.shape[2:]
    assert buildings.shape == valid.shape

    # Binary mask checks
    assert set(torch.unique(buildings).tolist()).issubset({0.0, 1.0})
    assert set(torch.unique(valid).tolist()).issubset({0.0, 1.0})

    print("\nUDM integration test passed!")
