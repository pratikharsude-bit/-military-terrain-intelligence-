from torch.utils.data import DataLoader
from dataset_eurosat_bands import EuroSATBandsDataset

# Load training dataset
train_dataset = EuroSATBandsDataset(split="train")

print("\n--- SINGLE IMAGE TEST ---")

image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Label:", label)
print("Image data type:", image.dtype)
print("Minimum pixel value:", image.min().item())
print("Maximum pixel value:", image.max().item())

print("\n--- DATALOADER TEST ---")

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch labels shape:", labels.shape)
print("Batch labels:", labels)

print("\n--- LOADER TEST COMPLETE ---")