from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/SpaceNet7"

mask_files = sorted(DATA.rglob("*_mask.npy"))

print("Total masks:", len(mask_files))

non_empty = 0
empty = 0

for path in mask_files:
    mask = np.load(path)
    if mask.any():
        non_empty += 1
    else:
        empty += 1

print("Non-empty masks:", non_empty)
print("Empty masks:", empty)

total_pixels = 0
building_pixels = 0

for path in mask_files:
    mask = np.load(path)

    total_pixels += mask.size
    building_pixels += np.count_nonzero(mask == 1)

background_pixels = total_pixels - building_pixels

print("\n===== CLASS DISTRIBUTION =====")
print("Total pixels:", total_pixels)
print("Building pixels:", building_pixels)
print("Background pixels:", background_pixels)

print(
    "Building percentage:",
    round(building_pixels / total_pixels * 100, 4)
)

print(
    "Background percentage:",
    round(background_pixels / total_pixels * 100, 4)
)

# Select a sample containing buildings
sample = next(
    (p for p in mask_files if np.load(p).any()),
    None
)

if sample is None:
    raise RuntimeError("No non-empty building masks found!")

image_path = sample.with_name(
    sample.name.replace("_mask.npy", "_image.npy")
)

image = np.load(image_path)
mask = np.load(sample)

print("\nSample:", sample.name)
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Building pixels:", int(mask.sum()))
print("Building coverage:", round(mask.mean() * 100, 3), "%")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(np.moveaxis(image, 0, -1))
axes[0].set_title("Satellite Image")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Building Mask")

axes[2].imshow(np.moveaxis(image, 0, -1))

building_overlay = np.ma.masked_where(mask == 0, mask)

axes[2].imshow(
    building_overlay,
    cmap="autumn",
    alpha=0.85,
    interpolation="nearest"
)

axes[2].set_title("Building Overlay")
axes[2].set_title("Building Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output = ROOT / "outputs/spacenet_mask_inspection.png"
output.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output, dpi=150)
plt.show()

print("\nSaved visualization:", output)