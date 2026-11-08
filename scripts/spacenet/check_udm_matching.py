from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data/processed/SpaceNet7/images"
UDM_DIR = ROOT / "data/processed/SpaceNet7_UDM/ignore_masks"

images = list(IMAGE_DIR.glob("*_image.npy"))
udm_files = list(UDM_DIR.rglob("*_ignore.npy"))

udm_index = {
    p.name.replace("_ignore.npy", ""): p
    for p in udm_files
}

matched = []
missing = []

for image in images:
    key = image.name.replace("_image.npy", "")

    if key in udm_index:
        matched.append((image, udm_index[key]))
    else:
        missing.append(image)

print("\n========== UDM MATCHING REPORT ==========")
print("Total images:", len(images))
print("Total UDM masks:", len(udm_files))
print("Matched images:", len(matched))
print("Missing UDM:", len(missing))

print("\nMatched examples:")
for image, udm in matched[:5]:
    print(image.name)
    print(" ->", udm.name)

print("\nMissing examples:")
for image in missing[:5]:
    print(image.name)