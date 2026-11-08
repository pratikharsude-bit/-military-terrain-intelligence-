import torch
from pathlib import Path

checkpoint_dir = Path("models/checkpoints")

files = [
    "skyview_resnet18_best.pth",
    "deepglobe_unet_best.pth",
    "eurosat_allbands_resnet18_best.pth",
]

for filename in files:
    path = checkpoint_dir / filename

    print(f"\n--- {filename} ---")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    print("Checkpoint type:", type(checkpoint).__name__)

    if isinstance(checkpoint, dict):
        print("Top-level keys:", list(checkpoint.keys()))

        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint)
        )

        if isinstance(state_dict, dict):
            print("First 10 parameter keys:")
            for key in list(state_dict.keys())[:10]:
                print(" ", key)
    else:
        print("Checkpoint is not a dictionary.")