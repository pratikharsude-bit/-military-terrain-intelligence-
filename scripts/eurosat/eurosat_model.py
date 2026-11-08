
import torch
import torch.nn as nn
from torchvision import models


class EuroSATResNet18(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        # Create ResNet18 without pretrained weights
        self.model = models.resnet18(weights=None)

        # Modify first convolution:
        # RGB input (3 channels) -> Multispectral input (13 channels)
        self.model.conv1 = nn.Conv2d(
            in_channels=13,
            out_channels=64,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False
        )

        # Remove initial max pooling to preserve spatial information
        self.model.maxpool = nn.Identity()

        # Modify final classification layer for 10 EuroSAT classes
        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features,
            out_features=num_classes
        )

    def forward(self, x):
        return self.model(x)


def main():

    print("\n========== EUROSAT MODEL TEST ==========\n")

    model = EuroSATResNet18(num_classes=10)

    # Simulate one batch of 32 multispectral images
    dummy_input = torch.randn(32, 13, 64, 64)

    with torch.no_grad():
        output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))

    print("\n========== MODEL TEST COMPLETE ==========")


if __name__ == "__main__":
    main()
