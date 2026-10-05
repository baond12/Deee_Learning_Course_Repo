from __future__ import annotations

from torch import nn


class SoftmaxRegression(nn.Module):
    """Linear classifier on flattened images.

    The forward method returns logits. During training, nn.CrossEntropyLoss applies
    log-softmax internally, which is numerically more stable than applying softmax
    in the model itself.
    """

    def __init__(self, input_dim: int = 28 * 28, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.20),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(name: str) -> nn.Module:
    if name == "softmax":
        return SoftmaxRegression()
    if name == "mlp":
        return MLPClassifier()
    if name == "cnn":
        return SmallCNN()
    raise ValueError(f"Unknown model: {name}")

