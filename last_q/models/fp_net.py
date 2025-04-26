import torch.nn.functional as F
from torch import Tensor, nn

class FPNet(nn.Module):
    """
    Convolutional branch for fingerprint images, outputs a normalized 64-dim vector.

    Uses same architecture and parameters as IrisNet.

    Args:
        dropout_prob: Dropout probability for regularization.
    """

    def __init__(self, dropout_prob: float = 0.3) -> None:
        super().__init__()
        # Reuse same feature extractor structure
        self.features: nn.Sequential = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Weights initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for fingerprint images.

        Args:
            x: Input tensor of shape [B, 1, H, W].

        Returns:
            Normalized feature tensor of shape [B, 64].
        """
        feat: Tensor = self.features(x)
        feat = F.normalize(feat, p=2, dim=1)  # L2 normalize
        return feat
