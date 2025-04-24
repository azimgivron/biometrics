from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class IrisNet(nn.Module):
    """
    Convolutional branch for iris images, outputs a normalized 64-dim feature vector.

    Architecture:
      - Conv-BN-ReLU
      - MaxPool
      - Dropout2d
      - Repeat with increasing channels
      - AdaptiveAvgPool to [1,1]
      - Flatten

    Args:
        dropout_prob: Dropout probability for regularization.
    """

    def __init__(self, dropout_prob: float = 0.3) -> None:
        super().__init__()
        # Sequential feature extractor for iris modality
        self.features: nn.Sequential = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample H,W by 2
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample by another factor of 2
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Output spatial size [1,1]
            nn.Flatten(),  # → [B, 64]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for iris images.

        Args:
            x: Input tensor of shape [B, 1, H, W].

        Returns:
            Normalized feature tensor of shape [B, 64].
        """
        feat: Tensor = self.features(x)
        feat = F.normalize(feat, p=2, dim=1)  # L2 normalize
        return feat


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
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # → [B, 64]
        )

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


class MergerNet(nn.Module):
    """
    Merge head that combines two 64-dim features into an embedding of size `emb_size`.

    Architecture:
      - Linear 128→64
      - BatchNorm1d
      - ReLU
      - Dropout
      - Linear 64→emb_size

    Args:
        emb_size: Dimension of output embedding.
        dropout_prob: Dropout probability.
    """

    def __init__(self, emb_size: int = 32, dropout_prob: float = 0.3) -> None:
        super().__init__()
        # Define sequential merger
        self.merger: nn.Sequential = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, emb_size),
        )

    def forward(self, iris_feat: Tensor, fp_feat: Tensor) -> Tensor:
        """
        Forward pass to merge modality features.

        Args:
            iris_feat: Tensor [B, 64] from IrisNet.
            fp_feat:   Tensor [B, 64] from FPNet.

        Returns:
            emb: Joint embedding [B, emb_size].
        """
        # Concatenate along feature dimension
        merged: Tensor = torch.cat([iris_feat, fp_feat], dim=1)  # [B,128]
        emb: Tensor = self.merger(merged)
        return emb
