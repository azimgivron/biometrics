import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TwoBranchScratchNet(nn.Module):
    """
    Two-branch convolutional network from scratch for iris and fingerprint fusion.

    Processes grayscale iris and fingerprint images separately through small CNN branches,
    normalizes embeddings, concatenates, and projects to a joint embedding space.

    Args:
        emb_size: Output embedding dimension for the joint representation.
        dropout_prob: Dropout probability applied in each branch and merger.
    """

    def __init__(self, emb_size: int = 32, dropout_prob: float = 0.3) -> None:
        super().__init__()
        # ── Iris branch: Conv→BN→ReLU→Pooling→Dropout stacks
        self.branch_iris: nn.Sequential = nn.Sequential(
            # Input [B,1,H,W] → [B,16,H,W]
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Downsample by 2
            nn.Dropout2d(dropout_prob),
            # [B,16,H/2,W/2] → [B,32,H/2,W/2]
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Downsample by 2 again
            nn.Dropout2d(dropout_prob),
            # [B,32,H/4,W/4] → [B,64,1,1] via adaptive pooling
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # → [B,64]
        )

        # ── Fingerprint branch: same architecture as iris
        self.branch_fp: nn.Sequential = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # → [B,64]
        )

        # ── Joint head: merge 64+64 features → emb_size
        self.merger: nn.Sequential = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, emb_size),
        )

    def forward(self, iris: Tensor, fp: Tensor) -> Tensor:
        """
        Forward pass for paired iris and fingerprint inputs.

        Args:
            iris: Tensor of shape [B,1,H1,W1], grayscale iris images.
            fp:   Tensor of shape [B,1,H2,W2], grayscale fingerprint images.

        Returns:
            emb: Joint embedding tensor of shape [B, emb_size].
        """
        # Extract branch features
        iris_feat: Tensor = self.branch_iris(iris)  # [B,64]
        fp_feat: Tensor = self.branch_fp(fp)  # [B,64]

        # L2-normalize each branch's output
        iris_feat = F.normalize(iris_feat, p=2, dim=1)
        fp_feat = F.normalize(fp_feat, p=2, dim=1)

        # Concatenate and project
        merged: Tensor = torch.cat([iris_feat, fp_feat], dim=1)  # [B,128]
        emb: Tensor = self.merger(merged)  # [B,emb_size]
        return emb
