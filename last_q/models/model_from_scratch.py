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

    def __init__(self, emb_size: int = 64, dropout_prob: float = 0.3) -> None:
        super().__init__()
        
        def make():
            return nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout2d(dropout_prob),
                nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout2d(dropout_prob),
                nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Flatten(),
            )

        # input size: 200, 500
        self.branch_iris: nn.Sequential = make()

        # ── Fingerprint branch: same architecture as iris
        # input size: 480, 320
        self.branch_fp: nn.Sequential = make()
        
        def _get_flatten_size(branch, H, W):
            x = torch.zeros(1, 1, H, W)
            return branch(x).shape[-1]

        iris_dim = _get_flatten_size(self.branch_iris, 200, 500)
        fp_dim   = _get_flatten_size(self.branch_fp,   480, 320)
        merger_in = iris_dim + fp_dim

        self.merger: nn.Sequential = nn.Sequential(
            nn.Linear(merger_in, 128),
            nn.ReLU(),
            nn.Linear(128, emb_size),
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
