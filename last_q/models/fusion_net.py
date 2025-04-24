import torch
from torch import Tensor, nn


class FusionNet(nn.Module):
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

    def __init__(self, input_size: int, n_classes: int, dropout_prob: float = 0.6) -> None:
        super().__init__()
        # Define sequential merger
        self.merger: nn.Sequential = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout2d(dropout_prob),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
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

    def forward(self,  fp_feat: Tensor, iris_feat: Tensor) -> Tensor:
        """
        Forward pass to merge modality features.

        Args:
            fp_feat:   Tensor from FPNet.
            iris_feat: Tensor from IrisNet.

        Returns:
            emb: Joint embedding [B, n_classes].
        """
        # Concatenate along feature dimension
        merged: Tensor = torch.cat([iris_feat, fp_feat], dim=1)
        emb: Tensor = self.merger(merged)
        return emb
