import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoConfig, AutoImageProcessor, EfficientNetModel

# Suppress deprecation/other warnings
warnings.filterwarnings("ignore")

# Load EfficientNet-B0 configuration and preprocessing
config = AutoConfig.from_pretrained("google/efficientnet-b0")
image_processor = AutoImageProcessor.from_pretrained(
    "google/efficientnet-b0", use_fast=True
)
# Spatial size to which all input images will be resized
BACKBONE_SPATIAL: Tuple[int, int] = (
    image_processor.size["height"],
    image_processor.size["width"],
)


class TwoBranchEfficientNet(nn.Module):
    """
    Two-branch EfficientNet model for fusion of iris and fingerprint images.

    Processes each modality independently through a shared EfficientNet backbone,
    then merges, reduces dimensionality via conv layers, and projects to class logits.

    Args:
        n_classes: Number of output classes.
        freeze_until: Name of the first backbone parameter to unfreeze. All
            parameters before this name will be frozen, others remain trainable.
    """

    def __init__(
        self, n_classes: int = 100, freeze_until: Optional[str] = "encoder.blocks.15"
    ) -> None:
        super().__init__()
        # Register ImageNet mean/std for normalization
        mean = torch.tensor(image_processor.image_mean)[None, :, None, None]
        std = torch.tensor(image_processor.image_std)[None, :, None, None]
        self.register_buffer("imgnet_mean", mean)
        self.register_buffer("imgnet_std", std)

        # Shared EfficientNet-B0 backbone
        self.backbone: EfficientNetModel = EfficientNetModel.from_pretrained(
            "google/efficientnet-b0"
        )
        hidden_dim: int = self.backbone.config.hidden_dim

        # Freeze backbone parameters up to `freeze_until`
        unfreeze = False
        for name, param in self.backbone.named_parameters():
            # When we hit freeze_until or already unfreezing, keep grad
            if freeze_until and freeze_until in name:
                unfreeze = True
            param.requires_grad = unfreeze

        # Channel-pool to reduce 2×hidden_dim features to hidden_dim/4
        self.channel_pool = nn.AdaptiveAvgPool1d(hidden_dim * 2 // 4)
        # Convolutional head after merging branches
        self.post_merge_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2 // 4, hidden_dim * 2 // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2 // 8, hidden_dim * 2 // 16, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Final MLP classifier
        self.merger = nn.Sequential(
            nn.Linear(hidden_dim * 2 // 16, hidden_dim * 2 // 32),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2 // 32, n_classes),
        )

    def forward(self, iris: Tensor, fp: Tensor) -> Tensor:
        """
        Forward pass for paired iris and fingerprint tensors.

        Args:
            iris: Tensor of shape [B, C, H1, W1], iris images.
            fp:   Tensor of shape [B, C, H2, W2], fingerprint images.

        Returns:
            Logits tensor of shape [B, n_classes].
        """
        # 1) Resize both modalities to backbone's expected spatial size
        iris_resized = F.interpolate(
            iris, size=BACKBONE_SPATIAL, mode="bilinear", align_corners=False
        )
        fp_resized = F.interpolate(
            fp, size=BACKBONE_SPATIAL, mode="bilinear", align_corners=False
        )

        # 2) Normalize using ImageNet statistics
        iris_norm = (iris_resized - self.imgnet_mean) / self.imgnet_std
        fp_norm = (fp_resized - self.imgnet_mean) / self.imgnet_std

        # 3) Extract pooled features from EfficientNet backbone
        iris_feat = self.backbone(
            pixel_values=iris_norm
        ).pooler_output  # [B, hidden_dim]
        fp_feat = self.backbone(pixel_values=fp_norm).pooler_output  # [B, hidden_dim]

        # 4) L2 normalize embeddings
        iris_feat = F.normalize(iris_feat, p=2, dim=1)
        fp_feat = F.normalize(fp_feat, p=2, dim=1)

        # 5) Concatenate features and reduce dimensionality
        merged = torch.cat([iris_feat, fp_feat], dim=1)  # [B, 2*hidden_dim]
        pooled = self.channel_pool(merged.unsqueeze(2))  # [B, (2*hidden_dim)/4, 1]
        # Reshape to 4D for conv: [B, channels, 1, 1]
        conv_in = pooled.unsqueeze(-1)
        conv_out = self.post_merge_conv(conv_in)  # [B, (2*hidden_dim)/16, 1, 1]
        feat = conv_out.view(conv_out.size(0), -1)  # [B, (2*hidden_dim)/16]

        # 6) Final classifier projection
        logits = self.merger(feat)  # [B, n_classes]
        return logits
