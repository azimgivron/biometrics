import random
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.morphology import (binary_dilation, footprint_rectangle,
                                skeletonize)
from torchvision import transforms

from last_q.src.fingerprint_enhancer import FingerprintImageEnhancer


def read_DB(path: Union[str, Path]) -> Tuple[List[np.ndarray], List[str]]:
    """
    Read all PNG images from a directory (recursively) and extract their labels.

    Args:
        path: Path or string pointing to the root folder containing PNG images.

    Returns:
        A tuple with two lists:
          - images: Grayscale image arrays.
          - labels: Corresponding labels extracted from filename stems (first 3 characters).
    """
    images: List[np.ndarray] = []
    labels: List[str] = []
    image_paths = sorted(Path(path).rglob("*.png"))
    for image_path in image_paths:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.ndim > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)
        labels.append(image_path.stem[:3])
    return images, labels


class AddGaussianNoise(torch.nn.Module):
    """
    Add Gaussian noise to a tensor.

    Attributes:
        mean: Noise mean.
        std: Noise standard deviation.
    """

    def __init__(self, mean: float = 0.0, std: float = 0.05) -> None:
        super().__init__()
        self.mean: float = mean
        self.std: float = std

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the input tensor.

        Args:
            tensor: Input tensor.

        Returns:
            Tensor with added Gaussian noise.
        """
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class Clamp(torch.nn.Module):
    """
    Clamp tensor values to [0, 1].
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(tensor, 0.0, 1.0)


class EnhanceFP(torch.nn.Module):
    """
    Apply fingerprint enhancement and mask within a torchvision transform.

    Args:
        fe: FingerprintImageEnhancer instance.
    """

    def __init__(self, fe: FingerprintImageEnhancer) -> None:
        super().__init__()
        self.fe = fe
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Enhance a single-channel fingerprint tensor and apply its mask.

        Args:
            tensor: Tensor of shape (1, H, W) in [0, 1].

        Returns:
            Enhanced tensor of same shape.
        """
        arr = (tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        enhanced = self.fe.enhance(arr)
        mask = self.fe._mask
        out = (enhanced * mask).astype(np.uint8)
        pil = Image.fromarray(out, mode="L")
        return self.to_tensor(pil)


class RandomThinning(torch.nn.Module):
    """
    Randomly thin skeletonized fingerprint image.

    Args:
        p: Probability to apply thinning.
        threshold: Binarization threshold.
        max_thickness: Maximum dilation thickness (odd integer).
    """

    def __init__(
        self, p: float = 0.5, threshold: int = 128, max_thickness: int = 5
    ) -> None:
        super().__init__()
        self.p: float = p
        self.th: int = threshold
        self.max_thickness: int = (
            max_thickness if max_thickness % 2 == 1 else max_thickness + 1
        )
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply random thinning to a binary fingerprint tensor.

        Args:
            img: Tensor of shape (1, H, W) in [0, 1].

        Returns:
            Possibly thinned tensor or original if not applied.
        """
        if random.random() < self.p:
            arr = (img.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            bw = arr > self.th
            skel = skeletonize(bw)
            odds = list(range(1, self.max_thickness + 1, 2))
            thickness = random.choice(odds)
            thick = binary_dilation(
                skel,
                footprint_rectangle((thickness, thickness)),
            )
            out = thick.astype(np.uint8) * 255
            pil = Image.fromarray(out, mode="L")
            return self.to_tensor(pil)
        return img


def augment_variants(
    args: Tuple[np.ndarray, int, transforms.Compose],
) -> List[np.ndarray]:
    """
    Generate multiple augmented variants of an image tensor.

    Args:
        args: Tuple containing:
          - img_np: Numpy image array.
          - num_augs: Number of augmentations to generate.
          - transform: torchvision transforms pipeline.

    Returns:
        List of augmented numpy arrays.
    """
    img_np, num_augs, transform = args
    return [transform(img_np).squeeze(0).cpu().numpy() for _ in range(num_augs)]
