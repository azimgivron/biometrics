import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def make_splits(n: int, tn: int, vn: int) -> Dict[str, np.ndarray]:
    """
    Randomly partition indices [0..n) into train, val, and test sets.

    Args:
        n: Total number of samples.
        tn: Number of training samples.
        vn: Number of validation samples.

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to sorted index arrays.
    """
    # Generate a random permutation of indices
    perm = np.random.permutation(n)
    # First tn are training, next vn are validation, rest are test
    train_idx = np.sort(perm[:tn])
    val_idx = np.sort(perm[tn : tn + vn])
    test_idx = np.sort(perm[tn + vn :])
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def load(
    iris_root: Path, fp_root: Path, train_pct: float, val_pct: float
) -> Dict[str, List[Tuple[Path, Path, int]]]:
    """
    Load paired iris and fingerprint samples, splitting into train/val/test.

    Args:
        iris_root: Path to root folder of iris images organized by label.
        fp_root: Path to root folder of fingerprint images by label (000–999).
        train_pct: Fraction of each label's samples for training (0<train_pct<1).
        val_pct: Fraction of each label's samples for validation (0<val_pct<1).

    Returns:
        Dictionary mapping 'train', 'val', 'test' to lists of tuples
        (iris_path, fingerprint_path, label).
    """
    samples: Dict[str, List[Tuple[Path, Path, int]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    # Iterate over each label directory in the iris dataset
    for label_dir in sorted(iris_root.iterdir()):
        if not label_dir.is_dir():
            continue

        label = int(label_dir.name)  # Convert folder name to integer label
        # Find all iris and fingerprint files for this label
        ips = sorted(label_dir.glob("*.jpg"))
        fps = sorted((fp_root / f"{label:03d}").glob("*.png"))
        # Ensure same number of iris and fingerprint samples
        assert len(ips) == len(
            fps
        ), f"Mismatch for label {label}: {len(ips)} vs {len(fps)}"

        n = len(ips)
        tn = math.floor(train_pct * n)  # Number of training samples
        vn = math.ceil(val_pct * n)  # Number of validation samples
        # Basic sanity checks
        assert tn > 0 and vn > 0 and (tn + vn) < n, "Invalid split proportions"

        # Generate random splits for iris and fingerprint independently
        iris_splits = make_splits(n, tn, vn)
        fp_splits = make_splits(n, tn, vn)

        # Pair each iris index with each fingerprint index in the same split
        for mode in ("train", "val", "test"):
            for i in iris_splits[mode]:
                for j in fp_splits[mode]:
                    assert (
                        int(ips[i][:-8]) == int(fps[j][:-9]) == label
                    ), "Mismatch for label"
                    samples[mode].append((ips[i], fps[j], label))

    return samples


class IrisFingerprintDataset(Dataset):
    """
    PyTorch Dataset for paired iris and fingerprint images.

    Each sample consists of a tuple ((iris_tensor, fp_tensor), label).

    Args:
        samples: List of tuples (iris_path, fingerprint_path, label).
        transform: Callable to apply to both images (e.g., normalization, augment).
    """

    def __init__(self, samples: List[Tuple[Path, Path, int]], transform=None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        # Total number of paired samples
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tuple[Image.Image, Image.Image], int]:
        # Retrieve file paths and label for this index
        iris_p, fp_p, label = self.samples[idx]
        # Load grayscale images
        iris = Image.open(iris_p).convert("L")
        fp = Image.open(fp_p).convert("L")
        # Apply shared transform if provided
        if self.transform:
            iris = self.transform(iris)
            fp = self.transform(fp)
        # Return paired images and label
        return (iris, fp), label
