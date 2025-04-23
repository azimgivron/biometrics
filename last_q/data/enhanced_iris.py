import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .src.irismodules.fnc import normalize, segment

# Suppress warnings from underlying libraries
warnings.filterwarnings("ignore")


def read_iris_db_paths(path: Path) -> List[Tuple[Path, int]]:
    """
    Recursively collect all .jpg image paths under `path`, returning
    a list of (image_path, label) tuples.

    Args:
        path: Root directory of the iris database, organized by numeric subfolders.

    Returns:
        List of tuples: (Path to image file, integer label).
    """
    items: List[Tuple[Path, int]] = []
    # Walk through directory tree
    for dirpath, _, filenames in os.walk(path):
        label_str = os.path.basename(dirpath)
        # Skip non-numeric directories
        if not label_str.isdigit():
            continue
        label = int(label_str)
        # Collect all .jpg files in this label directory
        for fname in filenames:
            if not fname.lower().endswith(".jpg"):
                continue
            items.append((Path(dirpath) / fname, label))
    return items


def segment_iris(
    img: np.ndarray, radial_res: int, angular_res: int, eyelashes_thres: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment and normalize an iris image into polar coordinates.

    Args:
        img: Grayscale iris image array.
        radial_res: Radial resolution for normalization.
        angular_res: Angular resolution for normalization.
        eyelashes_thres: Threshold to detect and mask eyelashes.

    Returns:
        polar: Normalized iris in polar coordinates.
        mask: Boolean mask where False indicates invalid (masked) regions.
    """
    # Perform circular segmentation (iris & pupil detection)
    ciriris, cirpupil, im_with_noise = segment.segment(
        img, eyelashes_thres=eyelashes_thres
    )
    # Unwrap to polar coordinates and generate mask
    polar, mask = normalize.normalize(
        im_with_noise,
        ciriris[1],
        ciriris[0],
        ciriris[2],
        cirpupil[1],
        cirpupil[0],
        cirpupil[2],
        radial_res,
        angular_res,
    )
    # mask == 0 indicates regions to exclude
    return polar, (mask == 0)


def get_filter_bank(
    ksize: int = 5,
    sigma: float = 4.0,
    theta_range: np.ndarray = np.arange(0, np.pi, np.pi / 16),
    lambd: float = 10.0,
    gamma: float = 0.5,
    psi: float = 0.0,
) -> List[np.ndarray]:
    """
    Create a bank of Gabor filters at multiple orientations.

    Args:
        ksize: Kernel size (square) for each Gabor kernel.
        sigma: Standard deviation of the Gaussian envelope.
        theta_range: Array of orientations (radians).
        lambd: Wavelength of the sinusoidal factor.
        gamma: Spatial aspect ratio.
        psi: Phase offset.

    Returns:
        List of normalized Gabor kernels as NumPy arrays.
    """
    filters: List[np.ndarray] = []
    for theta in theta_range:
        kern = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F
        )
        # Normalize kernel energy
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def enhance_iris(img: np.ndarray) -> np.ndarray:
    """
    Enhance iris texture by applying a Gabor filter bank.

    Args:
        img: Polar-form iris image array.

    Returns:
        Enhanced image array (uint8).
    """
    # Apply each filter and stack responses
    filters = get_filter_bank()
    responses = np.stack(
        [cv2.filter2D(img, ddepth=-1, kernel=k) for k in filters], axis=0
    )
    # Normalize responses to [0,255]
    responses = cv2.normalize(responses, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Use maximum response across orientations
    enhanced = np.max(responses, axis=0)
    return enhanced.astype(np.uint8)


def process_and_save(
    item_idx: int, item: Tuple[Path, int], output_base: Path, params: Dict[str, float]
) -> str:
    """
    Worker function to segment, enhance, mask, and save one iris image.

    Args:
        item_idx: Index of the item (used to assign session filenames).
        item: Tuple (image_path, label).
        output_base: Base directory to write enhanced images.
        params: Dictionary with keys 'radial_res', 'angular_res', 'eyelashes_thres'.

    Returns:
        Path string of the saved enhanced image.
    """
    img_path, label = item
    # Read grayscale image
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    # Segment and normalize to polar form
    polar, mask = segment_iris(
        img_gray, params["radial_res"], params["angular_res"], params["eyelashes_thres"]
    )
    # Enhance iris texture
    enhanced = enhance_iris(polar)
    # Zero out masked regions
    enhanced[~mask] = 0

    # Build label-specific output directory
    label_dir = output_base / f"{label:03d}"
    label_dir.mkdir(parents=True, exist_ok=True)
    # Cycle item_idx through pseudo-sessions for filename uniqueness
    session = 1 if (item_idx % 7) < 3 else 2
    idx_in_session = (item_idx % 3) + 1 if session == 1 else (item_idx % 4) + 1
    out_fname = f"{label:03d}_{session}_{idx_in_session}.jpg"
    out_path = label_dir / out_fname
    # Save result
    Image.fromarray(enhanced).save(out_path)
    return str(out_path)


if __name__ == "__main__":
    # Base data directories
    data_path = Path("../data")
    assert data_path.exists(), f"Data path not found: {data_path}"
    iris_data_path = data_path / "CASIA1"
    output_path = data_path / "CASIA1-enhanced"
    output_path.mkdir(parents=True, exist_ok=True)

    # Parameters for segmentation and normalization
    params: Dict[str, float] = {
        "eyelashes_thres": 40,
        "radial_res": 200,
        "angular_res": 500,
    }

    # Read all image paths and labels
    items = read_iris_db_paths(iris_data_path)

    # Process images in parallel, collecting output paths
    results: List[str] = []
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_and_save, idx, item, output_path, params): idx
            for idx, item in enumerate(items)
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing"
        ):
            outpath = future.result()
            results.append(outpath)

    print(f"Processed {len(results)} images to {output_path}.")
