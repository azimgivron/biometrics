import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from last_q.data.augment_fp import (AddGaussianNoise, Clamp, EnhanceFP,
                                    RandomThinning, augment_variants, read_DB)
from last_q.data.enhanced_iris import process_and_save, read_iris_db_paths
from last_q.src.fingerprint_enhancer import FingerprintImageEnhancer


def main():
    parser = argparse.ArgumentParser(
        description="Augment and save NIST301 fingerprint images and CASIA1 iris images"
    )
    parser.add_argument(
        "casia1_dir", type=Path, help="Path to the CASIA1 dataset folder"
    )
    parser.add_argument(
        "nist301_dir", type=Path, help="Path to the NIST301 dataset folder"
    )
    args = parser.parse_args()

    fp_path: Path = args.nist301_dir
    if not fp_path.exists():
        parser.error(f"Dataset path not found: {fp_path}")

    # Fix random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Load dataset
    fp_images, fp_labels = read_DB(str(fp_path))
    print(len(fp_labels))

    # Build augmentation pipeline
    train_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=180,
                translate=(0.35, 0.35),
                scale=(0.9, 1.6),
                shear=20,
                fill=255,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            AddGaussianNoise(0.0, 0.1),
            Clamp(),
            EnhanceFP(fe=FingerprintImageEnhancer()),
            RandomThinning(p=0.4, threshold=128, max_thickness=5),
        ]
    )

    num_augs = 6
    args_list = [(img, num_augs, train_transforms) for img in fp_images]

    # Generate augmented variants in parallel
    with ProcessPoolExecutor() as executor:
        results_iter = executor.map(augment_variants, args_list)
        augmented_images = list(
            tqdm(results_iter, total=len(fp_images), desc="FP augmentation")
        )

    # Prepare output directory
    generated_root = fp_path.parent / f"{fp_path.name}-augmented"
    generated_root.mkdir(parents=True, exist_ok=True)

    # Save originals and augmentations
    fe = FingerprintImageEnhancer()
    for orig_img, label, variants in tqdm(
        zip(fp_images, fp_labels, augmented_images), total=len(fp_labels), desc=f"Saving at {generated_root}"
    ):
        label_dir = generated_root / label
        label_dir.mkdir(exist_ok=True)

        # Save enhanced original
        enhanced = fe.enhance(orig_img)
        mask = fe._mask
        out = (enhanced * mask).astype(np.uint8)
        Image.fromarray(out, mode="L").save(label_dir / f"{label}_orig.png")

        # Save each augmented variant
        for idx, var in enumerate(variants, start=1):
            arr = var if var.dtype == np.uint8 else (var * 255).astype(np.uint8)
            Image.fromarray(arr, mode="L").save(label_dir / f"{label}_aug{idx}.png")

    iris_data_path = args.casia1_dir
    if not iris_data_path.exists():
        parser.error(f"Input path not found: {iris_data_path}")

    output_path = iris_data_path.parent / f"{iris_data_path.name}-enhanced"
    output_path.mkdir(parents=True, exist_ok=True)

    params = {
        "eyelashes_thres": 40,
        "radial_res": 200,
        "angular_res": 500,
    }

    items = read_iris_db_paths(iris_data_path)

    results = []
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_and_save, idx, item, output_path, params): idx
            for idx, item in enumerate(items)
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"Processing iris images and saving at {output_path}"
        ):
            results.append(future.result())

    print(f"Processing done !")


if __name__ == "__main__":
    main()
