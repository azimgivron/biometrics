import random
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms

from last_q.data.data import load, IrisFingerprintDataset
from last_q.models.model_from_pretrained import TwoBranchEfficientNet

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    """
    Load a pretrained TwoBranchEfficientNet and perform inference on the test set.

    Steps:
      1. Set random seeds and determine compute device.
      2. Load test-only DataLoader from pre-specified train/val/test splits.
      3. Instantiate model, load checkpoint, and set to eval mode.
      4. Run inference to collect logits and labels.
      5. Compute accuracy, softmax confidences, and save results to CSV.
    """
    # ----- 1. Reproducibility & Device Setup -----
    SEED: int = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Select device: MPS > CUDA > CPU
    mps_ok: bool = (
        hasattr(torch.backends, 'mps') and
        torch.backends.mps.is_available() and
        torch.backends.mps.is_built()
    )
    device: torch.device = torch.device(
        'mps' if mps_ok else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f"Using device: {device}")

    # ----- 2. DataLoader for Test Set -----
    transform = transforms.ToTensor()  # convert PIL images to FloatTensor [0,1]
    iris_root: Path = Path("../data/CASIA1-enhanced")
    fp_root: Path   = Path("../data/NIST301-augmented")
    train_pct, val_pct = 0.7, 0.15
    samples = load(iris_root, fp_root, train_pct, val_pct)

    # Build only test dataset and loader
    test_ds: IrisFingerprintDataset = IrisFingerprintDataset(
        samples['test'], transform=transform
    )
    test_loader: DataLoader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"Test samples: {len(test_ds)}")

    # ----- 3. Model & Checkpoint Loading -----
    n_classes: int = 100  # total number of identity classes
    model = TwoBranchEfficientNet(n_classes=n_classes, freeze_until=None).to(device)
    checkpoint_path: Path = Path("results/pretrained/best_net_iris_fp.pth")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()  # set model to evaluation mode

    # ----- 4. Inference Loop -----
    all_logits: List[Tensor] = []
    all_labels: List[Tensor] = []
    with torch.no_grad():
        for (iris, fp), labels in test_loader:
            iris, fp = iris.to(device), fp.to(device)
            logits = model(iris, fp)             # [B, n_classes]
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    # Concatenate batch-wise tensors
    all_logits_tensor: Tensor = torch.cat(all_logits, dim=0)  # [N_test, n_classes]
    all_labels_tensor: Tensor = torch.cat(all_labels, dim=0)  # [N_test]
    print("Logits shape:", all_logits_tensor.shape)
    print("Labels shape:", all_labels_tensor.shape)

    # ----- 5. Compute Predictions and Save -----
    preds: Tensor = all_logits_tensor.argmax(dim=1)           # predicted class indices
    accuracy: float = (preds == all_labels_tensor).float().mean().item()
    print(f"Test accuracy: {accuracy:.2%}")

    # Compute softmax probabilities and confidence per sample
    probs: Tensor = torch.softmax(all_logits_tensor, dim=1)
    confidences: np.ndarray = probs[range(len(preds)), preds].numpy()

    # Build DataFrame of results
    results_df: pd.DataFrame = pd.DataFrame({
        'predicted': preds.numpy(),
        'actual':    all_labels_tensor.numpy(),
        'confidence': confidences
    })
    out_path: Path = Path("results/pretrained") / "test_predictions.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


if __name__ == '__main__':
    main()