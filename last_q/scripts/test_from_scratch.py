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

from last_q.data.data import IrisFingerprintDataset, load
from last_q.models.model_from_scratch import TwoBranchScratchNet

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    """
    Evaluate a trained TwoBranchScratchNet on test data using k-NN retrieval.

    Steps:
      1) Load test, train, and validation sets.
      2) Build gallery embeddings from train+val.
      3) Compute embeddings for test set.
      4) Perform k-NN search and majority-vote predictions.
      5) Save predictions and confidence scores to CSV.
    """
    iris_root = Path("../data/CASIA1-enhanced")
    fp_root = Path("../data/NIST301-augmented")
    
    # ----- 1. Setup and data -----
    # Path to checkpoint and fixed random seed
    checkpoint_path: str = "results/from_scratch/best_from_scratch.pth"
    SEED: int = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load data splits
    transform = transforms.ToTensor()
    train_pct, val_pct = 0.7, 0.15
    samples = load(iris_root, fp_root, train_pct, val_pct)

    # Create datasets for train, validation, and test
    train_ds = IrisFingerprintDataset(samples["train"], transform=transform)
    val_ds = IrisFingerprintDataset(samples["val"], transform=transform)
    test_ds = IrisFingerprintDataset(samples["test"], transform=transform)

    # DataLoaders (no shuffling for evaluation)
    batch_size: int = 32
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Select device: MPS > CUDA > CPU
    mps_ok = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
    device = torch.device(
        "mps" if mps_ok else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Load model checkpoint
    model = TwoBranchScratchNet(emb_size=32).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()  # set to evaluation mode

    # ----- 2. Build gallery embeddings -----
    gallery_embeddings: List[Tensor] = []
    gallery_labels: List[Tensor] = []
    with torch.no_grad():
        # Iterate over training and validation loaders
        for loader in (train_loader, val_loader):
            for (iris, fp), labels in loader:
                # Move inputs to device
                iris, fp = iris.to(device), fp.to(device)
                # Compute embeddings
                emb = model(iris, fp)  # shape [B, D]
                gallery_embeddings.append(emb.cpu())
                gallery_labels.append(labels)
    # Concatenate all gallery embeddings and labels
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)
    gallery_labels = torch.cat(gallery_labels, dim=0)

    # ----- 3. Compute test embeddings -----
    test_embeddings: List[Tensor] = []
    test_labels: List[Tensor] = []
    with torch.no_grad():
        for (iris, fp), labels in test_loader:
            iris, fp = iris.to(device), fp.to(device)
            emb = model(iris, fp)
            test_embeddings.append(emb.cpu())
            test_labels.append(labels)
    test_embeddings = torch.cat(test_embeddings, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    # ----- 4. k-NN search and majority-vote -----
    # Compute pairwise distances [N_test, N_gallery]
    dists = torch.cdist(test_embeddings, gallery_embeddings)
    k: int = 6
    # Find k smallest distances and their indices
    knn_dists, knn_inds = torch.topk(dists, k, largest=False)
    # Retrieve labels of k nearest neighbors
    knn_labels = gallery_labels[knn_inds]  # shape [N_test, k]

    preds: List[int] = []
    confidences: List[float] = []
    # For each test sample, perform majority vote
    for neighbors in knn_labels:
        vals, counts = np.unique(neighbors.numpy(), return_counts=True)
        # Most frequent label among k neighbors
        pred = vals[counts.argmax()]
        preds.append(pred)
        # Confidence as fraction of neighbors agreeing
        confidences.append(counts.max() / k)

    preds = np.array(preds)
    confidences = np.array(confidences)

    # Compute overall accuracy
    accuracy = (preds == test_labels.numpy()).mean()
    print(f"6-NN accuracy on test set: {accuracy:.4%}")

    # ----- 5. Save results -----
    results_df = pd.DataFrame(
        {"predicted": preds, "confidence": confidences, "actual": test_labels.numpy()}
    )
    out_path = Path("results/from_scratch") / "test_predictions.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Saved predictions and confidences to {out_path}")


if __name__ == "__main__":
    main()
