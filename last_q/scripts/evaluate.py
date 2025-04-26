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
from torchinfo import summary

from last_q.data.data import IrisFingerprintDataset, load
from last_q.models.fp_net import FPNet
from last_q.models.iris_net import IrisNet
from last_q.models.fusion_net import FusionNet
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
    
    # Path to checkpoint and fixed random seed
    checkpoint_path = Path("results/fusion/best_model.pth")
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

    test_ds = IrisFingerprintDataset(samples["test"], transform=transform)

    # DataLoaders (no shuffling for evaluation)
    batch_size: int = 32
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

    # Model
    fp_model = FPNet().to(device)
    iris_model = IrisNet().to(device)
    fusion_model = FusionNet(n_classes=100, input_size=800).to(device)

    # Summary
    (iri_sample, fp_sample), _ = next(iter(test_loader))
    iri_sample, fp_sample = iri_sample.to(device), fp_sample.to(device)
    summary(fp_model, input_data=(fp_sample,))
    summary(iris_model, input_data=(iri_sample,))
    
    # If FPNet produces embeddings, you would need a linear head for classification:
    summary(fusion_model, input_size=((batch_size, 128), (batch_size, 672)), device=device)

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        iris_model.load_state_dict(ckpt["iris_model_state_dict"])
        fp_model.load_state_dict(ckpt["fp_model_state_dict"])
        fusion_model.load_state_dict(ckpt["fusion_model_state_dict"])
        best_val_loss = ckpt["best_val_loss"]
        start_epoch = ckpt["epoch"]
        print(f"→ Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4e}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    fusion_model.eval()
    iris_model.eval()
    fp_model.eval()
        
    preds: List[Tensor] = []
    test_labels: List[Tensor] = []
    correct, total = 0, 0
    with torch.no_grad():
        # Iterate over training and validation loaders
        for (iris, fp), labels in test_loader:
            # Move inputs to device
            iris, fp, labels = iris.to(device), fp.to(device), labels.to(device)
            # Compute embeddings
            fp_emb = fp_model(fp)
            iris_emb = iris_model(iris)
            logits = fusion_model(fp_emb, iris_emb)

            preds.append(logits)
            test_labels.append(labels)
            
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Accuracy: {acc*100:.3f}%")

    preds = torch.cat(preds, dim=0)         # shape [N, C]
    test_labels = torch.cat(test_labels, dim=0)  # shape [N]
    probs = torch.softmax(preds, dim=1)     # shape [N, C]
    probs_np = probs.cpu().numpy()          # shape [N, C]
    num_classes = probs_np.shape[1]
    col_names = [f"class_{i}_score" for i in range(num_classes)]

    results_df = pd.DataFrame(probs_np, columns=col_names)
    results_df["actual"] = test_labels.cpu().numpy()
    print(results_df.head())

    # 4) Save to CSV
    out_dir = Path("results/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_predictions_full_dist.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Saved full score distributions and labels to {out_path}")



if __name__ == "__main__":
    main()
