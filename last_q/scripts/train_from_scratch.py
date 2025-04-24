import random
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from pytorch_metric_learning import losses, miners
from torch import Tensor, optim
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from tqdm import tqdm

from last_q.data.data import IrisFingerprintDataset, load
from last_q.models.model_from_scratch import TwoBranchScratchNet

# Suppress warnings for cleaner console output
warnings.filterwarnings("ignore")


def main():
    iris_root = Path("../data/CASIA1-enhanced") # might to be adjusted
    fp_root = Path("../data/NIST301-augmented")
    
    if not iris_root.exists():
        raise FileNotFoundError(f"Iris data path not found: {iris_root.absolute()}")
    if not fp_root.exists():
        raise FileNotFoundError(f"Fingerprint data path not found: {fp_root.absolute()}")

    # Directory setup for results and checkpoints
    results_dir = Path("results/from_scratch")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = results_dir / "best_from_scratch.pth"

    # Fix random seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---------- 0. Data transforms ----------
    transform = transforms.ToTensor()  # convert PIL images to FloatTensor [0,1]
    train_pct, val_pct = 0.7, 0.15
    samples = load(iris_root, fp_root, train_pct, val_pct)

    # Create PyTorch Datasets and DataLoaders
    train_ds = IrisFingerprintDataset(samples["train"], transform=transform)
    val_ds = IrisFingerprintDataset(samples["val"], transform=transform)
    batch_size = 32
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # ---------- 2. Model, loss, optimizer, scheduler ----------
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

    # Instantiate the two-branch network
    model = TwoBranchScratchNet(emb_size=64).to(device)

    # Print model summary on a batch of data
    (iris_sample, fp_sample), _ = next(iter(train_loader))
    iris_sample, fp_sample = iris_sample.to(device), fp_sample.to(device)
    summary(
        model,
        input_data=(iris_sample, fp_sample),
        col_names=("output_size", "num_params", "trainable"),
    )

    # Metric learning components
    miner = miners.BatchEasyHardMiner(
        pos_strategy=miners.BatchEasyHardMiner.SEMIHARD,
        neg_strategy=miners.BatchEasyHardMiner.HARD,
    )
    loss_func = losses.TripletMarginLoss(margin=0.7)  # triplet loss

    # Optionally resume from checkpoint
    start_epoch = 1
    best_val_loss = float("inf")
    if checkpoint_path.exists():
        try:
            print(f"→ Loading checkpoint from {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
            start_epoch = ckpt.get("epoch", start_epoch)
            print(
                f"   Resuming from epoch {start_epoch} with best_val_loss={best_val_loss:.4e}"
            )
        except RuntimeError:
            pass
    
    # Optimizer only updates trainable params
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=25, min_lr=1e-5, verbose=True
    )

    # Training hyperparameters
    max_epochs = 100
    patience = 60
    since_improve = 0

    # Logs
    train_losses: List[float] = []
    val_losses: List[float] = []
    val_accs: List[float] = []

    # ---------- 3. Train & validate with 1-NN ----------
    for epoch in tqdm(
        range(start_epoch, max_epochs + 1), total=max_epochs, desc="Epochs"
    ):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        for (iris, fp), labels in train_loader:
            iris, fp, labels = iris.to(device), fp.to(device), labels.to(device)
            emb: Tensor = model(iris, fp)  # get embeddings
            hard_pairs = miner(emb, labels)  # mine hard triplets
            loss = loss_func(emb, labels, hard_pairs)  # compute triplet loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Build gallery embeddings for 1-NN ---
        model.eval()
        gallery_embs: List[Tensor] = []
        gallery_lbls: List[Tensor] = []
        with torch.no_grad():
            for (iris, fp), labels in train_loader:
                iris, fp = iris.to(device), fp.to(device)
                emb = model(iris, fp)
                gallery_embs.append(emb.cpu())
                gallery_lbls.append(labels.cpu())
        gallery_embs = torch.cat(gallery_embs)
        gallery_lbls = torch.cat(gallery_lbls)

        # --- Validation Phase ---
        total_val_loss = 0.0
        val_embs: List[Tensor] = []
        val_lbls: List[Tensor] = []
        with torch.no_grad():
            for (iris, fp), labels in val_loader:
                iris, fp, labels = iris.to(device), fp.to(device), labels.to(device)
                emb = model(iris, fp)
                hard_pairs = miner(emb, labels)
                total_val_loss += loss_func(emb, labels, hard_pairs).item()
                val_embs.append(emb.cpu())
                val_lbls.append(labels.cpu())
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_embs = torch.cat(val_embs)
        val_lbls = torch.cat(val_lbls)

        # Compute 1-NN accuracy
        dists = torch.cdist(val_embs, gallery_embs)  # pairwise distances
        nn_idx = dists.argmin(dim=1)
        preds = gallery_lbls[nn_idx]
        acc = (preds == val_lbls).float().mean().item()
        val_accs.append(acc)

        # --- Scheduler & Checkpointing ---
        scheduler.step(avg_val_loss)
        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            since_improve = 0
            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                checkpoint_path,
            )
            status = "→ saved"
        else:
            since_improve += 1

        # Log epoch metrics
        print(
            f"\nEpoch {epoch:02d}  "
            f"train_loss={avg_train_loss:.4e}  "
            f"val_loss={avg_val_loss:.4e}  "
            f"val_1nn_acc={acc:.2%}   "
            f"lr={optimizer.param_groups[0]['lr']:.4e}   {status}"
        )

        # Early stopping
        if since_improve >= patience:
            print(f"No val improvement for {patience} epochs—stopping.")
            break

    # Persist training history to CSV
    history = pd.DataFrame(
        {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_1nn_acc": val_accs,
        }
    )
    history.to_csv(results_dir / "training_history.csv", index=False)


if __name__ == "__main__":
    main()
