import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_metric_learning import losses, miners
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from last_q.data.data import IrisFingerprintDataset, load
from last_q.models.sep_nets import FPNet, IrisNet, MergerNet

# Suppress warnings for cleaner console output
warnings.filterwarnings("ignore")


def train_fp(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int,
    device: torch.device,
) -> None:
    """
    Train fingerprint branch with triplet loss and evaluate using 1-NN accuracy.

    Args:
        model: FPNet instance to train.
        train_loader: DataLoader for training samples.
        val_loader:   DataLoader for validation samples.
        max_epochs:  Maximum number of epochs to train.
        device:      Device to run training on (cpu/cuda).
    """
    # Linear projection from 64-D feature to 32-D embedding for metric learning
    classifier = nn.Linear(64, 32, bias=False).to(device)

    # Combine model + classifier parameters, filter only those requiring gradients
    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            list(model.parameters()) + list(classifier.parameters()),
        ),
        lr=1e-3,
    )

    # Scheduler: reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10, min_lr=1e-5, verbose=True
    )

    # Tracking variables
    best_val_loss = float("inf")
    train_losses, val_losses, val_accuracies = [], [], []
    since_improve = 0  # epochs since last validation improvement
    patience = 20      # early-stopping patience

    # Choose mining strategy and loss function for triplet loss
    miner = miners.BatchEasyHardMiner(
        pos_strategy=miners.BatchEasyHardMiner.HARD,
        neg_strategy=miners.BatchEasyHardMiner.SEMIHARD,
    )
    loss_func = losses.TripletMarginLoss(margin=0.8)

    for epoch in tqdm(range(1, max_epochs + 1), total=max_epochs, desc="FP Epochs"):
        # ----- Training Phase -----
        model.train()
        classifier.train()
        total_train_loss = 0.0
        for (_, fp), labels in train_loader:
            # Move data to device
            fp, labels = fp.to(device), labels.to(device)
            # Forward pass: feature extraction + projection
            emb = classifier(model(fp))
            # Mine hard triplets
            hard_pairs = miner(emb, labels)
            loss = loss_func(emb, labels, hard_pairs)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ----- Build Gallery for 1-NN -----
        model.eval()
        classifier.eval()
        gallery_embs, gallery_lbls = [], []
        with torch.no_grad():
            for (_, fp), labels in train_loader:
                emb = classifier(model(fp.to(device)))
                gallery_embs.append(emb.cpu())
                gallery_lbls.append(labels)
        gallery_embs = torch.cat(gallery_embs)
        gallery_lbls = torch.cat(gallery_lbls)

        # ----- Validation Phase -----
        total_val_loss = 0.0
        val_embs, val_lbls = [], []
        with torch.no_grad():
            for (_, fp), labels in val_loader:
                fp, labels = fp.to(device), labels.to(device)
                emb = classifier(model(fp))
                total_val_loss += loss_func(emb, labels, miner(emb, labels)).item()
                val_embs.append(emb.cpu())
                val_lbls.append(labels.cpu())
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_embs = torch.cat(val_embs)
        val_lbls = torch.cat(val_lbls)

        # Compute 1-NN accuracy using gallery
        dists = torch.cdist(val_embs, gallery_embs)
        preds = gallery_lbls[dists.argmin(dim=1)]
        acc = (preds == val_lbls).float().mean().item()
        val_accuracies.append(acc)

        # ----- Scheduler & Checkpoint -----
        scheduler.step(avg_val_loss)
        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            since_improve = 0
            # Save best checkpoint
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                },
                CHECKPOINT_PATH_FP,
            )
            status = "→ checkpoint saved"
        else:
            since_improve += 1

        # Log metrics for this epoch
        print(
            f"FP Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val 1-NN Acc: {acc:.2%} | {status}"
        )

        # Early stopping check
        if since_improve >= patience:
            print(f"No FP val improvement for {patience} epochs, stopping.")
            break

    # Save training history to CSV for later analysis
    pd.DataFrame(
        {"train_loss": train_losses, "val_loss": val_losses, "val_1nn_acc": val_accuracies}
    ).to_csv(Path(RESULTS_DIR) / "fp_training_history.csv", index=False)


def train_iris(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int,
    device: torch.device,
) -> None:
    """
    Train iris branch with triplet loss and evaluate using 1-NN accuracy.

    Args:
        model: IrisNet instance to train.
        train_loader: DataLoader for training samples.
        val_loader:   DataLoader for validation samples.
        max_epochs:  Maximum number of epochs to train.
        device:      Device to run training on (cpu/cuda).
    """
    # Linear projection head
    classifier = nn.Linear(64, 32, bias=False).to(device)
    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            list(model.parameters()) + list(classifier.parameters()),
        ),
        lr=1e-3,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10, min_lr=1e-5, verbose=True
    )
    best_val_loss = float("inf")
    train_losses, val_losses, val_accuracies = [], [], []
    since_improve, patience = 0, 20

    miner = miners.BatchEasyHardMiner(
        pos_strategy=miners.BatchEasyHardMiner.HARD,
        neg_strategy=miners.BatchEasyHardMiner.SEMIHARD,
    )
    loss_func = losses.TripletMarginLoss(margin=0.8)

    for epoch in tqdm(range(1, max_epochs + 1), total=max_epochs, desc="Iris Epochs"):
        # Training loop identical to train_fp but on iris images
        model.train(); classifier.train()
        total_train_loss = 0.0
        for (iris, _), labels in train_loader:
            iris, labels = iris.to(device), labels.to(device)
            emb = classifier(model(iris))
            hard_pairs = miner(emb, labels)
            loss = loss_func(emb, labels, hard_pairs)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Build gallery for 1-NN
        model.eval(); classifier.eval()
        gallery_embs, gallery_lbls = [], []
        with torch.no_grad():
            for (iris, _), labels in train_loader:
                emb = classifier(model(iris.to(device)))
                gallery_embs.append(emb.cpu()); gallery_lbls.append(labels)
        gallery_embs = torch.cat(gallery_embs); gallery_lbls = torch.cat(gallery_lbls)

        # Validation step
        total_val_loss = 0.0; val_embs, val_lbls = [], []
        with torch.no_grad():
            for (iris, _), labels in val_loader:
                iris, labels = iris.to(device), labels.to(device)
                emb = classifier(model(iris))
                total_val_loss += loss_func(emb, labels, miner(emb, labels)).item()
                val_embs.append(emb.cpu()); val_lbls.append(labels.cpu())
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_embs = torch.cat(val_embs); val_lbls = torch.cat(val_lbls)

        # Compute 1-NN accuracy
        dists = torch.cdist(val_embs, gallery_embs)
        preds = gallery_lbls[dists.argmin(dim=1)]
        acc = (preds == val_lbls).float().mean().item()
        val_accuracies.append(acc)

        # Scheduler & checkpointing
        scheduler.step(avg_val_loss)
        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss; since_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()},
                CHECKPOINT_PATH_IRIS,
            )
            status = "→ checkpoint saved"
        else:
            since_improve += 1

        print(
            f"Iris Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val 1-NN Acc: {acc:.2%} | {status}"
        )
        if since_improve >= patience:
            print(f"No Iris val improvement for {patience} epochs, stopping.")
            break

    # Persist iris training metrics\    
    pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses, "val_1nn_acc": val_accuracies}).to_csv(
        Path(RESULTS_DIR) / "iris_training_history.csv", index=False
    )


if __name__ == "__main__":
    # Constants for results and checkpoints
    RESULTS_DIR = "results/from_scratch"
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH_FP = Path(RESULTS_DIR) / "best_fp.pth"
    CHECKPOINT_PATH_IRIS = Path(RESULTS_DIR) / "best_iris.pth"

    # Reproducibility seeds
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Data loading and transforms setup
    transform = transforms.ToTensor()
    iris_root = Path("../data/CASIA1-enhanced")  # enhanced iris data folder
    fp_root = Path("../data/NIST301-augmented")  # augmented fingerprint data folder

    # Split percentages for train/validation
    train_pct, val_pct = 0.7, 0.15
    samples = load(iris_root, fp_root, train_pct, val_pct)

    # Build dataset and dataloader objects
    train_ds = IrisFingerprintDataset(samples["train"], transform=transform)
    val_ds = IrisFingerprintDataset(samples["val"], transform=transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Select device: MPS, CUDA, or CPU
    mps_available = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built()
    )
    device = torch.device("mps" if mps_available else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Instantiate model branches and merger
    iris_model = IrisNet().to(device)
    fp_model = FPNet().to(device)
    merger_model = MergerNet(emb_size=32).to(device)

    # Shared miner and loss for metric learning
    miner = miners.BatchEasyHardMiner(
        pos_strategy=miners.BatchEasyHardMiner.HARD,
        neg_strategy=miners.BatchEasyHardMiner.SEMIHARD,
    )
    loss_func = losses.TripletMarginLoss(margin=0.8)

    # Optimizer and scheduler for fusion training
    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            list(merger_model.parameters()) + list(iris_model.parameters()) + list(fp_model.parameters()),
        ),
        lr=1e-3,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10, min_lr=1e-5, verbose=True
    )

    # ----- Phase 1: Train branches separately -----
    train_iris(iris_model, train_loader, val_loader, max_epochs=100, device=device)
    train_fp(fp_model, train_loader, val_loader, max_epochs=100, device=device)

    # ----- Phase 2: Train fusion network -----
    max_epochs = 100
    patience = 20
    since_improve = 0
    best_val_loss = float("inf")
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in tqdm(range(1, max_epochs + 1), total=max_epochs, desc="Fusion Epochs"):
        # Training pass for fusion
        merger_model.train(); iris_model.train(); fp_model.train()
        total_loss = 0.0
        for (iris, fp), labels in train_loader:
            iris, fp, labels = iris.to(device), fp.to(device), labels.to(device)
            emb = merger_model(iris_model(iris), fp_model(fp))
            hard_pairs = miner(emb, labels)
            loss = loss_func(emb, labels, hard_pairs)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Build gallery embeddings
        merger_model.eval(); iris_model.eval(); fp_model.eval()
        gallery_embs, gallery_lbls = [], []
        with torch.no_grad():
            for (iris, fp), labels in train_loader:
                emb = merger_model(iris_model(iris.to(device)), fp_model(fp.to(device)))
                gallery_embs.append(emb.cpu()); gallery_lbls.append(labels)
        gallery_embs = torch.cat(gallery_embs); gallery_lbls = torch.cat(gallery_lbls)

        # Validation pass
        total_val_loss = 0.0; val_embs, val_lbls = [], []
        with torch.no_grad():
            for (iris, fp), labels in val_loader:
                iris, fp, labels = iris.to(device), fp.to(device), labels.to(device)
                emb = merger_model(iris_model(iris), fp_model(fp))
                total_val_loss += loss_func(emb, labels, miner(emb, labels)).item()
                val_embs.append(emb.cpu()); val_lbls.append(labels.cpu())
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_embs = torch.cat(val_embs); val_lbls = torch.cat(val_lbls)

        # 1-NN accuracy calculation
        dists = torch.cdist(val_embs, gallery_embs)
        preds = gallery_lbls[dists.argmin(dim=1)]
        acc = (preds == val_lbls).float().mean().item()
        val_accuracies.append(acc)

        # Scheduler & checkpoint logic
        scheduler.step(avg_val_loss)
        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss; since_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": merger_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, Path(RESULTS_DIR) / "best_fusion.pth")
            status = "→ checkpoint saved"
        else:
            since_improve += 1

        print(
            f"Fusion Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val 1-NN Acc: {acc:.2%} | {status}"
        )
        if since_improve >= patience:
            print(f"No Fusion val improvement for {patience} epochs, stopping.")
            break

    # Save fusion training history CSV
    pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses, "val_1NN_acc": val_accuracies}).to_csv(
        Path(RESULTS_DIR) / "fusion_training_history.csv", index=False
    )
