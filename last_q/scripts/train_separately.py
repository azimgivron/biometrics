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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def train_fp(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
             max_epochs: int, device: torch.device) -> None:
    """
    Train fingerprint branch with triplet loss and evaluate using 1-NN accuracy.

    Args:
        model: FPNet instance to train.
        train_loader: DataLoader for training samples.
        val_loader:   DataLoader for validation samples.
        max_epochs:  Maximum number of epochs.
        device:      Device to run training on (cpu/cuda).
    """
    # Classifier head mapping 64-dim features → 32-dim embedding
    classifier = nn.Linear(64, 32, bias=False).to(device)
    # Combine model and classifier parameters for optimization
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad,
               list(model.parameters()) + list(classifier.parameters())),
        lr=1e-3,
    )
    # LR scheduler reduces lr on plateau of validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10,
        min_lr=1e-5, verbose=True
    )
    best_val_loss = float("inf")  # track best validation loss
    train_losses, val_losses, val_accuracies = [], [], []
    since_improve = 0  # epochs since last improvement
    patience = 20      # early stopping patience

    for epoch in tqdm(range(1, max_epochs+1), total=max_epochs, desc="FP Epochs"):
        # ---------- Training Phase ----------
        model.train(); classifier.train()
        total_train_loss = 0.0
        for (_, fp), labels in train_loader:
            fp, labels = fp.to(device), labels.to(device)
            # Forward pass: extract features and project
            emb = classifier(model(fp))
            # Mine hard pairs for triplet loss
            hard_pairs = miner(emb, labels)
            loss = loss_func(emb, labels, hard_pairs)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---------- Build Gallery Embeddings for 1-NN ----------
        model.eval(); classifier.eval()
        gallery_embs, gallery_lbls = [], []
        with torch.no_grad():
            for (_, fp), labels in train_loader:
                fp = fp.to(device)
                emb = classifier(model(fp))
                gallery_embs.append(emb.cpu())
                gallery_lbls.append(labels.cpu())
        gallery_embs = torch.cat(gallery_embs)
        gallery_lbls = torch.cat(gallery_lbls)

        # ---------- Validation Phase ----------
        total_val_loss = 0.0
        val_embs, val_lbls = [], []
        with torch.no_grad():
            for (_, fp), labels in val_loader:
                fp, labels = fp.to(device), labels.to(device)
                emb = classifier(model(fp))
                hard_pairs = miner(emb, labels)
                total_val_loss += loss_func(emb, labels, hard_pairs).item()
                val_embs.append(emb.cpu()); val_lbls.append(labels.cpu())
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_embs = torch.cat(val_embs); val_lbls = torch.cat(val_lbls)

        # Compute 1-NN accuracy on validation set
        dists = torch.cdist(val_embs, gallery_embs)  # pairwise distances
        nn_idx = dists.argmin(dim=1)
        preds = gallery_lbls[nn_idx]
        acc = (preds == val_lbls).float().mean().item()
        val_accuracies.append(acc)

        # ---------- Scheduler & Checkpointing ----------
        scheduler.step(avg_val_loss)
        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            since_improve = 0
            # Save best model checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
            }, CHECKPOINT_PATH_FP)
            status = "→ checkpoint saved"
        else:
            since_improve += 1

        # Log epoch metrics
        print(f"FP Epoch {epoch:02d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val 1-NN Acc: {acc:.2%} | {status}")

        # Early stopping
        if since_improve >= patience:
            print(f"No FP val improvement for {patience} epochs, stopping.")
            break

    # Save training history to CSV
    pd.DataFrame({
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_1nn_acc': val_accuracies,
    }).to_csv(Path(RESULTS_DIR) / "fp_training_history.csv", index=False)


def train_iris(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               max_epochs: int, device: torch.device) -> None:
    """
    Train iris branch with triplet loss and evaluate using 1-NN accuracy.

    Args:
        model: IrisNet instance to train.
        train_loader: DataLoader for training samples.
        val_loader:   DataLoader for validation samples.
        max_epochs:  Maximum number of epochs.
        device:      Device to run training on (cpu/cuda).
    """
    # Similar setup as train_fp, but operating on iris images
    classifier = nn.Linear(64, 32, bias=False).to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad,
               list(model.parameters()) + list(classifier.parameters())),
        lr=1e-3,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10,
        min_lr=1e-5, verbose=True
    )
    best_val_loss = float("inf")
    train_losses, val_losses, val_accuracies = [], [], []
    since_improve, patience = 0, 20

    for epoch in tqdm(range(1, max_epochs+1), total=max_epochs, desc="Iris Epochs"):
        # ---------- Training ----------
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

        # ---------- Build Gallery ----------
        model.eval(); classifier.eval()
        gallery_embs, gallery_lbls = [], []
        with torch.no_grad():
            for (iris, _), labels in train_loader:
                iris = iris.to(device)
                emb = classifier(model(iris))
                gallery_embs.append(emb.cpu()); gallery_lbls.append(labels.cpu())
        gallery_embs = torch.cat(gallery_embs); gallery_lbls = torch.cat(gallery_lbls)

        # ---------- Validation ----------
        total_val_loss = 0.0
        val_embs, val_lbls = [], []
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

        # ---------- Scheduler & Checkpoint ----------
        scheduler.step(avg_val_loss)
        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss; since_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, CHECKPOINT_PATH_IRIS)
            status = "→ checkpoint saved"
        else:
            since_improve += 1

        print(f"Iris Epoch {epoch:02d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val 1-NN Acc: {acc:.2%} | {status}")

        if since_improve >= patience:
            print(f"No Iris val improvement for {patience} epochs, stopping.")
            break

    # Persist iris training history
    pd.DataFrame({
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_1nn_acc': val_accuracies,
    }).to_csv(Path(RESULTS_DIR) / "iris_training_history.csv", index=False)


if __name__ == "__main__":
    # Root directory for results and checkpoints
    RESULTS_DIR = "results/from_scratch"
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH_FP = Path(RESULTS_DIR) / "best_fp.pth"
    CHECKPOINT_PATH_IRIS = Path(RESULTS_DIR) / "best_iris.pth"

    # Fix random seeds for reproducibility
    SEED = 42
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Prepare data transformations and loaders
    transform = transforms.ToTensor()
    iris_root = Path("../data/CASIA1-enhanced")
    fp_root   = Path("../data/NIST301-augmented")
    train_pct, val_pct = 0.7, 0.15
    samples = load(iris_root, fp_root, train_pct, val_pct)

    train_ds = IrisFingerprintDataset(samples['train'], transform=transform)
    val_ds   = IrisFingerprintDataset(samples['val'],   transform=transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    # Device selection: MPS, CUDA, or CPU
    mps_available = (hasattr(torch.backends, "mps")
                     and torch.backends.mps.is_available()
                     and torch.backends.mps.is_built())
    device = torch.device("mps" if mps_available else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Instantiate models
    iris_model  = IrisNet().to(device)
    fp_model    = FPNet().to(device)
    merger_model = MergerNet(emb_size=32).to(device)

    # Loss and miner for metric learning
    miner = miners.BatchEasyHardMiner(
        pos_strategy=miners.BatchEasyHardMiner.HARD,
        neg_strategy=miners.BatchEasyHardMiner.SEMIHARD,
    )
    loss_func = losses.TripletMarginLoss(margin=0.8)

    # Optimizer and scheduler for merger + branches
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad,
               list(merger_model.parameters()) +
               list(iris_model.parameters()) +
               list(fp_model.parameters())),
        lr=1e-3
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10,
        min_lr=1e-5, verbose=True
    )

    # ---------- Phase 1: Train individual branches ----------
    train_iris(iris_model, train_loader, val_loader, max_epochs=100, device=device)
    train_fp(fp_model,   train_loader, val_loader, max_epochs=100, device=device)

    # ---------- Phase 2: Train fusion network ----------
    max_epochs = 100
    patience = 20
    since_improve = 0
    best_val_loss = float("inf")
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in tqdm(range(1, max_epochs+1), total=max_epochs, desc="Fusion Epochs"):
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

        # ---------- Build gallery for 1-NN ----------
        merger_model.eval(); iris_model.eval(); fp_model.eval()
        gallery_embs, gallery_lbls = [], []
        with torch.no_grad():
            for (iris, fp), labels in train_loader:
                iris, fp = iris.to(device), fp.to(device)
                emb = merger_model(iris_model(iris), fp_model(fp))
                gallery_embs.append(emb.cpu()); gallery_lbls.append(labels.cpu())
        gallery_embs = torch.cat(gallery_embs); gallery_lbls = torch.cat(gallery_lbls)

        # ---------- Validation ----------
        total_val_loss = 0.0
        val_embs, val_lbls = [], []
        with torch.no_grad():
            for (iris, fp), labels in val_loader:
                iris, fp, labels = iris.to(device), fp.to(device), labels.to(device)
                emb = merger_model(iris_model(iris), fp_model(fp))
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

        # Scheduler and checkpointing
        scheduler.step(avg_val_loss)
        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss; since_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': merger_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, Path(RESULTS_DIR)/"best_fusion.pth")
            status = "→ checkpoint saved"
        else:
            since_improve += 1

        print(f"Fusion Epoch {epoch:02d} | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val 1-NN Acc: {acc:.2%} | {status}")
        if since_improve >= patience:
            print(f"No Fusion val improvement for {patience} epochs, stopping.")
            break

    # Save fusion training history
    pd.DataFrame({
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_1NN_acc': val_accuracies
    }).to_csv(Path(RESULTS_DIR)/"fusion_training_history.csv", index=False)