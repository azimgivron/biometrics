import random
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from tqdm import tqdm

from last_q.data.data import IrisFingerprintDataset, load
from last_q.models.model_from_pretrained import TwoBranchEfficientNet

# Suppress library warnings for cleaner logs
warnings.filterwarnings("ignore")


def main():
    """
    Train and validate a TwoBranchEfficientNet model on paired iris and fingerprint data.

    Workflow:
      1. Set random seeds for reproducibility.
      2. Prepare data transforms and split into train/val sets.
      3. Create DataLoaders.
      4. Instantiate model, loss, optimizer, and scheduler.
      5. Run training loop with validation and early stopping.
      6. Save best checkpoint and training history.
    """
    # ----- 1. Setup -----
    # Results directory and checkpoint path
    results_dir: Path = Path("results/pretrained")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path: Path = results_dir / "best_resnet_iris_fp.pth"

    # Fix random seeds for reproducibility
    SEED: int = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ----- 2. Data preparation -----
    # Simple transform: convert PIL Image to FloatTensor [0,1]
    transform = transforms.ToTensor()

    # Paths to enhanced iris and augmented fingerprint data
    iris_root: Path = Path("../data/CASIA1-enhanced")
    fp_root: Path = Path("../data/NIST301-augmented")
    # Fraction of samples for train and validation
    train_pct, val_pct = 0.7, 0.15
    samples = load(iris_root, fp_root, train_pct, val_pct)

    # Determine number of classes from loaded samples
    n_classes: int = len({lbl for _, _, lbl in samples['train'] + samples['val'] + samples['test']})

    # Create PyTorch datasets and loaders
    train_ds = IrisFingerprintDataset(samples['train'], transform=transform)
    val_ds   = IrisFingerprintDataset(samples['val'],   transform=transform)
    batch_size: int = 32
    train_loader: DataLoader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader: DataLoader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # ----- 3. Model, loss, optimizer, scheduler -----
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

    # Instantiate two-branch EfficientNet, outputting logits for n_classes
    model = TwoBranchEfficientNet(n_classes=n_classes, freeze_until=None).to(device)

    # Print model summary on a sample batch
    (iris_sample, fp_sample), _ = next(iter(train_loader))
    iris_sample, fp_sample = iris_sample.to(device), fp_sample.to(device)
    summary(
        model,
        input_data=(iris_sample, fp_sample),
        col_names=("output_size", "num_params", "trainable")
    )

    # Define loss criterion and optimizer
    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=1e-2)
    # Scheduler that reduces LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        min_lr=1e-5,
        verbose=True
    )

    # ----- 4. Training loop -----
    best_val_loss: float = float('inf')
    since_improve: int = 0
    patience: int = 20
    max_epochs: int = 100

    # History logging
    history: Dict[str, List[float]] = {
        'train_loss': [],
        'val_loss':   [],
        'val_acc':    []
    }

    for epoch in tqdm(range(1, max_epochs + 1), total=max_epochs, desc="Epochs"):
        # -- Training phase --
        model.train()
        total_train_loss: float = 0.0
        for (iris, fp), labels in train_loader:
            iris, fp, labels = iris.to(device), fp.to(device), labels.to(device)
            logits: Tensor = model(iris, fp)            # [B, n_classes]
            loss: Tensor = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_loss: float = total_train_loss / len(train_loader)
        history['train_loss'].append(train_loss)

        # -- Validation phase --
        model.eval()
        total_val_loss: float = 0.0
        correct: int = 0
        total: int = 0
        with torch.no_grad():
            for (iris, fp), labels in val_loader:
                iris, fp, labels = iris.to(device), fp.to(device), labels.to(device)
                logits: Tensor = model(iris, fp)
                loss: Tensor = criterion(logits, labels)
                total_val_loss += loss.item()

                preds: Tensor = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss: float = total_val_loss / len(val_loader)
        val_acc: float = correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # -- Scheduler & checkpointing --
        scheduler.step(val_loss)
        status: str = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            since_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            status = '→ saved'
        else:
            since_improve += 1

        # Log epoch metrics
        print(
            f"Epoch {epoch:02d}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4%}  "
            f"lr={optimizer.param_groups[0]['lr']:.4e}   {status}"
        )

        # Early stopping
        if since_improve >= patience:
            print(f"No improvement in {patience} epochs, stopping training.")
            break

    # ----- 5. Save history -----
    pd.DataFrame(history).to_csv(results_dir / "training_history.csv", index=False)


if __name__ == '__main__':
    main()
