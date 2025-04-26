import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from tqdm import tqdm

from last_q.data.data import FingerprintDataset, load_fp_only
from last_q.models.fp_net import FPNet

# Suppress warnings for cleaner console output
warnings.filterwarnings("ignore")


def main():
    fp_root = Path("../data/NIST301-augmented2")
    if not fp_root.exists():
        raise FileNotFoundError(f"Data path not found: {fp_root.absolute()}")

    results_dir = Path("results/fp")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = results_dir / "best_model2.pth"

    # Reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Data
    transform = transforms.ToTensor()
    train_pct = 0.85
    samples = load_fp_only(fp_root, train_pct)
    train_ds = FingerprintDataset(samples["train"], transform=transform)
    val_ds = FingerprintDataset(samples["val"], transform=transform)

    batch_size = 64
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # Device
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device = torch.device("mps" if mps_ok else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Model
    model = FPNet().to(device)

    # Summary
    fp_sample, _ = next(iter(train_loader))
    fp_sample = fp_sample.to(device)
    summary(model, input_data=(fp_sample,), col_names=("output_size", "num_params", "trainable"))
    
    # If FPNet produces embeddings, you would need a linear head for classification:
    num_classes = 100
    classifier = nn.Linear(128, num_classes)
    nn.init.kaiming_normal_(
        classifier.weight,
        mode='fan_out',              # recommended for Linear/Conv when followed by ReLU
        nonlinearity='relu'
    )
    nn.init.zeros_(classifier.bias)
    classifier = classifier.to(device)
    summary(classifier, input_size=(batch_size, 128), device=device)

    # Resume
    start_epoch = 1
    best_val_loss = float("inf")
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        classifier.load_state_dict(ckpt["classifier_state_dict"])
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        start_epoch = ckpt.get("epoch", start_epoch)
        print(f"→ Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4e}")
    checkpoint_path = results_dir / "best_model.pth"

    # Loss, optim, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier.parameters()), lr=1e-2
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10, min_lr=1e-5, verbose=True
    )

    # Training settings
    max_epochs = 1000
    patience = 30
    no_improve = 0

    train_losses, val_losses, val_accs = [], [], []

    for epoch in tqdm(range(start_epoch, max_epochs+1), desc="Epochs"):
        # ---- Train ----
        model.train()
        classifier.train()
        running_loss = 0.0
        for fp, labels in train_loader:
            fp, labels = fp.to(device), labels.to(device)
            emb = model(fp)
            logits = classifier(emb)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=1.0)
            clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_train = running_loss / len(train_loader)
        train_losses.append(avg_train)

        # ---- Validate ----
        model.eval()
        classifier.eval()
        running_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for fp, labels in val_loader:
                fp, labels = fp.to(device), labels.to(device)
                emb = model(fp)
                logits = classifier(emb)
                running_loss += criterion(logits, labels).item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val = running_loss / len(val_loader)
        acc = correct / total
        val_losses.append(avg_val)
        val_accs.append(acc)

        scheduler.step(avg_val)

        # Checkpointing
        status = ""
        if avg_val < best_val_loss:
            best_val_loss, no_improve = avg_val, 0
            torch.save({
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "model_state_dict": model.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, checkpoint_path)
            status = "→ saved"
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={avg_train:.4e} | "
            f"val_loss={avg_val:.4e} | "
            f"val_acc={acc:.2%} | "
            f"lr={optimizer.param_groups[0]['lr']:.1e} {status}"
        )

        if no_improve >= patience:
            print(f"No improvement for {patience} epochs, stopping.")
            break

    # Save history
    pd.DataFrame({
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_acc": val_accs
    }).to_csv(results_dir / "training_history.csv", index=False)


if __name__ == "__main__":
    main()
