import random
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchinfo import summary

from last_q.data.data import IrisFingerprintDataset, load
from last_q.models.fp_net import FPNet
from last_q.models.iris_net import IrisNet
from last_q.models.fusion_net import FusionNet

warnings.filterwarnings("ignore")


def get_device() -> torch.device:
    """
    Select the best available device: MPS > CUDA > CPU.

    Returns:
        torch.device: The selected device.
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict:
    """
    Load a model checkpoint from disk into CPU/GPU memory.

    Args:
        path (Path): Path to the .pth checkpoint file.
        device (torch.device): Device for loading the checkpoint.

    Returns:
        dict: The loaded checkpoint dictionary.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    return torch.load(path, map_location=device)


def init_models(device: torch.device,
                num_classes: int = 100,
                fusion_input: int = 800) -> dict:
    """
    Initialize the iris, fingerprint, and fusion models plus standalone classifiers.

    Args:
        device (torch.device): Device to move models to.
        num_classes (int): Number of output classes for classifiers.
        fusion_input (int): Combined feature-size input for fusion network.

    Returns:
        dict: A mapping of model names to instantiated PyTorch modules.
    """
    return {
        "iris": IrisNet().to(device),
        "iris_fusion": IrisNet().to(device),
        "fp": FPNet().to(device),
        "fp_fusion": FPNet().to(device),
        "fusion": FusionNet(n_classes=num_classes, input_size=fusion_input).to(device),
        "iris_cls": nn.Linear(672, num_classes).to(device),
        "fp_cls": nn.Linear(128, num_classes).to(device),
    }


def evaluate(model: str,
             loader: DataLoader,
             device: torch.device,
             out_path: Path,
             models: Dict[str, Any]) -> None:
    """
    Evaluate a model+head on the test loader, compute accuracy, and save probabilities.

    Args:
        model (str): Model type.
        loader (DataLoader): DataLoader for test data.
        device (torch.device): Device for computation.
        out_path (Path): File path to save the CSV of probabilities and labels.
        models (Dict[str, Any]): The models.
    """
    preds, labels = [], []
    correct = total = 0

    with torch.no_grad():
        for (iris, fp), y in loader:
            iris, fp, y = iris.to(device), fp.to(device), y.to(device)

            # Compute embeddings
            if model == "fusion":
                fp_emb = models["fp_fusion"](fp)
                iris_emb = models["iris_fusion"](iris)
                logits = models["fusion"](fp_emb, iris_emb)
            elif model == "fp":
                emb = models["fp"](fp)
                logits = models["fp_cls"](emb)
            else:  # iris model
                emb = models["iris"](iris)
                logits = models["iris_cls"](emb)
            
            preds.append(logits)
            labels.append(y)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"{model.capitalize()} Accuracy: {acc * 100:.3f}%")

    probs = torch.softmax(torch.cat(preds), dim=1).cpu().numpy()
    df = pd.DataFrame(probs, columns=[f"class_{i}_score" for i in range(probs.shape[1])])
    df["actual"] = torch.cat(labels).cpu().numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


def main():
    """
    Main script to:
      1) Set up seeds and device.
      2) Load data splits and DataLoader.
      3) Initialize models and load checkpoint.
      4) Print model summaries.
      5) Evaluate fusion, iris-only, and fp-only models.
    """
    # ---- Config ----
    iris_root = Path("../data/CASIA1-enhanced")
    fp_root = Path("../data/NIST301-augmented")
    batch_size = 32
    train_pct, val_pct = 0.7, 0.15
    num_classes = 100

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = get_device()
    print(f"Using device: {device}")

    # ---- Data ----
    samples = load(iris_root, fp_root, train_pct, val_pct)
    test_ds = IrisFingerprintDataset(samples["test"], transform=transforms.ToTensor())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ---- Models & Checkpoint ----
    models = init_models(device, num_classes=num_classes)
    ckpt_path = Path("results/fusion/best_model.pth")
    ckpt = load_checkpoint(ckpt_path, device)
    models["iris_fusion"].load_state_dict(ckpt["iris_model_state_dict"])
    models["fp_fusion"].load_state_dict(ckpt["fp_model_state_dict"])
    models["fusion"].load_state_dict(ckpt["fusion_model_state_dict"])
    ckpt_path = Path("results/iris/best_model.pth")
    ckpt = load_checkpoint(ckpt_path, device)
    models["iris"].load_state_dict(ckpt["model_state_dict"])
    models["iris_cls"].load_state_dict(ckpt["classifier_state_dict"])
    ckpt_path = Path("results/fp/best_model.pth")
    ckpt = load_checkpoint(ckpt_path, device)
    models["fp"].load_state_dict(ckpt["model_state_dict"])
    models["fp_cls"].load_state_dict(ckpt["classifier_state_dict"])
    
    for _, model in models.items():
        model.eval()
    
    # ---- Evaluation ----
    evaluate("fusion", test_loader, device, Path("results/eval/fusion_evaluation.csv"), models)
    evaluate("iris", test_loader, device, Path("results/eval/iris_evaluation.csv"), models)
    evaluate("fp", test_loader, device, Path("results/eval/fp_evaluation.csv"), models)

if __name__ == "__main__":
    main()