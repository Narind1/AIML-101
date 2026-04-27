"""
Train & Evaluate the Modified Transformer on classification tasks.
==================================================================
Datasets:  MNIST  (10 classes, 28×28 grayscale images)
           Wine   (3 classes, 13 features — sklearn)

Each image/sample is treated as a sequence fed into the transformer encoder;
a [CLS]-style pooled representation is used for classification.
"""

import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from tabulate import tabulate
import torchvision
import torchvision.transforms as transforms

from modified_transformer import (
    Encoder,
    RotaryPositionalEncoding,
    ALiBiPositionalBias,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    HybridCNNRNNBlock,
)

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}\n")


# =========================================================================
# 1. Transformer-based Classifier
# =========================================================================

class TransformerClassifier(nn.Module):
    """
    Uses the modified transformer encoder (with all modifications) as
    a feature extractor, then mean-pools over the sequence and projects
    to `num_classes`.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
        pos_encoding: str = "rope",
        attention_type: str = "vanilla",
        ffn_type: str = "gated",
        use_hybrid_block: bool = True,
        window_size: int = 64,
    ):
        super().__init__()
        self.d_model = d_model

        # Project raw features to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        rope, alibi = None, None
        head_dim = d_model // num_heads

        if pos_encoding == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        elif pos_encoding == "learned":
            self.pos_enc = LearnedPositionalEncoding(d_model, max_len, dropout)
        elif pos_encoding == "rope":
            rope = RotaryPositionalEncoding(head_dim, max_len)
            self.pos_enc = nn.Dropout(dropout)
        elif pos_encoding == "alibi":
            alibi = ALiBiPositionalBias(num_heads, max_len)
            self.pos_enc = nn.Dropout(dropout)

        # Encoder
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            attention_type=attention_type,
            ffn_type=ffn_type,
            use_hybrid_block=use_hybrid_block,
            window_size=window_size,
            rope=rope,
            alibi=alibi,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        Returns: (batch, num_classes) logits
        """
        x = self.input_proj(x)  # (B, L, d_model)
        x = self.pos_enc(x)
        enc_outputs = self.encoder(x)  # list of layer outputs
        out = enc_outputs[-1]           # last layer: (B, L, d_model)
        pooled = out.mean(dim=1)        # mean-pool over sequence
        return self.classifier(pooled)


# =========================================================================
# 2. Data Loading
# =========================================================================

def load_mnist(batch_size: int = 128, train_subset: int = 10000, test_subset: int = 2000):
    """
    MNIST: 28×28 images treated as sequences of 28 time-steps, each of dim 28.
    Uses subsets for CPU-friendly training time.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Use subsets for faster CPU training
    if train_subset and train_subset < len(train_ds):
        indices = torch.randperm(len(train_ds))[:train_subset]
        train_ds = torch.utils.data.Subset(train_ds, indices)
    if test_subset and test_subset < len(test_ds):
        indices = torch.randperm(len(test_ds))[:test_subset]
        test_ds = torch.utils.data.Subset(test_ds, indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def load_wine_dataset(batch_size: int = 32):
    """
    Wine: 13 features → treated as a sequence of 13 time-steps, each dim 1.
    """
    data = load_wine()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Reshape: (N, 13) → (N, 13, 1)  — sequence of 13 steps, 1 feature each
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


# =========================================================================
# 3. Training & Evaluation Loop
# =========================================================================

def train_one_epoch(model, loader, optimizer, criterion, dataset_name="mnist"):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        if dataset_name == "mnist":
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # (B, 1, 28, 28) → (B, 28, 28) — 28 rows of 28 pixels
            x = images.squeeze(1)
        else:
            x, labels = batch
            x, labels = x.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, dataset_name="mnist"):
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        if dataset_name == "mnist":
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            x = images.squeeze(1)
        else:
            x, labels = batch
            x, labels = x.to(DEVICE), labels.to(DEVICE)

        logits = model(x)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    metrics = {
        "Accuracy":  accuracy_score(all_labels, all_preds),
        "Precision (macro)": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "Recall (macro)":    recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "F1 (macro)":        f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "Precision (weighted)": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "Recall (weighted)":    recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "F1 (weighted)":        f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }
    cm = confusion_matrix(all_labels, all_preds)
    return metrics, cm, all_preds, all_labels


# =========================================================================
# 4. Per-class Metrics Table
# =========================================================================

def per_class_table(all_labels, all_preds, class_names):
    """Build a per-class precision / recall / f1 table."""
    from sklearn.metrics import classification_report
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4, zero_division=0
    )
    return report


# =========================================================================
# 5. Main — run both datasets
# =========================================================================

def run_experiment(
    dataset_name: str,
    train_loader,
    test_loader,
    input_dim: int,
    num_classes: int,
    class_names: list,
    epochs: int = 10,
    lr: float = 1e-3,
):
    print(f"\n{'='*70}")
    print(f"  Dataset: {dataset_name.upper()}")
    print(f"{'='*70}")

    model_cfg = dict(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=128,
        num_layers=3,
        num_heads=4,
        d_ff=256,
        dropout=0.1,
        max_len=512,
        pos_encoding="rope",
        attention_type="vanilla",
        ffn_type="gated",
        use_hybrid_block=True,
        window_size=64,
    )

    model = TransformerClassifier(**model_cfg).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")
    print(f"  Config: pos=rope | attn=vanilla | ffn=gated | hybrid=True")
    print(f"  Epochs: {epochs} | LR: {lr}")
    print(f"{'─'*70}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(1, epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, dataset_name)
        scheduler.step()
        if epoch % max(1, epochs // 5) == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={loss:.4f}  train_acc={acc:.4f}")
    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.1f}s")

    # Evaluate
    metrics, cm, preds, labels = evaluate(model, test_loader, dataset_name)
    print(f"\n  {'─'*50}")
    print(f"  Test Results — {dataset_name.upper()}")
    print(f"  {'─'*50}")
    for k, v in metrics.items():
        print(f"    {k:25s}: {v:.4f}")

    print(f"\n  Confusion Matrix:\n{cm}\n")

    # Per-class report
    print("  Per-class Classification Report:")
    print(per_class_table(labels, preds, class_names))

    return metrics, elapsed


def main():
    all_results = {}

    # ---------- MNIST ----------
    print("Loading MNIST...")
    mnist_train, mnist_test = load_mnist(batch_size=128)
    class_names_mnist = [str(i) for i in range(10)]
    metrics_mnist, time_mnist = run_experiment(
        dataset_name="mnist",
        train_loader=mnist_train,
        test_loader=mnist_test,
        input_dim=28,        # each row of the image is a 28-dim feature
        num_classes=10,
        class_names=class_names_mnist,
        epochs=5,
        lr=1e-3,
    )
    all_results["MNIST"] = {**metrics_mnist, "Time (s)": time_mnist}

    # ---------- Wine ----------
    print("\nLoading Wine...")
    wine_train, wine_test = load_wine_dataset(batch_size=32)
    class_names_wine = ["Class 0", "Class 1", "Class 2"]
    metrics_wine, time_wine = run_experiment(
        dataset_name="wine",
        train_loader=wine_train,
        test_loader=wine_test,
        input_dim=1,         # single feature per time-step
        num_classes=3,
        class_names=class_names_wine,
        epochs=30,
        lr=1e-3,
    )
    all_results["Wine"] = {**metrics_wine, "Time (s)": time_wine}

    # =====================================================================
    # FINAL COMPARISON TABLE
    # =====================================================================
    print("\n" + "=" * 80)
    print("  FINAL COMPARISON TABLE — Modified Transformer Classification")
    print("=" * 80)

    headers = [
        "Dataset",
        "Accuracy",
        "Precision\n(macro)",
        "Recall\n(macro)",
        "F1\n(macro)",
        "Precision\n(weighted)",
        "Recall\n(weighted)",
        "F1\n(weighted)",
        "Time (s)",
    ]

    rows = []
    for ds_name, m in all_results.items():
        rows.append([
            ds_name,
            f"{m['Accuracy']:.4f}",
            f"{m['Precision (macro)']:.4f}",
            f"{m['Recall (macro)']:.4f}",
            f"{m['F1 (macro)']:.4f}",
            f"{m['Precision (weighted)']:.4f}",
            f"{m['Recall (weighted)']:.4f}",
            f"{m['F1 (weighted)']:.4f}",
            f"{m['Time (s)']:.1f}",
        ])

    print(tabulate(rows, headers=headers, tablefmt="grid", stralign="center"))
    print()


if __name__ == "__main__":
    main()
