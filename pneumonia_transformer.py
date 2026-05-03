# Kaggle script: Train modified transformer on Chest X-Ray (Pneumonia)
# Dataset structure expected:
# /kaggle/input/datasets/chest_xray/
#   train/NORMAL, train/PNEUMONIA
#   val/NORMAL,   val/PNEUMONIA
#   test/NORMAL,  test/PNEUMONIA

import os
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# You need modified_transformer.py available in the notebook runtime.
# Upload it to /kaggle/working or add its path to sys.path.
import sys
sys.path.append("/kaggle/working")

from modified_transformer import (
    Encoder,
    RotaryPositionalEncoding,
    ALiBiPositionalBias,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
)

# ---------------------------
# 1) Config
# ---------------------------
SEED = 42
# Set this to your downloaded chest_xray dataset folder.
# Expected structure:
# DATA_ROOT/
#   train/NORMAL, train/PNEUMONIA
#   val/NORMAL,   val/PNEUMONIA
#   test/NORMAL,  test/PNEUMONIA
DATA_ROOT = None
BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCHS = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4

IMG_SIZE = 224
PATCH_SIZE = 16

D_MODEL = 128
NUM_LAYERS = 3
NUM_HEADS = 4
D_FF = 256
DROPOUT = 0.1
POS_ENCODING = "rope"      # "sinusoidal" | "learned" | "rope" | "alibi"
ATTENTION_TYPE = "vanilla" # "vanilla" | "linear" | "local"
FFN_TYPE = "gated"         # "standard" | "gated" | "depthwise_cnn"
USE_HYBRID_BLOCK = True
WINDOW_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_dataset_root() -> Path:
    """Locate a local dataset root containing train/val/test folders."""
    candidates = []

    if DATA_ROOT:
        candidates.append(Path(DATA_ROOT))

    env_root = os.getenv("CHEST_XRAY_DATASET_DIR")
    if env_root:
        candidates.append(Path(env_root))

    # Common local defaults
    candidates.extend([
        Path("./chest_xray"),
        Path("./data/chest_xray"),
    ])

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (
            (candidate / "train").exists()
            and (candidate / "val").exists()
            and (candidate / "test").exists()
        ):
            return candidate

    checked = "\n".join(str(p.expanduser()) for p in candidates)
    raise FileNotFoundError(
        "Could not find local chest_xray dataset root with train/val/test folders.\n"
        "Set DATA_ROOT in this script or set CHEST_XRAY_DATASET_DIR env var.\n"
        f"Checked:\n{checked}"
    )


# ---------------------------
# 2) Model
# ---------------------------
class XRayTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes=2,
        in_channels=1,
        img_size=224,
        patch_size=16,
        d_model=128,
        num_layers=3,
        num_heads=4,
        d_ff=256,
        dropout=0.1,
        pos_encoding="rope",
        attention_type="vanilla",
        ffn_type="gated",
        use_hybrid_block=True,
        window_size=64,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid = img_size // patch_size
        self.seq_len = self.grid * self.grid

        # Patch embedding: (B, C, H, W) -> (B, d_model, H/P, W/P)
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

        rope = None
        alibi = None
        head_dim = d_model // num_heads
        max_len = self.seq_len + 1  # +1 for cls token

        if pos_encoding == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        elif pos_encoding == "learned":
            self.pos_enc = LearnedPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        elif pos_encoding == "rope":
            rope = RotaryPositionalEncoding(head_dim=head_dim, max_len=max_len)
            self.pos_enc = nn.Dropout(dropout)
        elif pos_encoding == "alibi":
            alibi = ALiBiPositionalBias(num_heads=num_heads, max_len=max_len)
            self.pos_enc = nn.Dropout(dropout)
        else:
            raise ValueError(f"Unsupported pos_encoding: {pos_encoding}")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

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

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (B, 1, H, W)
        x = self.patch_embed(x)                    # (B, d_model, G, G)
        x = x.flatten(2).transpose(1, 2)           # (B, seq_len, d_model)

        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)             # (B, 1+seq_len, d_model)
        x = self.pos_enc(x)

        enc_outputs = self.encoder(x)
        cls_out = enc_outputs[-1][:, 0]            # CLS token output
        return self.head(cls_out)


# ---------------------------
# 3) Data
# ---------------------------
def get_dataloaders(data_root: Path, batch_size: int, num_workers: int):
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25]),
    ])

    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25]),
    ])

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_root / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_root / "test", transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes


# ---------------------------
# 4) Train / Eval
# ---------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    all_labels, all_preds, all_probs = [], [], []
    total_loss = 0.0
    total_count = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        probs = torch.softmax(logits, dim=1)[:, 1]  # prob of class 1
        preds = torch.argmax(logits, dim=1)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_count += bs

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(total_count, 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float("nan")

    return avg_loss, acc, f1, auc, all_labels, all_preds


def main():
    set_seed(SEED)
    print("Device:", DEVICE)

    data_root = resolve_dataset_root()
    print(f"Using dataset root: {data_root}")

    train_loader, val_loader, test_loader, classes = get_dataloaders(data_root, BATCH_SIZE, NUM_WORKERS)
    print("Classes:", classes)

    model = XRayTransformerClassifier(
        num_classes=len(classes),
        in_channels=1,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        pos_encoding=POS_ENCODING,
        attention_type=ATTENTION_TYPE,
        ffn_type=FFN_TYPE,
        use_hybrid_block=USE_HYBRID_BLOCK,
        window_size=WINDOW_SIZE,
    ).to(DEVICE)

    # Handle class imbalance
    class_counts = np.bincount([y for _, y in train_loader.dataset.samples])
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_auc = -1.0
    save_path = Path("./best_xray_transformer.pt")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_f1, tr_auc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, va_f1, va_auc, _, _ = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} auc {tr_auc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f} auc {va_auc:.4f}"
        )

        if np.isfinite(va_auc) and va_auc > best_val_auc:
            best_val_auc = va_auc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "config": {
                        "img_size": IMG_SIZE,
                        "patch_size": PATCH_SIZE,
                        "d_model": D_MODEL,
                        "num_layers": NUM_LAYERS,
                        "num_heads": NUM_HEADS,
                        "d_ff": D_FF,
                        "dropout": DROPOUT,
                        "pos_encoding": POS_ENCODING,
                        "attention_type": ATTENTION_TYPE,
                        "ffn_type": FFN_TYPE,
                        "use_hybrid_block": USE_HYBRID_BLOCK,
                        "window_size": WINDOW_SIZE,
                    },
                },
                save_path,
            )

    print(f"\nBest model saved to: {save_path}")

    # Test with best checkpoint
    ckpt = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    te_loss, te_acc, te_f1, te_auc, y_true, y_pred = run_epoch(model, test_loader, criterion, optimizer=None)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Test Metrics ===")
    print(f"Loss: {te_loss:.4f}")
    print(f"Accuracy: {te_acc:.4f}")
    print(f"F1 (binary): {te_f1:.4f}")
    print(f"ROC-AUC: {te_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()