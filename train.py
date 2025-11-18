# train.py
import json, os, time, shutil
from pathlib import Path
import torch, torch.nn as nn
import torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from urllib.error import URLError

# -------- config --------
DATA_DIR = "data"
TRAIN_DIR = f"{DATA_DIR}/train"
VAL_DIR   = f"{DATA_DIR}/val"
LABELS_F  = "labels.json"
OUT_DIR   = "models"
MODEL_OUT = f"{OUT_DIR}/expr_resnet18.pt"
LOCAL_PRETRAIN_F = f"{OUT_DIR}/resnet18-f37072fd.pth"
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 20
IMG_SIZE = 224
NUM_WORKERS = 4
FREEZE_BACKBONE = False  # set True if very small dataset

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def prune_empty_class_dirs(root):
    removed = []
    root_path = Path(root)
    if not root_path.exists():
        return removed
    for child in root_path.iterdir():
        if not child.is_dir():
            continue
        has_images = any(f.suffix.lower() in IMG_EXTENSIONS for f in child.iterdir())
        if not has_images:
            shutil.rmtree(child)
            removed.append(child.name)

def build_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_tfms, val_tfms, mean, std

def build_datasets(train_tfms, val_tfms):
    pruned_train = prune_empty_class_dirs(TRAIN_DIR)
    pruned_val = prune_empty_class_dirs(VAL_DIR)
    if pruned_train or pruned_val:
        msg_parts = []
        if pruned_train:
            msg_parts.append(f"train({', '.join(pruned_train)})")
        if pruned_val:
            msg_parts.append(f"val({', '.join(pruned_val)})")
        print("ℹ️  Removed empty class folders:", " & ".join(msg_parts))

    train_ds = tv.datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
    val_ds   = tv.datasets.ImageFolder(VAL_DIR,   transform=val_tfms)

    if len(train_ds.classes) == 0:
        raise RuntimeError("No training data found in data/train. Collect data and/or run split_data.py first.")
    if len(val_ds) == 0:
        raise RuntimeError("Validation set is empty. Run split_data.py so we can evaluate during training.")

    return train_ds, val_ds

def build_model(num_classes, device):
    weights_enum = tv.models.ResNet18_Weights.IMAGENET1K_V1
    model = None
    if os.path.exists(LOCAL_PRETRAIN_F):
        print(f"🔁 Loading pretrained weights from {LOCAL_PRETRAIN_F}")
        state = torch.load(LOCAL_PRETRAIN_F, map_location="cpu")
        model = tv.models.resnet18(weights=None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"ℹ️  Loaded local weights with missing={len(missing)} unexpected={len(unexpected)} layers.")
    if model is None:
        try:
            model = tv.models.resnet18(weights=weights_enum)
        except URLError as err:
            print(f"⚠️  Could not download pretrained weights ({err}). Falling back to random initialization.")
            model = tv.models.resnet18(weights=None)
    if FREEZE_BACKBONE:
        for p in model.parameters():
            p.requires_grad = False
        # unfreeze last block
        for p in model.layer4.parameters():
            p.requires_grad = True

    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model.to(device)

def evaluate(model, val_loader, device, criterion, classes):
    model.eval()
    y_true, y_pred = [], []
    val_loss, n = 0.0, 0
    with torch.no_grad():
        for x,y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            val_loss += loss.item() * x.size(0)
            n += x.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true += y.cpu().tolist()
            y_pred += preds.cpu().tolist()
    val_loss /= max(n,1)
    if len(y_true) == 0:
        return val_loss, 0.0, [], []
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return val_loss, macro_f1, y_true, y_pred

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(LABELS_F, "r") as f:
        declared_classes = json.load(f)

    train_tfms, val_tfms, mean, std = build_transforms()
    train_ds, val_ds = build_datasets(train_tfms, val_tfms)

    classes = train_ds.classes
    missing = [c for c in declared_classes if c not in classes]
    if missing:
        print("⚠️  Skipping classes without training data:", ", ".join(missing))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    effective_workers = NUM_WORKERS
    if device == "cpu" and NUM_WORKERS > 0:
        # Mac/Windows default start method = spawn, which needs guarded main.
        # Drop to single-process loading if user hits spawn issues.
        effective_workers = 0

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=effective_workers)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=effective_workers)

    model = build_model(num_classes=len(classes), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_f1 = 0.0
    best_state = None
    best_report_txt = ""

    for epoch in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for x,y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
        scheduler.step()

        val_loss, macro_f1, y_true, y_pred = evaluate(model, val_loader, device, criterion, classes)
        if y_true:
            report_txt = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
            print(report_txt)
        else:
            report_txt = "No validation predictions to report."
        print(f"\nVal loss: {val_loss:.4f} | Macro-F1: {macro_f1:.4f}")
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_state = {
                "model": model.state_dict(),
                "classes": classes,
                "img_size": IMG_SIZE,
                "mean": mean,
                "std": std
            }
            torch.save(best_state, MODEL_OUT)
            print(f"✅ Saved best model to {MODEL_OUT}")
            best_report_txt = report_txt

    print("\nBest validation report:")
    print(best_report_txt or "No validation data available.")
    print(f"Best Macro-F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    main()
