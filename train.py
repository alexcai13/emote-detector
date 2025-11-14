# train.py
import json, os, time
from pathlib import Path
import torch, torch.nn as nn
import torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

# -------- config --------
DATA_DIR = "data"
TRAIN_DIR = f"{DATA_DIR}/train"
VAL_DIR   = f"{DATA_DIR}/val"
LABELS_F  = "labels.json"
OUT_DIR   = "models"
MODEL_OUT = f"{OUT_DIR}/expr_resnet18.pt"
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 20
IMG_SIZE = 224
NUM_WORKERS = 4
FREEZE_BACKBONE = False  # set True if very small dataset

# -------- setup --------
os.makedirs(OUT_DIR, exist_ok=True)
with open(LABELS_F, "r") as f:
    classes = json.load(f)
num_classes = len(classes)

device = "cuda" if torch.cuda.is_available() else "cpu"

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

train_ds = tv.datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
val_ds   = tv.datasets.ImageFolder(VAL_DIR,   transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# -------- model --------
model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
if FREEZE_BACKBONE:
    for p in model.parameters():
        p.requires_grad = False
    # unfreeze last block
    for p in model.layer4.parameters():
        p.requires_grad = True

in_feats = model.fc.in_features
model.fc = nn.Linear(in_feats, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_f1 = 0.0
best_state = None

# -------- training loop --------
def evaluate():
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
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    macro_f1 = report["macro avg"]["f1-score"]
    return val_loss, macro_f1, report

for epoch in range(1, EPOCHS+1):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    running = 0.0
    for x,y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
        pbar.set_postfix(loss=loss.item())
    scheduler.step()

    val_loss, macro_f1, report = evaluate()
    print(f"\nVal loss: {val_loss:.4f} | Macro-F1: {macro_f1:.4f}")
    if macro_f1 > best_val_f1:
        best_val_f1 = macro_f1
        best_state = {
            "model": model.state_dict(),
            "classes": classes,
            "img_size": IMG_SIZE,
            "mean": mean, "std": std
        }
        torch.save(best_state, MODEL_OUT)
        print(f"✅ Saved best model to {MODEL_OUT}")

print("\nFinal validation report:")
print(classification_report(
    [c for _,c in val_ds.samples],
    [], target_names=classes, zero_division=0
))
print(f"Best Macro-F1: {best_val_f1:.4f}")
