"""
Split training data into train/val sets (80/20 split)
"""
import os
import shutil
import random
import json
from pathlib import Path

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
VAL_SPLIT = 0.2  # 20% for validation

with open("labels.json", "r") as f:
    CLASSES = json.load(f)

print("=" * 60)
print("SPLITTING DATA INTO TRAIN/VAL")
print("=" * 60)

random.seed(42)  # for reproducibility

total_moved = 0

for class_name in CLASSES:
    train_folder = os.path.join(TRAIN_DIR, class_name)
    val_folder = os.path.join(VAL_DIR, class_name)
    
    if not os.path.exists(train_folder):
        print(f"⚠ Skipping {class_name}: no training data found")
        continue
    
    # Get all images
    images = [f for f in os.listdir(train_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(images) == 0:
        print(f"⚠ Skipping {class_name}: no images found")
        continue
    
    # Shuffle and split
    random.shuffle(images)
    val_count = max(1, int(len(images) * VAL_SPLIT))  # at least 1 for val
    val_images = images[:val_count]
    
    # Move to val folder
    os.makedirs(val_folder, exist_ok=True)
    for img in val_images:
        src = os.path.join(train_folder, img)
        dst = os.path.join(val_folder, img)
        shutil.move(src, dst)
    
    total_moved += len(val_images)
    train_remaining = len(images) - len(val_images)
    
    print(f"✓ {class_name}: {train_remaining} train, {len(val_images)} val")

print("=" * 60)
print(f"Total: {total_moved} images moved to validation")
print("Split complete! Ready for training.")
print("=" * 60)
