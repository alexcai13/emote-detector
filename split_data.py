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
VAL_SPLIT = 0.2  

with open("labels.json", "r") as f:
    CLASSES = json.load(f)

print("Splitting data")
print("=" * 60)

random.seed(42)  

total_moved = 0

for class_name in CLASSES:
    train_folder = os.path.join(TRAIN_DIR, class_name)
    val_folder = os.path.join(VAL_DIR, class_name)
    
    images = [f for f in os.listdir(train_folder) if f.endswith(('.jpg'))]
    random.shuffle(images)
    val_count = max(1, int(len(images) * VAL_SPLIT)) 
    val_images = images[:val_count]
    os.makedirs(val_folder, exist_ok=True)
    for img in val_images:
        src = os.path.join(train_folder, img)
        dst = os.path.join(val_folder, img)
        shutil.move(src, dst)
    
    total_moved += len(val_images)
    train_remaining = len(images) - len(val_images)


print("Finished splitting")

