"""
Data collection script - captures images from webcam for training
Press SPACE to take photo, N for next gesture, Q to quit
"""
import cv2
import json
import os
import time

# Load labels
with open("labels.json", "r") as f:
    CLASSES = json.load(f)

TRAIN_DIR = "data/train"
TARGET_IMAGES = 50  # target images per class

print("=" * 60)
print("EMOJI GESTURE DATA COLLECTION")
print("=" * 60)
print("\nControls:")
print("  SPACE - Take ONE photo")
print("  N     - Next gesture class")
print("  Q     - Quit")
print(f"\nTarget: {TARGET_IMAGES} images per class")
print("\nTips for good data:")
print("  - Move your head around (different angles)")
print("  - Change distance from camera")
print("  - Try different lighting")
print("  - Move to different positions/backgrounds")
print("=" * 60)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

current_class_idx = 0
images_captured = 0

def count_images(class_name):
    folder = os.path.join(TRAIN_DIR, class_name)
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))])

print(f"\nStarting with: {CLASSES[current_class_idx]}")
print("Press SPACE to take a photo!\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    display = frame.copy()
    
    current_class = CLASSES[current_class_idx]
    images_captured = count_images(current_class)
    
    # Draw info box
    cv2.rectangle(display, (10, 10), (630, 150), (0, 0, 0), -1)
    cv2.rectangle(display, (10, 10), (630, 150), (0, 255, 0), 2)
    
    cv2.putText(display, f"Current: {current_class} ({current_class_idx + 1}/{len(CLASSES)})", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(display, f"Images: {images_captured}/{TARGET_IMAGES}", 
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display, "Press SPACE to capture", 
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    if images_captured >= TARGET_IMAGES:
        cv2.putText(display, "COMPLETE! Press N for next", 
                    (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.imshow("Data Collection", display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        # Take ONE photo
        folder = os.path.join(TRAIN_DIR, current_class)
        os.makedirs(folder, exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = os.path.join(folder, f"{current_class}_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        
        # Flash effect
        flash = display.copy()
        cv2.rectangle(flash, (0, 0), (display.shape[1], display.shape[0]), (255, 255, 255), 30)
        cv2.imshow("Data Collection", flash)
        cv2.waitKey(100)
        
        print(f"📸 Captured {current_class} ({count_images(current_class)}/{TARGET_IMAGES})")
        
    elif key == ord('n'):
        current_class_idx = (current_class_idx + 1) % len(CLASSES)
        print(f"\n➡️  Switched to: {CLASSES[current_class_idx]}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("COLLECTION SUMMARY:")
print("=" * 60)
for cls in CLASSES:
    count = count_images(cls)
    status = "✓" if count >= TARGET_IMAGES else "⚠"
    print(f"{status} {cls}: {count} images")
print("=" * 60)
