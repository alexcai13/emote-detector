import cv2
import json
import os
import time

with open("labels.json", "r") as f:
    CLASSES = json.load(f)

TRAIN_DIR = "data/train"
TARGET_IMAGES = 50 

print("\nControls:")
print("  SPACE - Take ONE photo")
print("  N     - Next gesture class")
print("  Q     - Quit")
print(f"\nTarget: {TARGET_IMAGES} images per class")
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
    
    cv2.rectangle(display, (10, 10), (630, 150), (0, 0, 0), -1)
    cv2.rectangle(display, (10, 10), (630, 150), (0, 255, 0), 2)
    
    cv2.putText(display, f"Current: {current_class} ({current_class_idx + 1}/{len(CLASSES)})", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(display, f"Images: {images_captured}/{TARGET_IMAGES}", 
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display, "Press space to capture", 
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    if images_captured >= TARGET_IMAGES:
        cv2.putText(display, "Complete, Press N for next", 
                    (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.imshow("Data Collection", display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        folder = os.path.join(TRAIN_DIR, current_class)
        os.makedirs(folder, exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = os.path.join(folder, f"{current_class}_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        
        flash = display.copy()
        cv2.rectangle(flash, (0, 0), (display.shape[1], display.shape[0]), (255, 255, 255), 30)
        cv2.imshow("Data Collection", flash)
        cv2.waitKey(100)
        
        print(f"Captured {current_class} ({count_images(current_class)}/{TARGET_IMAGES})")
        
    elif key == ord('n'):
        current_class_idx = (current_class_idx + 1) % len(CLASSES)
        print(f"\nSwitched to: {CLASSES[current_class_idx]}")

cap.release()
cv2.destroyAllWindows()

print("\n")
print("COLLECTION SUMMARY:")
for cls in CLASSES:
    count = count_images(cls)
    status = "Yes" if count >= TARGET_IMAGES else "No"
    print(f"{status} {cls}: {count} images")
print("=" * 60)
