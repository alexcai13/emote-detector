# realtime_emotes.py
import os, json, time
from pathlib import Path
import numpy as np
import cv2
import torch
import torchvision as tv
import torch.nn.functional as F

# ---------- config ----------
MODEL_EXPR_PATH = "models/expr_resnet18.pt"
EMOTE_MAP_PATH  = "emote_map.json"
LABELS_PATH     = "labels.json"
# OpenCV DNN face detector (CNN)
FACE_PROTO = "models/face/deploy.prototxt"
FACE_MODEL = "models/face/res10_300x300_ssd_iter_140000.caffemodel"

CONF_THRESH = 0.6
SMOOTH_ALPHA = 0.3  # temporal smoothing
DISPLAY_SIZE = (720, 450)

# ---------- load expression model ----------
ckpt = torch.load(MODEL_EXPR_PATH, map_location="cpu")
classes = ckpt["classes"]
img_size = ckpt["img_size"]
mean, std = ckpt["mean"], ckpt["std"]

# build same model arch
model = tv.models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["model"])
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# preproc
import torchvision.transforms as T
preproc = T.Compose([
    T.ToPILImage(),
    T.Resize((img_size, img_size)),
    T.ToTensor(),
    T.Normalize(mean, std)
])

# emote map
with open(EMOTE_MAP_PATH, "r") as f:
    EMOTE_MAP = json.load(f)

# ---------- load face CNN ----------
net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

# load emote images
def load_png(path, size):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None
    return cv2.resize(img, size)

# keep a cache
emote_cache = {}

# blending function (supports RGBA)
def overlay_rgba(bg_bgr, fg_rgba):
    if fg_rgba is None:
        return bg_bgr
    if fg_rgba.shape[2] == 3:
        # no alpha—just place
        return fg_rgba
    fg_rgb = fg_rgba[:,:,:3]
    alpha  = fg_rgba[:,:,3:4].astype(np.float32)/255.0
    bg_rgb = bg_bgr.astype(np.float32)
    out = alpha*fg_rgb.astype(np.float32) + (1-alpha)*bg_rgb
    return out.astype(np.uint8)

# temporal smoothing
probs_smooth = None

# webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam"); exit()

cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
cv2.namedWindow('Emote', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera', *DISPLAY_SIZE)
cv2.resizeWindow('Emote',  *DISPLAY_SIZE)

def detect_faces_dnn(frame_bgr):
    (h, w) = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300, 300)),
                                 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0,0,i,2]
        if conf >= CONF_THRESH:
            box = detections[0,0,i,3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0,x1), max(0,y1)
            x2, y2 = min(w-1,x2), min(h-1,y2)
            boxes.append((x1,y1,x2,y2, conf))
    # return biggest face
    if not boxes: return None
    boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return boxes[0]

while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    vis  = frame.copy()

    det = detect_faces_dnn(frame)
    top_label = "neutral"
    top_prob  = 0.0

    if det is not None:
        x1,y1,x2,y2,_ = det
        # add a little margin
        mx = int(0.15*(x2-x1)); my = int(0.20*(y2-y1))
        x1 = max(0, x1-mx); y1 = max(0, y1-my)
        x2 = min(frame.shape[1]-1, x2+mx); y2 = min(frame.shape[0]-1, y2+my)

        face = frame[y1:y2, x1:x2, :]
        if face.size > 0:
            inp = preproc(face).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(inp)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            # smooth
            global probs_smooth
            if probs_smooth is None:
                probs_smooth = probs
            else:
                probs_smooth = SMOOTH_ALPHA*probs + (1-SMOOTH_ALPHA)*probs_smooth

            idx = int(np.argmax(probs_smooth))
            top_label = classes[idx]
            top_prob  = float(probs_smooth[idx])

            # draw box
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            txt = f"{top_label}: {top_prob:.2f}"
            cv2.putText(vis, txt, (x1, max(20,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # load emote for top_label
    emote_path = EMOTE_MAP.get(top_label, None)
    emote_view = np.zeros((DISPLAY_SIZE[1], DISPLAY_SIZE[0], 3), dtype=np.uint8)
    if emote_path:
        if emote_path not in emote_cache:
            emote_cache[emote_path] = load_png(emote_path, DISPLAY_SIZE)
        emote_img = emote_cache[emote_path]
        if emote_img is not None:
            # if RGBA, composite over black bg
            emote_view = overlay_rgba(emote_view, emote_img)

    cam_view = cv2.resize(vis, DISPLAY_SIZE)

    cv2.imshow("Camera", cam_view)
    cv2.imshow("Emote", emote_view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
