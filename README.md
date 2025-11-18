# 🎭 Emote Gesture Detection

Train a CNN to detect your gestures and map them to emojis in real-time!

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect training data (50+ images per gesture)
python collect_data.py

# 3. Split data into train/val
python split_data.py

# 4. Train your model
python train.py

# 5. Run real-time detection (desktop)
python realtime.py

# 6. Export ONNX for the browser demo (after training)
python export_onnx.py
```

## 📋 What You Need

**Before training, collect images of yourself doing each gesture:**
- Target: 50-100 images per gesture
- Move around for variety (angles, lighting, distance)
- The more diverse, the better the model

**Your gestures:**
- 67_emote
- goblin_crying
- wizard_dabbing
- biting_nails
- cheering
- neutral (idle/relaxed)

## 🎮 Controls

**collect_data.py:**
- `SPACE` - Start/pause capture
- `N` - Next gesture
- `Q` - Quit

**realtime.py:**
- `Q` - Quit

## 🧠 How It Works

```
Camera → Face Detection (CNN) → Gesture Recognition (ResNet18 CNN) → Emoji Display
```

1. OpenCV detects faces in webcam
2. ResNet18 CNN recognizes your gesture
3. Matching emoji is displayed

## 📁 Project Structure

```
imagedetection/
├── collect_data.py      # Capture training images
├── split_data.py        # Split train/val (80/20)
├── train.py             # Train ResNet18 CNN
├── export_onnx.py       # Convert PyTorch weights → ONNX + web config
├── realtime.py          # Live detection (OpenCV)
├── labels.json          # Gesture classes
├── emote_map.json       # Gesture→emoji mapping
├── index.html           # Web demo
├── data/
│   ├── train/          # Training images
│   └── val/            # Validation images
├── emotes/             # Your emoji images
└── models/             # Trained CNN models
```

## 🌐 Browser Demo & Deployment

`index.html` now runs your gesture model directly in the browser via [ONNX Runtime Web](https://onnxruntime.ai/docs/execution-providers/Web.html)—no Python backend required.

1. Train the model locally (`python train.py`) and make sure `models/expr_resnet18.pt` exists.
2. Export the weights + config for the web client:
   ```bash
   python export_onnx.py \
     --ckpt models/expr_resnet18.pt \
     --onnx models/expr_resnet18.onnx \
     --config models/web_model_config.json
   ```
3. Host the following files on any static host (e.g., GitHub Pages, Netlify, Vercel):
   - `index.html`
   - `emote_map.json`
   - `emotes/` (emoji images)
   - `models/expr_resnet18.onnx`
   - `models/web_model_config.json`
4. Open the hosted page, click **Start Camera**, and the UI will stream webcam frames, run ONNX inference client-side, show probabilities, and render the detected emote.

Tip: re-run `export_onnx.py` whenever you retrain so the web build stays in sync.
