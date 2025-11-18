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

# 6. (Maintainers only) Export ONNX when you retrain
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

### Hosted demo (recommended for viewers)
- Just open the deployed GitHub Pages/Netlify URL and click **Start Camera**. Everything—model download, webcam capture, inference—runs directly in your browser. No setup required.

### Updating the hosted model (maintainers only)
1. Train the model locally (`python train.py`) so `models/expr_resnet18.pt` is current.
2. Run:
   ```bash
   python export_onnx.py \
     --ckpt models/expr_resnet18.pt \
     --onnx models/expr_resnet18.onnx \
     --config models/web_model_config.json
   ```
3. Deploy/commit the updated `models/expr_resnet18.onnx` and `models/web_model_config.json` along with `index.html`, `emote_map.json`, and `emotes/`.
4. Reload the hosted page—viewers automatically get the new model.
