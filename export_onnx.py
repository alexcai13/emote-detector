"""
Export the trained PyTorch model to ONNX for use in the web demo.

Usage:
    python3 export_onnx.py --ckpt models/expr_resnet18.pt \
                           --onnx models/expr_resnet18.onnx \
                           --config models/web_model_config.json
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision as tv


def parse_args():
    parser = argparse.ArgumentParser(description="Export trained model to ONNX + web config")
    parser.add_argument("--ckpt", type=Path, default=Path("models/expr_resnet18.pt"),
                        help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--onnx", type=Path, default=Path("models/expr_resnet18.onnx"),
                        help="Output ONNX file path")
    parser.add_argument("--config", type=Path, default=Path("models/web_model_config.json"),
                        help="Output JSON config that stores classes + normalization")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt["classes"]
    mean = ckpt["mean"]
    std = ckpt["std"]
    img_size = ckpt["img_size"]

    model = tv.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size)
    args.onnx.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        args.onnx,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"✅ Exported ONNX model to {args.onnx}")

    config = {
        "classes": classes,
        "img_size": img_size,
        "mean": mean,
        "std": std,
    }
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(json.dumps(config, indent=2))
    print(f"✅ Wrote web config to {args.config}")


if __name__ == "__main__":
    main()
