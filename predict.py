"""用可信的三分类 ResNet18 权重预测单张图片。"""

import argparse
import pickle
from pathlib import Path

import torch
from PIL import Image, ImageOps

from train import CLASS_NAMES, DEFAULT_CHECKPOINT, choose_device, load_trained_model, make_transform


def main():
    parser = argparse.ArgumentParser(description="预测图片是奶龙、奶蛙还是其他")
    parser.add_argument("--image", type=Path, required=True, help="待预测图片路径")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="三分类权重，不能使用旧二分类权重")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()
    device = choose_device(args.device)
    try:
        model, classes = load_trained_model(args.checkpoint, device)
    except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        parser.error(str(exc))
    try:
        with Image.open(args.image) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = make_transform(training=False)(image).unsqueeze(0).to(device)
    except (OSError, ValueError) as exc:
        parser.error(f"无法读取图片：{exc}")
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
    predicted = classes[int(probabilities.argmax().item())]
    print(f"预测结果：{CLASS_NAMES[predicted]} ({predicted})")
    for label, score in zip(classes, probabilities.tolist()):
        print(f"  {CLASS_NAMES[label]}：{score:.2%}")
    print("注意：分数不是准确率保证；其他类仅代表模型学过的负样本，不能保证识别所有未知图片。")


if __name__ == "__main__":
    main()
