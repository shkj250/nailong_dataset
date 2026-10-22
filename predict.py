"""加载训练得到的最佳模型，对单张图片进行分类。

示例：python predict.py --image data/test/naiwa/ba7a42d1d2389d86.jpg
"""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps

# 复用训练时的网络和评估预处理，避免预测与训练条件不一致。
# train.py 有主入口保护，所以这里 import 不会启动训练。
from train import ROOT, build_model, choose_device, make_transform


def main():
    parser = argparse.ArgumentParser(description="预测图片是奶龙还是奶蛙")
    parser.add_argument("--image", type=Path, required=True, help="待预测图片路径")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "best_model.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()
    device = choose_device(args.device)

    # 1. 读取自己训练的权重和类别顺序，不要加载来源不明的 checkpoint。
    # 先加载到 CPU，再把网络转到目标设备，减少加载期间的显存占用。
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = build_model(len(classes), pretrained=False)  # 已有权重，不必再次下载。
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()  # 预测必须使用评估模式，不能继续更新 BatchNorm 统计量。

    # 2. 把图片转换成 RGB，兼容灰度图、带透明通道的 PNG 等输入。
    # exif_transpose 根据照片方向信息纠正旋转；不会修改原文件。
    with Image.open(args.image) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = make_transform(training=False)(image)
    # 单张图片 [3,224,224] -> 一张图片的 batch [1,3,224,224]。
    tensor = tensor.unsqueeze(0).to(device)

    # 3. 推理不需要梯度，inference_mode 进一步减少开销。
    with torch.inference_mode():
        logits = model(tensor)
        # 训练用原始 logits；展示结果时才用 softmax 转成和为 1 的分数。
        probabilities = torch.softmax(logits, dim=1)[0].cpu()
    predicted_index = int(probabilities.argmax().item())
    names = {"nailong": "奶龙", "naiwa": "奶蛙"}
    predicted_class = classes[predicted_index]
    print(f"预测结果：{names.get(predicted_class, predicted_class)} ({predicted_class})")
    for label, probability in zip(classes, probabilities.tolist()):
        print(f"  {names.get(label, label)}：{probability:.2%}")
    # 二分类模型总会选这两类之一，即使输入是猫或风景。
    # softmax 分数不代表经过校准的可靠概率，也不能据此保证判断正确。
    print("注意：模型只在这两类中选择；分数不等于真实可靠程度。")


if __name__ == "__main__":
    main()
