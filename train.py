"""用 ResNet18 迁移学习，训练奶龙 / 奶蛙图片分类器。

运行：python train.py
离线且没有预训练权重时：python train.py --no-pretrained
本文件也提供预测脚本需要的模型结构和预处理函数。
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# 默认路径以脚本所在目录为基准，从其他目录运行也能找到数据。
ROOT = Path(__file__).resolve().parent
# 预训练 ResNet 使用 ImageNet 的均值和标准差，训练和预测都要保持一致。
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    """固定随机数，减少数据打乱、初始化等过程带来的结果差异。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 提高 CUDA 运算的可复现性；不同设备/版本间仍不保证完全一致。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(name):
    """自动优先用 NVIDIA GPU，其次 Apple GPU，最后 CPU。"""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_transform(training=False):
    """把不同尺寸的图片变成可组成一个 batch 的标准化张量。"""
    if training:
        # 随机裁剪和翻转只用于训练：让模型不要死记固定构图。
        # 裁剪面积至少保留原图的 80%，降低把角色完全裁掉的风险。
        steps = [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    else:
        # 验证、测试、预测不做随机增强，保证评估条件一致。
        # 固定缩放保留整张图片，但会轻微改变非正方形图片的比例。
        steps = [transforms.Resize((224, 224))]
        
    return transforms.Compose(steps + [
        # PIL 图片 [0,255] -> float32 张量 [3,224,224]，数值范围 [0,1]。
        transforms.ToTensor(),
        # 对每个通道执行 (像素 - 均值) / 标准差。
        transforms.Normalize(MEAN, STD),
    ])


def build_model(num_classes, pretrained=True):
    """加载特征提取网络，把原来的 1000 类输出层换成当前类别数。"""
    # 首次使用预训练模式可能需要联网下载权重；之后使用本地缓存。
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    # ResNet18 将图片提取为 512 维特征，最后通过 Linear 映射成两个分数。
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, device, optimizer=None):
    """执行一轮训练或评估；传 optimizer 才会反向传播并更新参数。"""
    training = optimizer is not None
    # train() 启用训练行为；eval() 让 BatchNorm 使用已记录的统计量。
    model.train(training)
    total_loss, total_correct, total_samples = 0.0, 0, 0
    # 混淆矩阵：行是真实类别，列是预测类别，索引顺序来自 ImageFolder。
    num_classes = model.fc.out_features
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    # 评估不记录梯度，节省内存和计算；训练则正常记录计算图。
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            # images: [batch_size, 3, 224, 224]；labels: [batch_size]。
            images = images.to(device)
            labels = labels.to(device)
            if training:
                # PyTorch 默认累加梯度，所以每个 batch 更新前必须清空。
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)  # [batch_size, 2]，未经过 softmax 的分数。
            # CrossEntropyLoss 内部包含 log_softmax；这里不要提前 softmax。
            # labels 必须是整数类别索引，而不是 one-hot 或浮点数。
            loss = criterion(logits, labels)
            if training:
                loss.backward()  # 链式求导：计算每个可训练参数的梯度。
                optimizer.step()  # 根据梯度与学习率，实际更新模型权重。

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            # 乘 batch_size 后累加，避免最后一个小 batch 导致平均值偏差。
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size
            # 合并真实/预测索引，用 bincount 一次统计整个 batch。
            indices = labels.detach().cpu() * num_classes + predictions.detach().cpu()
            confusion += torch.bincount(
                indices, minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)

    return total_loss / total_samples, total_correct / total_samples, confusion


def parse_args():
    parser = argparse.ArgumentParser(description="奶龙 / 奶蛙二分类训练")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--epochs", type=int, default=15, help="完整遍历训练集的次数")
    parser.add_argument("--batch-size", type=int, default=16, help="一次处理的图片数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率：每步更新幅度")
    parser.add_argument("--seed", type=int, default=42)
    # 默认 0 避免初学者遇到 Windows/交互环境多进程问题。
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--no-pretrained", action="store_true", help="随机初始化，不下载权重")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.num_workers < 0:
        parser.error("epochs、batch-size、lr 必须大于 0，num-workers 不能小于 0")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 第一步：按目录读取图片。ImageFolder 自动把类别目录名映射为整数。
    # 三个划分必须拥有相同类别；不重新拆分现有数据，也不混用测试集。
    splits = {}
    for split in ("train", "val", "test"):
        split_dir = args.data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"缺少数据目录：{split_dir}")
        splits[split] = datasets.ImageFolder(
            split_dir, transform=make_transform(training=(split == "train"))
        )
    classes = splits["train"].classes
    if set(classes) != {"nailong", "naiwa"}:
        raise ValueError(f"预期类别目录 nailong、naiwa，实际为 {classes}")
    for split, dataset in splits.items():
        if dataset.class_to_idx != splits["train"].class_to_idx:
            raise ValueError(f"{split} 的类别映射与训练集不一致")

    # 第二步：DataLoader 按 batch 取数据。只打乱训练集，不丢弃最后一个 batch。
    loaders = {
        split: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(split == "train"),
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        )
        for split, dataset in splits.items()
    }
    print(f"设备：{device}；类别映射：{splits['train'].class_to_idx}")
    print("样本数：", {split: len(ds) for split, ds in splits.items()})

    # 第三步：迁移学习。默认所有层都参与微调，小学习率保护预训练特征。
    model = build_model(len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    # AdamW 自动调整参数更新尺度；weight_decay 是抑制过拟合的正则化项。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_acc, best_loss = -1.0, float("inf")
    best_path = args.output_dir / "best_model.pt"
    history = []

    # 第四步：训练 -> 验证 -> 保存最佳权重。验证集不会参与梯度更新。
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _ = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc, _ = run_epoch(model, loaders["val"], criterion, device)
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss={train_loss:.4f} acc={train_acc:.2%} | "
              f"val loss={val_loss:.4f} acc={val_acc:.2%}")
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc))
        # 优先选验证准确率更高的模型；准确率相同时选 loss 更低的。
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss = val_acc, val_loss
            # 保存 state_dict 而不是整个 Python 模型，并保存类别顺序供预测使用。
            # 此文件用于推理，不包含恢复训练所需的优化器状态。
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "epoch": epoch,
                "val_acc": val_acc,
                "architecture": "resnet18",
            }, best_path)
            print(f"  已保存最佳模型：{best_path}")

    # 第五步：训练结束后，加载验证集选出的模型，仅在此时评估测试集。
    # weights_only=True 适合此处的张量/基础类型字典；仍只加载可信文件。
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, confusion = run_epoch(model, loaders["test"], criterion, device)
    print(f"\n最佳轮次：{checkpoint['epoch']}；测试 loss={test_loss:.4f} acc={test_acc:.2%}")
    print(f"混淆矩阵（行=真实，列=预测，类别顺序={classes}）：\n{confusion}")
    # 保存训练曲线的原始数值，后续可以用来画图、检查过拟合。
    report = {"history": history, "best_epoch": checkpoint["epoch"],
              "classes": classes, "test_loss": test_loss, "test_acc": test_acc,
              "confusion_matrix": confusion.tolist()}
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# 多进程 DataLoader 要用主入口保护，避免子进程重复启动训练。
if __name__ == "__main__":
    main()
