"""ResNet18 三分类训练：奶龙、奶蛙、其他。也提供预测共用的模型与预处理。"""

import argparse
from collections import Counter
import json
import random
import re
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = {"nailong": "奶龙", "naiwa": "奶蛙", "other": "其他（都不是）"}
EXPECTED_CLASSES = tuple(sorted(CLASS_NAMES))
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "three_class"
DEFAULT_CHECKPOINT = DEFAULT_OUTPUT_DIR / "best_model.pt"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class LegacyCheckpointError(ValueError):
    """旧二分类模型不能被当作三分类模型使用。"""


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def augmentation_settings(color_jitter=0.4, grayscale_prob=0.25):
    if not 0 <= color_jitter <= 1 or not 0 <= grayscale_prob <= 1:
        raise ValueError("color_jitter 和 grayscale_prob 必须在 0 到 1 之间。")
    return {
        "color_jitter_strength": color_jitter,
        "brightness": 0.5 * color_jitter,
        "contrast": 0.5 * color_jitter,
        "saturation": 1.5 * color_jitter,
        "hue": 0.25 * color_jitter,
        "grayscale_prob": grayscale_prob,
    }


def make_transform(training=False, color_jitter=0.4, grayscale_prob=0.25):
    if training:
        settings = augmentation_settings(color_jitter, grayscale_prob)
        steps = [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        # 只改变训练输入的副本，降低固定黄色/背景颜色成为捷径的机会。
        if color_jitter > 0:
            steps.append(transforms.ColorJitter(
                brightness=settings["brightness"], contrast=settings["contrast"],
                saturation=settings["saturation"], hue=settings["hue"],
            ))
        if grayscale_prob > 0:
            steps.append(transforms.RandomGrayscale(p=grayscale_prob))
    else:
        # 验证、测试、网页、命令行预测始终保留原色，兼容已有权重。
        steps = [transforms.Resize((224, 224))]
    return transforms.Compose(steps + [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_model(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def read_checkpoint(path):
    """读取可信权重并核验类别及输出层；保留权重中真实的类别顺序。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"没有找到三分类权重：{path}。请先运行 python train.py --device cpu，"
            "或通过 --checkpoint 指定已训练的三分类权重。"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("权重必须包含 model_state_dict 和 classes。")
    classes = checkpoint.get("classes")
    if not isinstance(classes, (list, tuple)) or not all(isinstance(c, str) for c in classes):
        raise ValueError("权重中缺少有效的 classes 类别列表。")
    if len(classes) == 2 and set(classes) == {"nailong", "naiwa"}:
        raise LegacyCheckpointError(
            "这是旧二分类权重，不能判断其他类。请重新训练三分类模型；"
            "默认权重为 outputs/three_class/best_model.pt，旧权重无需删除。"
        )
    if len(classes) != len(EXPECTED_CLASSES) or set(classes) != set(EXPECTED_CLASSES):
        raise ValueError("三分类权重的类别必须且只能包含 nailong、naiwa、other，各一次。")
    if checkpoint.get("num_classes", len(classes)) != len(classes):
        raise ValueError("权重的 num_classes 与类别列表不一致。")
    mapping = checkpoint.get("class_to_idx")
    if mapping is not None and mapping != {label: i for i, label in enumerate(classes)}:
        raise ValueError("权重的 class_to_idx 与类别顺序不一致。")
    if checkpoint.get("architecture", "resnet18") != "resnet18":
        raise ValueError("当前预测入口仅支持 ResNet18 权重。")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("权重缺少有效的 model_state_dict。")
    weight, bias = state.get("fc.weight"), state.get("fc.bias")
    if (not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (3, 512)
            or not isinstance(bias, torch.Tensor) or tuple(bias.shape) != (3,)):
        raise ValueError("权重输出层与三分类类别列表不一致，不能只修改 classes 来转换旧模型。")
    return checkpoint


def load_trained_model(path, device="cpu"):
    checkpoint = read_checkpoint(path)
    classes = list(checkpoint["classes"])
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model, classes


def load_datasets(data_dir, color_jitter=0.4, grayscale_prob=0.25):
    """要求三个划分均含三类非空图片目录，不擅自拆分用户的数据。"""
    splits = {}
    for split in ("train", "val", "test"):
        directory = Path(data_dir) / split
        if not directory.is_dir():
            raise FileNotFoundError(f"缺少数据划分目录：{directory}")
        actual = {p.name for p in directory.iterdir() if p.is_dir()}
        expected = set(EXPECTED_CLASSES)
        if actual != expected:
            missing, extra = sorted(expected - actual), sorted(actual - expected)
            raise ValueError(
                f"{split} 必须包含 nailong、naiwa、other 三个类别目录；"
                f"缺少：{missing or '无'}；多余：{extra or '无'}。"
            )
        try:
            splits[split] = datasets.ImageFolder(
                directory, transform=make_transform(
                    training=(split == "train"), color_jitter=color_jitter,
                    grayscale_prob=grayscale_prob,
                )
            )
        except FileNotFoundError as exc:
            raise ValueError(f"{split} 的每个类别都必须至少包含一张支持的图片。") from exc
    for split, dataset in splits.items():
        if dataset.class_to_idx != splits["train"].class_to_idx:
            raise ValueError(f"{split} 的类别映射与训练集不一致。")
    return splits


def find_source_overlap(splits):
    """只检查文件名中可识别的 BV，不等于全量感知去重或来源审计。"""
    sources = {}
    for split, dataset in splits.items():
        for filename, _ in dataset.samples:
            match = re.search(r"BV[A-Za-z0-9]{10}", Path(filename).name)
            if match:
                sources.setdefault(match.group(), set()).add(split)
    return {bv: sorted(names) for bv, names in sorted(sources.items()) if len(names) > 1}


def classification_report(confusion, classes):
    """混淆矩阵行=真实类别、列=预测类别；无需额外的 sklearn 依赖。"""
    matrix = torch.as_tensor(confusion, dtype=torch.float64)
    if tuple(matrix.shape) != (len(classes), len(classes)):
        raise ValueError("混淆矩阵尺寸与类别数量不一致。")
    per_class = {}
    for i, label in enumerate(classes):
        tp = float(matrix[i, i])
        support, predicted = float(matrix[i].sum()), float(matrix[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision, "recall": recall, "f1": f1, "support": int(support),
        }
    return {
        "per_class": per_class,
        "macro_precision": sum(v["precision"] for v in per_class.values()) / len(classes),
        "macro_recall": sum(v["recall"] for v in per_class.values()) / len(classes),
        "macro_f1": sum(v["f1"] for v in per_class.values()) / len(classes),
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total_samples = 0.0, 0, 0
    num_classes = model.fc.out_features
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)  # [batch_size, num_classes]，不提前做 softmax。
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            predictions = logits.argmax(dim=1)
            n = labels.size(0)
            total_loss += loss.item() * n
            total_correct += (predictions == labels).sum().item()
            total_samples += n
            indices = labels.detach().cpu() * num_classes + predictions.detach().cpu()
            confusion += torch.bincount(indices, minlength=num_classes ** 2).reshape(
                num_classes, num_classes
            )
    if not total_samples:
        raise ValueError("数据划分为空，无法训练或评估。")
    return total_loss / total_samples, total_correct / total_samples, confusion


def parse_args():
    parser = argparse.ArgumentParser(description="奶龙 / 奶蛙 / 其他三分类训练")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--no-pretrained", action="store_true", help="不加载 ImageNet 预训练权重")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                        help="balanced 按训练类频率的倒数加权，不改变验证/测试指标")
    parser.add_argument("--color-jitter", type=float, default=0.4,
                        help="训练颜色扰动强度，范围0到1；0关闭颜色扰动")
    parser.add_argument("--grayscale-prob", type=float, default=0.25,
                        help="训练图片随机转灰度的概率，范围0到1；0关闭")
    parser.add_argument("--ignore-source-overlap", action="store_true",
                        help="不打印同BV跨划分警告，仍将检测结果写入指标文件")
    parser.add_argument("--overwrite", action="store_true", help="明确允许覆盖所选输出目录的同名文件")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.num_workers < 0:
        parser.error("epochs、batch-size、lr 必须大于 0，num-workers 不能小于 0")
    try:
        augmentation_settings(args.color_jitter, args.grayscale_prob)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    augmentation = augmentation_settings(args.color_jitter, args.grayscale_prob)
    splits = load_datasets(args.data_dir, args.color_jitter, args.grayscale_prob)
    classes = splits["train"].classes
    class_counts = {
        split: {label: Counter(ds.targets)[i] for i, label in enumerate(classes)}
        for split, ds in splits.items()
    }
    best_path, metrics_path = args.output_dir / "best_model.pt", args.output_dir / "metrics.json"
    if not args.overwrite and (best_path.exists() or metrics_path.exists()):
        raise FileExistsError("输出目录已有权重或指标。请换用 --output-dir，或确认后添加 --overwrite。")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaders = {
        split: DataLoader(ds, batch_size=args.batch_size, shuffle=(split == "train"),
                          num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
        for split, ds in splits.items()
    }
    print(f"设备：{device}；类别映射：{splits['train'].class_to_idx}")
    print("各划分类别数量：", class_counts)
    print("训练颜色增强（验证和预测不使用）：", augmentation)
    source_overlap = find_source_overlap(splits)
    if source_overlap and not args.ignore_source_overlap:
        print(f"警告：{len(source_overlap)} 个文件名可识别的 BV 跨划分，可能存在同源泄漏：")
        for bv, names in source_overlap.items():
            print(f"  {bv}: {', '.join(names)}")
        print("建议按来源视频分组划分；程序不会自动移动图片。此检查不能识别所有近似图。")
    model = build_model(len(classes), pretrained=not args.no_pretrained).to(device)
    weights = None
    if args.class_weight == "balanced":
        counts = class_counts["train"]
        weights = torch.tensor([len(splits["train"]) / (len(classes) * counts[c])
                                for c in classes], dtype=torch.float32, device=device)
        print("训练损失类别权重：", dict(zip(classes, weights.cpu().tolist())))
    train_criterion, eval_criterion = nn.CrossEntropyLoss(weight=weights), nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_score, best_loss, history = -1.0, float("inf"), []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _ = run_epoch(model, loaders["train"], train_criterion, device, optimizer)
        val_loss, val_acc, matrix = run_epoch(model, loaders["val"], eval_criterion, device)
        val_report = classification_report(matrix, classes)
        score = val_report["macro_f1"]
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss={train_loss:.4f} acc={train_acc:.2%} | "
              f"val loss={val_loss:.4f} acc={val_acc:.2%} macro-F1={score:.4f}")
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc, val_macro_f1=score))
        if score > best_score or (score == best_score and val_loss < best_loss):
            best_score, best_loss = score, val_loss
            torch.save({
                "model_state_dict": model.state_dict(), "classes": classes,
                "class_to_idx": splits["train"].class_to_idx, "num_classes": len(classes),
                "epoch": epoch, "val_acc": val_acc, "val_macro_f1": score,
                "architecture": "resnet18", "selection_metric": "val_macro_f1",
                "training_augmentation": augmentation,
            }, best_path)
            print(f"  已保存三分类最佳模型：{best_path}")
    model, classes = load_trained_model(best_path, device)
    test_loss, test_acc, confusion = run_epoch(model, loaders["test"], eval_criterion, device)
    test_report = classification_report(confusion, classes)
    checkpoint = read_checkpoint(best_path)
    print(f"\n最佳轮次：{checkpoint['epoch']}；测试 loss={test_loss:.4f} acc={test_acc:.2%} "
          f"macro-F1={test_report['macro_f1']:.4f}")
    print(f"混淆矩阵（行=真实，列=预测，类别顺序={classes}）：\n{confusion}")
    for label, values in test_report["per_class"].items():
        print(f"{CLASS_NAMES[label]}: precision={values['precision']:.2%}, "
              f"recall={values['recall']:.2%}, F1={values['f1']:.4f}, n={values['support']}")
    report = {
        "history": history, "best_epoch": checkpoint["epoch"], "classes": classes,
        "selection_metric": "val_macro_f1", "best_val_macro_f1": checkpoint["val_macro_f1"],
        "test_loss": test_loss, "test_acc": test_acc, "confusion_matrix": confusion.tolist(),
        "test_report": test_report, "dataset_counts": class_counts,
        "filename_bv_overlap": source_overlap,
        "training": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                     "seed": args.seed, "class_weight": args.class_weight,
                     "pretrained": not args.no_pretrained,
                     "augmentation": augmentation,
                     "source_overlap_warning_suppressed": args.ignore_source_overlap},
    }
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
