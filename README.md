# 奶龙 / 奶蛙图片分类

基于 **PyTorch、ResNet18 和 Streamlit** 的图片二分类项目，支持模型训练、命令行预测和浏览器交互。可使用 CPU 运行，无需配置 CUDA。

## 功能

- 使用 ImageNet 预训练的 ResNet18 进行迁移学习。
- 根据验证集表现保存最佳模型，并在测试集上评估。
- 对单张图片进行命令行预测。
- 在网页中选择测试图片、随机抽图或上传本地图片。
- 展示预测类别及两类分数。

类别目录约定：`nailong` 表示奶龙，`naiwa` 表示奶蛙。

## 项目结构

```text
nailong/
├── README.md
├── requirements.txt
├── train.py              # 训练、评估、模型结构与预处理
├── predict.py            # 命令行单图预测
├── web_app.py            # Streamlit 网页入口
├── data/
│   ├── train/
│   │   ├── nailong/
│   │   └── naiwa/
│   ├── val/
│   │   ├── nailong/
│   │   └── naiwa/
│   └── test/
│       ├── nailong/
│       └── naiwa/
└── outputs/
    ├── best_model.pt     # 训练生成的最佳权重
    └── metrics.json      # 训练记录及评估结果
```

主要入口：[训练](<train.py>)、[命令行预测](<predict.py>)、[网页分类](<web_app.py>)。

## 安装

以下命令适用于 Linux，在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 安装配套的 CPU 版 PyTorch 和 torchvision。
python -m pip install 'torch>=2.6' 'torchvision>=0.21' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

已有可用虚拟环境时，直接激活即可，不必重新创建或重复安装依赖。

[依赖清单](<requirements.txt>)定义版本范围，不是精确锁定文件。Python 版本和系统架构需有对应的 PyTorch 安装包，参见 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)。

## 准备数据

图片按照项目结构中的“划分 / 类别”目录存放，三个划分都必须包含两个类别：

| 划分 | 用途 |
| --- | --- |
| `train` | 更新模型参数 |
| `val` | 比较模型表现、选择最佳轮次 |
| `test` | 最终评估 |

注意事项：

- 检查图片标签，移除损坏、无关或主体不清晰的图片。
- 同一视频、同一场景的近似图片尽量放在同一划分，避免数据泄漏。
- 不要根据测试集结果反复调整模型参数。
- 新增图片不会自动更新已有模型，需要重新训练。

## 训练

使用 CPU 训练：

```bash
python train.py --device cpu --epochs 15 --batch-size 8 --lr 0.0001
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--device` | `auto` | `auto`、`cpu`、`cuda` 或 `mps` |
| `--epochs` | `15` | 训练轮数 |
| `--batch-size` | `16` | 每批图片数 |
| `--lr` | `0.0001` | 学习率 |
| `--data-dir` | `data` | 数据根目录 |
| `--output-dir` | `outputs` | 权重与指标输出目录 |
| `--seed` | `42` | 随机种子 |
| `--num-workers` | `0` | 数据加载子进程数 |
| `--no-pretrained` | 关闭 | 不加载 ImageNet 预训练权重 |

默认路径相对于脚本所在目录。首次使用预训练模型可能需要联网下载权重；没有网络且没有缓存时，可添加 `--no-pretrained` 从随机初始化开始训练。

训练后生成：

- [最佳权重](<outputs/best_model.pt>)：模型参数、类别顺序和最佳轮次等信息。
- [评估记录](<outputs/metrics.json>)：各轮训练与验证指标，以及测试结果。

**相同输出目录中的同名文件会被覆盖。** 建议为新实验指定独立目录：

```bash
python train.py --device cpu --epochs 15 --batch-size 8 --output-dir outputs/run2
```

权重不含优化器状态，不支持断点续训。重新执行训练命令会开始一次新的训练。

## 网页分类

准备好训练权重后，执行：

```bash
python -m streamlit run web_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

打开 [http://127.0.0.1:8501](http://127.0.0.1:8501)：

1. 在侧边栏选择输出目录中的可信模型。
2. 选择测试集图片、随机抽图，或上传图片。
3. 点击“开始分类”，查看预测类别与分数。

网页支持 JPG、PNG、WebP、BMP 和 GIF；GIF 只处理首帧，不做视频识别。上传图片在内存中处理，不写入数据集，也不调用云端分类 API。

按 `Ctrl + C` 停止终端中启动的服务。端口被占用时，可改用其他端口。应用未配置公网认证，不建议直接暴露到互联网。

## 命令行预测

将示例路径替换为待分类图片的实际路径：

```bash
python predict.py --device cpu --image "path/to/image.jpg"
```

使用其他实验的权重：

```bash
python predict.py --device cpu --image "path/to/image.jpg" --checkpoint outputs/run2/best_model.pt
```

查看完整参数：

```bash
python train.py --help
python predict.py --help
```

## 原理与限制

训练时对图片进行随机裁剪和水平翻转；评估与预测时固定缩放至 224 × 224，再按 ImageNet 均值和标准差归一化。三个入口共用[模型与预处理定义](<train.py>)，避免训练和推理条件不一致。

- 模型只区分两个类别，不具备“都不是”的拒识能力。
- 显示的 softmax 分数不是经过校准的真实概率，也不等于准确率。
- 背景、字幕、水印及近似重复可能影响泛化能力，应重视标签质量与数据划分。
- 仅加载自己训练或可信来源的模型权重。
- 图片版权归原权利人；本项目不为图片授予新的开放许可，公开发布或商业使用前应确认授权。
