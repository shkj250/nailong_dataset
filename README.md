# 奶龙 / 奶蛙 / 其他图片分类

基于 **PyTorch、ResNet18 和 Streamlit** 的三分类项目，支持训练、命令行预测和浏览器交互，可使用 CPU 运行。

| 类别目录 | 含义 |
| --- | --- |
| `nailong` | 奶龙 |
| `naiwa` | 奶蛙 |
| `other` | 其他：既不是奶龙，也不是奶蛙 |

`other` 是通过负样本训练得到的类别，不是能保证识别所有未知图片的通用拒识机制。

## 项目结构

```text
nailong/
├── README.md
├── requirements.txt
├── train.py                  # 训练、评估、共用模型和预处理
├── predict.py                # 命令行三分类预测
├── web_app.py                # Streamlit 网页分类
├── data/
│   ├── train/
│   │   ├── nailong/
│   │   ├── naiwa/
│   │   └── other/
│   ├── val/
│   │   ├── nailong/
│   │   ├── naiwa/
│   │   └── other/
│   └── test/
│       ├── nailong/
│       ├── naiwa/
│       └── other/
└── outputs/
    ├── best_model.pt         # 可能存在的旧二分类权重，保留但不用于三分类
    ├── metrics.json          # 旧训练指标（如存在）
    └── three_class/
        ├── best_model.pt     # 新训练生成的三分类最佳权重
        └── metrics.json      # 三分类训练和评估结果
```

入口：[训练](<train.py>)、[命令行预测](<predict.py>)、[网页分类](<web_app.py>)。

## 安装

在项目根目录执行以下 Linux 命令；已有可用虚拟环境时直接激活，不必重装。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.6' 'torchvision>=0.21' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

[依赖清单](<requirements.txt>)定义版本范围，不是锁定文件。Python 版本和平台需有兼容的 PyTorch 安装包，参见 [官方安装说明](https://pytorch.org/get-started/locally/)。

## 数据要求

- 训练、验证、测试三个划分都必须包含 `nailong`、`naiwa`、`other`，每类至少有一张支持的图片。
- 图片标签取自父级类别目录。类别顺序来自 `ImageFolder`，会写进权重，预测时读取权重中的顺序，不猜测类别索引。
- 只含其他角色、动物、人物或场景的图片可作为 `other`；含目标角色的混合画面、变色或变装角色应先明确标注规则。
- 同一视频及同图的裁剪、缩放和近似版本应放在同一个划分；由训练图裁剪出的困难负样本不要直接放进验证或测试集。
- 程序默认提示文件名中相同 BV 跨划分的情况，不移动图片。添加 `--ignore-source-overlap` 可静默此提示；检测结果仍写入指标文件，不影响训练运行。
- 不要通过反复查看测试集结果来选择参数。新增图片后必须重新训练，旧权重不会自动学习新数据。

## 训练三分类模型

推荐 CPU 命令：

```bash
python train.py --device cpu --epochs 15 --batch-size 8 --class-weight balanced --ignore-source-overlap --output-dir outputs/three_class_color
```

`balanced` 根据训练集中各类别样本数量的倒数设置损失权重，适合其他类样本相对较少的情况。它不会复制图片，也不会给验证、测试统计加权。若不使用此选项，默认采用普通交叉熵。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data-dir` | `data` | 数据根目录 |
| `--output-dir` | `outputs/three_class` | 三分类输出目录，与旧模型分开 |
| `--device` | `auto` | `auto`、`cpu`、`cuda` 或 `mps` |
| `--epochs` | `15` | 训练轮数 |
| `--batch-size` | `16` | 每批图片数 |
| `--lr` | `0.0001` | 学习率 |
| `--seed` | `42` | 随机种子 |
| `--num-workers` | `0` | 数据加载子进程数 |
| `--class-weight` | `none` | `none` 或 `balanced` |
| `--color-jitter` | `0.4` | 训练颜色扰动强度，范围0到1，0关闭 |
| `--grayscale-prob` | `0.25` | 训练输入随机转灰度的概率，范围0到1 |
| `--ignore-source-overlap` | 关闭 | 静默同BV跨划分的控制台提示，仍记录检测结果 |
| `--no-pretrained` | 关闭 | 不加载 ImageNet 预训练权重 |
| `--overwrite` | 关闭 | 明确允许覆盖所选输出目录的同名权重和指标 |

默认路径相对于脚本所在目录。首次加载 ImageNet 预训练模型可能需要联网；离线且没有缓存时，可添加 `--no-pretrained`，但从随机初始化训练通常需要更多数据和训练时间。

### 降低对黄色的依赖

默认只对训练输入启用颜色增强：

- 随机调整亮度、对比度、饱和度和色相，让同一角色不总以固定的黄色出现。
- 随机将约25%的训练输入转为灰度，保留三通道，让模型有机会更多学习轮廓、眼睛、嘴型等特征。
- 强度为0.4时，亮度/对比度扰动幅度为0.2，饱和度为0.6，色相为0.1；可以降低强度避免扰动过大。
- 只处理内存中的训练副本，不改写数据集图片。验证、测试、网页和命令行预测仍保留原色，处理方式与原模型兼容。

这是降低颜色捷径的训练手段，不保证模型完全不用颜色，也不保证准确率一定提高。已有权重不会自动改变，需要重新训练。推荐命令使用独立的 `outputs/three_class_color` 输出目录，网页中应选择该目录下的新权重；默认预测路径仍是标准三分类目录，使用颜色增强实验权重时需要显式指定：

```bash
python predict.py --device cpu --image "path/to/image.jpg" --checkpoint outputs/three_class_color/best_model.pt
```

做不使用颜色增强的对照实验时，设置 `--color-jitter 0 --grayscale-prob 0`，并使用另一个输出目录。比较验证集宏平均F1和逐类指标，不要只根据训练准确率判断增强是否有效。

### 保存与评估

- 使用三输出分类头，重新微调 ResNet18；不是给旧二分类权重改一个类别名字。
- **按验证集宏平均 F1 选择最佳模型**，相同时优先选择验证损失较低的轮次。宏平均 F1 对三类同等对待，减少少数类被总体准确率掩盖的问题。
- 训练结束后重新加载最佳模型，再评估测试集。
- 指标文件包含训练历史、数据量、类别顺序、3×3 混淆矩阵、逐类精确率/召回率/F1，以及测试宏平均指标。
- 混淆矩阵的行是真实类别，列是预测类别，顺序以指标中的 `classes` 为准。

默认不会覆盖已有结果。新实验建议使用新目录：

```bash
python train.py --device cpu --epochs 15 --batch-size 8 --class-weight balanced --output-dir outputs/three_class_run2
```

确实需要覆盖同名文件时，再显式添加 `--overwrite`。权重不包含优化器状态，不支持断点续训；再次启动训练是一次新训练。

## 网页分类

先完成三分类训练，再启动网页：

```bash
python -m streamlit run web_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

打开 [http://127.0.0.1:8501](http://127.0.0.1:8501)：

1. 选择可信的三分类模型，默认优先使用标准三分类输出目录中的最佳权重。
2. 从测试集中选图、随机抽图，或上传图片。
3. 点击“开始分类”，查看奶龙、奶蛙、其他三个分数及预测结果。

网页会排除旧二分类权重及未通过检查的权重。尚未训练三分类模型时会显示训练提示，不会把旧模型伪装成具有“其他”识别能力的模型。

支持 JPG、PNG、WebP、BMP 和 GIF；GIF 只处理首帧。上传图片仅在本机内存中处理，不写入数据集，也不调用云端分类 API。应用未配置公网认证，不建议直接暴露到互联网。

## 命令行预测

将示例图片路径替换为实际路径：

```bash
python predict.py --device cpu --image "path/to/image.jpg"
```

默认读取三分类权重。其他实验可显式指定：

```bash
python predict.py --device cpu --image "path/to/image.jpg" --checkpoint outputs/three_class_run2/best_model.pt
```

程序核验权重的类别、输出层和类别索引映射，拒绝把旧二分类权重当作三分类模型使用。训练、网页和命令行入口共用[模型、权重检查和预处理](<train.py>)。

```bash
python train.py --help
python predict.py --help
```

## 限制

- 三分类中的“其他”仍可能误判，效果取决于负样本的多样性和困难样本覆盖。
- softmax 分数不是经过校准的真实概率，也不是模型准确率。
- 背景、水印、字幕以及同源泄漏可能使评估分数虚高，不能只追求测试准确率。
- 仅加载自己训练或可信来源的权重；不要仅修改旧权重中的类别列表来转换任务。
- 图片版权归原权利人，公开发布或商业使用前应确认逐图许可及其他所需授权。
