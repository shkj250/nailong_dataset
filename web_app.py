"""奶龙 / 奶蛙 Web 分类界面。

启动：python -m streamlit run web_app.py --server.address 127.0.0.1
仅需 CPU；与训练脚本复用网络和预处理，不依赖桌面窗口组件。
"""

import io
import random
import threading

from PIL import Image, ImageOps
import streamlit as st
import torch

from train import ROOT, build_model, make_transform

NAMES = {"nailong": "奶龙", "naiwa": "奶蛙"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@st.cache_resource(show_spinner="正在加载 CPU 模型……")
def load_model(path, modified_ns):
    """缓存模型，交互时不重复加载；权重更新时间变化则重新加载。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    if len(classes) != 2 or set(classes) != set(NAMES):
        raise ValueError("权重类别必须是 nailong 和 naiwa。")
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    # 不同浏览器会话共用缓存模型，通过锁串行推理，避免 CPU 争抢。
    return model, classes, threading.Lock()


def decode_image(data):
    """内存中解码，不将上传文件写入项目，也不修改原始图片。"""
    with Image.open(io.BytesIO(data)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def main():
    st.set_page_config(page_title="奶龙 / 奶蛙分类", page_icon="🖼️", layout="wide")
    st.title("🖼️ 奶龙 / 奶蛙图片分类")
    st.caption("本地 CPU 推理 · 测试集选图 / 上传自己的图片 · 无需重新训练")

    with st.sidebar:
        st.header("模型设置")
        # 只允许在项目输出目录中选择已有权重，不接受上传不可信的模型。
        checkpoints = sorted((ROOT / "outputs").rglob("*.pt"))
        if not checkpoints:
            st.error("没有找到权重，请先训练模型。")
            st.stop()
        default_path = ROOT / "outputs" / "best_model.pt"
        default_index = checkpoints.index(default_path) if default_path in checkpoints else 0
        model_path = st.selectbox("已训练的模型", checkpoints, index=default_index,
                                  format_func=lambda p: str(p.relative_to(ROOT)))
        st.info("运行设备：CPU\n\n请选择自己训练的可信权重。")
        st.caption("上传图片只会发送到本机运行的应用，不调用云端分类服务。")

    source = st.radio("图片来源", ["测试集图片", "上传本地图片"], horizontal=True)
    data = None
    true_label = None
    image_name = ""
    if source == "测试集图片":
        test_dir = ROOT / "data" / "test"
        paths = sorted(p for p in test_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not paths:
            st.warning("测试集中没有图片，请切换到上传本地图片。")
            st.stop()
        options = [str(p.relative_to(test_dir)) for p in paths]
        if st.session_state.get("test_image") not in options:
            st.session_state.test_image = options[0]
        # 在 selectbox 创建之前更新它的状态，保证随机按钮与下拉框一致。
        if st.button("🎲 随机选一张"):
            alternatives = [name for name in options if name != st.session_state.test_image]
            st.session_state.test_image = random.choice(alternatives or options)
        selected = st.selectbox("选择测试集图片", options, key="test_image")
        path = paths[options.index(selected)]
        try:
            data = path.read_bytes()
        except OSError as exc:
            st.error(f"无法读取图片：{exc}")
            st.stop()
        image_name = selected
        true_label = path.parent.name if path.parent.name in NAMES else None
    else:
        uploaded = st.file_uploader("点击选择文件，或把图片拖到这里",
                                    type=["jpg", "jpeg", "png", "webp", "bmp", "gif"])
        if uploaded is not None:
            data = uploaded.getvalue()
            image_name = uploaded.name

    if data is None:
        st.info("请先上传一张图片。")
        st.stop()
    try:
        image = decode_image(data)
    except Exception as exc:
        st.error(f"无法解码图片，请选择有效的图片文件：{exc}")
        st.stop()

    preview, result = st.columns([3, 2], gap="large")
    with preview:
        st.subheader("图片预览")
        # 预览等比缩小，模型推理仍使用原始解码图片。
        thumbnail = image.copy()
        thumbnail.thumbnail((900, 600), Image.Resampling.LANCZOS)
        st.image(thumbnail, caption=f"{image_name} · {image.width} × {image.height}")
    with result:
        st.subheader("分类结果")
        if true_label:
            st.caption(f"目录标签：{NAMES[true_label]}（仅用于对照，不参与推理）")
        # Streamlit 每次交互重新运行；切换图片或模型后，旧结果自动清除。
        if st.button("开始分类", type="primary", use_container_width=True):
            try:
                model, classes, lock = load_model(str(model_path), model_path.stat().st_mtime_ns)
                with st.spinner("正在使用 CPU 分类……"):
                    batch = make_transform(training=False)(image).unsqueeze(0)
                    with lock, torch.inference_mode():
                        scores = torch.softmax(model(batch), dim=1)[0].tolist()
                predicted = classes[max(range(len(scores)), key=scores.__getitem__)]
                st.success(f"预测结果：{NAMES[predicted]}")
                for label, score in zip(classes, scores):
                    st.write(f"**{NAMES[label]}**　{score:.2%}")
                    st.progress(min(1.0, max(0.0, score)))
                if true_label:
                    if predicted == true_label:
                        st.success("与目录标签一致 ✓")
                    else:
                        st.warning("与目录标签不一致")
            except Exception as exc:
                st.error(f"分类失败，请检查模型文件：{exc}")
        else:
            st.info("图片已就绪，点击“开始分类”。")
        st.caption("分数为 softmax 输出，不是准确率保证。模型只会在奶龙和奶蛙中选择，不能判断“两类都不是”。")


if __name__ == "__main__":
    main()
