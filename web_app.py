"""奶龙 / 奶蛙 / 其他三分类网页；仅在本机进行 CPU 推理。"""

import io
import random
import threading

from PIL import Image, ImageOps
import streamlit as st
import torch

from train import (
    ROOT, CLASS_NAMES, LegacyCheckpointError, load_trained_model, make_transform,
)

NAMES = CLASS_NAMES
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@st.cache_data(show_spinner=False)
def checkpoint_status(path, modified_ns):
    """只缓存检查结果，文件更新时间变化后重新检查；不把旧两类权重冒充三类。"""
    try:
        # 完整加载并核验骨干参数，避免输出层正常、骨干损坏的文件进入选择列表。
        _, classes = load_trained_model(path, "cpu")
        return "ready", classes
    except LegacyCheckpointError as exc:
        return "legacy", str(exc)
    except Exception as exc:
        return "invalid", str(exc)


@st.cache_resource(show_spinner="正在加载 CPU 三分类模型……")
def load_model(path, modified_ns):
    model, classes = load_trained_model(path, "cpu")
    return model, classes, threading.Lock()


def decode_image(data):
    with Image.open(io.BytesIO(data)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def main():
    st.set_page_config(page_title="奶龙 / 奶蛙 / 其他分类", page_icon="🖼️", layout="wide")
    st.title("🖼️ 奶龙 / 奶蛙 / 其他图片分类")
    st.caption("本地 CPU 三分类 · 测试集选图 / 上传图片 · 首次使用需先训练三分类权重")

    with st.sidebar:
        st.header("模型设置")
        checkpoints, legacy_count, invalid = [], 0, []
        for path in sorted((ROOT / "outputs").rglob("*.pt")):
            try:
                status, detail = checkpoint_status(str(path), path.stat().st_mtime_ns)
            except OSError:
                continue
            if status == "ready":
                checkpoints.append(path)
            elif status == "legacy":
                legacy_count += 1
            else:
                invalid.append((path, detail))
        if legacy_count:
            st.caption(f"已保留 {legacy_count} 个旧二分类权重；它们不能识别“其他”，不列入三分类选择。")
        if invalid:
            with st.expander(f"有 {len(invalid)} 个权重未通过检查"):
                for path, detail in invalid:
                    st.write(f"{path.relative_to(ROOT)}：{detail}")
        if not checkpoints:
            st.warning("尚未找到可用的三分类模型。请先训练，旧二分类权重不能直接转换为三分类。")
            st.code("python train.py --device cpu --epochs 15 --batch-size 8", language="bash")
            st.stop()
        default_path = ROOT / "outputs" / "three_class" / "best_model.pt"
        default_index = checkpoints.index(default_path) if default_path in checkpoints else 0
        model_path = st.selectbox("已训练的三分类模型", checkpoints, index=default_index,
                                  format_func=lambda p: str(p.relative_to(ROOT)))
        st.info("运行设备：CPU\n\n类别：奶龙 / 奶蛙 / 其他")
        st.caption("仅加载可信模型。上传图片只发送到本机应用，不调用云端分类服务。")

    source = st.radio("图片来源", ["测试集图片", "上传本地图片"], horizontal=True)
    data, true_label, image_name = None, None, ""
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
            data, image_name = uploaded.getvalue(), uploaded.name
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
        thumbnail = image.copy()
        thumbnail.thumbnail((900, 600), Image.Resampling.LANCZOS)
        st.image(thumbnail, caption=f"{image_name} · {image.width} × {image.height}")
    with result:
        st.subheader("分类结果")
        if true_label:
            st.caption(f"目录标签：{NAMES[true_label]}（仅用于对照，不参与推理）")
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
                st.error(f"分类失败，请检查三分类权重：{exc}")
        else:
            st.info("图片已就绪，点击“开始分类”。")
        st.caption("分数不是准确率保证。“其他”是训练得到的类别，不能保证正确拒绝所有未知图片。")


if __name__ == "__main__":
    main()
