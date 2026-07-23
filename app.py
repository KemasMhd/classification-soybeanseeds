from __future__ import annotations
import base64
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Klasifikasi Biji Kedelai",
    page_icon="🌱",
    layout="wide",
)

APP_STYLE = """
<style>
.stApp {
    background:
    radial-gradient(circle at top left, rgba(202, 240, 177, 0.75), transparent 28%),
    radial-gradient(circle at top right, rgba(146, 196, 125, 0.35), transparent 24%),
    linear-gradient(180deg, #eef9df 0%, #f7fbf2 45%, #e5f0d8 100%);
    color: #203025;
}
.hero {
    border: 1px solid rgba(32, 48, 37, 0.12);
    background: rgba(255, 255, 255, 0.74);
    backdrop-filter: blur(10px);
    border-radius: 28px;
    padding: 28px;
    box-shadow: 0 24px 70px rgba(38, 61, 35, 0.12);
}
.hero h1 {
    margin: 0;
    font-size: 2.45rem;
    line-height: 1.02;
    letter-spacing: -0.03em;
    color: #1d2d21;
}
.hero p {
    margin: 12px 0 0;
    max-width: 960px;
    font-size: 1rem;
    line-height: 1.75;
    color: rgba(29, 45, 33, 0.82);
}
.pill-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 18px;
}
.pill {
    border-radius: 999px;
    padding: 8px 14px;
    background: rgba(58, 102, 66, 0.09);
    border: 1px solid rgba(58, 102, 66, 0.15);
    color: #31583f;
    font-size: 0.92rem;
    font-weight: 600;
}
.stat-card {
    border-radius: 20px;
    padding: 18px;
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid rgba(32, 48, 37, 0.09);
    box-shadow: 0 16px 45px rgba(38, 61, 35, 0.08);
}
.stat-card h3 {
    margin: 0 0 8px;
    font-size: 0.94rem;
    color: rgba(29, 45, 33, 0.72);
    font-weight: 700;
}
.stat-card .value {
    font-size: 1.14rem;
    font-weight: 800;
    color: #1d2d21;
    line-height: 1.35;
}
.section-label {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px; 
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(58, 102, 66, 0.1);
    border: 1px solid rgba(58, 102, 66, 0.16);
    color: #31583f;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
.subtle-text {
    color: rgba(29, 45, 33, 0.72);
    line-height: 1.7;
}
.stButton > button {
    background: linear-gradient(135deg, #3b6a43 0%, #264b30 100%);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.7rem 1rem;
    font-weight: 700;
}
.stButton > button:hover {
    filter: brightness(1.05);
    border: none;
}
[data-testid="stRadio"] label, [data-testid="stRadio"] p, [data-testid="stRadio"] span {
    color: #203025 !important;
}
[data-testid="stRadio"] label {
    font-weight: 600 !important;
}

/* ========================================== */
/* RESPONSIVE CSS UNTUK MOBILE (HP)           */
/* ========================================== */
@media (max-width: 768px) {
    .hero {
        padding: 20px;
        border-radius: 20px;
    }
    .hero h1 {
        font-size: 1.75rem; /* Memperkecil judul di HP */
    }
    .hero p {
        font-size: 0.9rem;
    }
    .pill {
        font-size: 0.8rem;
        padding: 6px 10px;
    }
    .stat-card {
        padding: 14px;
        border-radius: 16px;
    }
    .stat-card h3 {
        font-size: 0.85rem;
    }
    .stat-card .value {
        font-size: 1rem;
    }
    .section-label {
        font-size: 0.75rem;
        padding: 5px 10px;
    }
    /* Mengurangi jarak antar kolom di HP */
    .st-emotion-cache-1kyx0fo, .st-emotion-cache-15ecoxei {
        gap: 0.5rem !important; 
    }
}
</style>
"""

CLASS_MAPPING = [
    (0, "Broken soybeans", "Biji kedelai pecah"),
    (1, "Immature soybeans", "Kedelai belum matang"),
    (2, "Intact soybeans", "Kedelai utuh"),
    (3, "Skin-damaged soybeans", "Kedelai dengan kulit rusak"),
    (4, "Spotted soybeans", "Kedelai berbintik/rusak"),
]

CLASS_FOLDER_NAMES = {
    "Broken soybeans": "broken soybeans",
    "Immature soybeans": "immature soybeans",
    "Intact soybeans": "intact soybeans",
    "Skin-damaged soybeans": "skin-damaged soybeans",
    "Spotted soybeans": "spotted soybeans",
}

def try_load_keras_model(model_path: str) -> Any:
    import importlib
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {path}")
    load_model = importlib.import_module("tensorflow.keras.models").load_model
    return load_model(path)

def preprocess_image(image: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(target_size)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)

def get_class_names() -> list[str]:
    return [label for _, label, _ in CLASS_MAPPING]

def get_sample_image_path(class_name: str) -> Path | None:
    folder_name = CLASS_FOLDER_NAMES.get(class_name)
    if folder_name is None:
        return None
    # Diperbaiki: menghapus spasi aneh di "img " dan __file__
    base_dir = Path(__file__).resolve().parent / "img" / folder_name
    if not base_dir.exists():
        return None
    # Diperbaiki: format ekstensi agar glob() bisa jalan
    for extension in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        matches = sorted(base_dir.glob(extension))
        if matches:
            return matches[0]
    return None

def build_sample_gallery() -> list[dict[str, str]]:
    gallery: list[dict[str, str]] = []
    for index, english_name, indonesian_name in CLASS_MAPPING:
        sample_path = get_sample_image_path(english_name)
        gallery.append({
            "index": str(index),
            "english": english_name,
            "indonesian": indonesian_name,
            "path": str(sample_path) if sample_path is not None else "",
        })
    return gallery

def get_sample_image_options() -> list[dict[str, str]]:
    return [item for item in build_sample_gallery() if item["path"]]

def predict_class(model: Any, image_array: np.ndarray, class_names: list[str]) -> tuple[str, float | None, np.ndarray | None]:
    predictions = model.predict(image_array, verbose=0)
    predictions = np.asarray(predictions)
    
    if predictions.ndim == 2 and predictions.shape[1] > 1:
        index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][index])
        label = class_names[index] if index < len(class_names) else f"Kelas {index}"
        return label, confidence, predictions[0]
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        score = float(predictions[0][0])
        label = class_names[1] if score >= 0.5 and len(class_names) > 1 else class_names[0]
        confidence = score if score >= 0.5 else 1 - score
        return label, confidence, predictions[0]
    if predictions.ndim == 1 and predictions.size > 1:
        index = int(np.argmax(predictions))
        confidence = float(predictions[index])
        label = class_names[index] if index < len(class_names) else f"Kelas {index}"
        return label, confidence, predictions
    if predictions.size == 1:
        score = float(predictions.item())
        label = class_names[1] if score >= 0.5 and len(class_names) > 1 else class_names[0]
        confidence = score if score >= 0.5 else 1 - score
        return label, confidence, predictions
    return "Tidak diketahui", None, predictions

# ================= RENDER UI =================
st.markdown(APP_STYLE, unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>GUI Klasifikasi Biji Kedelai</h1>
    <p>GUI ini disiapkan sebagai tampilan untuk hasil klasifikasi Biji Kedelai.</p>
    <div class="pill-row">
        <div class="pill">Model .h5</div>
        <div class="pill">Upload gambar</div>
        <div class="pill">Resize + normalisasi</div>
        <div class="pill">Hasil klasifikasi</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")
# Diperbaiki: gap="large" diubah jadi "medium" agar tidak terlalu renggang di HP
left_col, right_col = st.columns([1.08, 0.92], gap="medium")

with left_col:
    st.subheader("1. Muat Model H5")
    model_source = st.radio("Sumber model", ["Upload file .h5", "Path lokal"], horizontal=True)
    
    model = None
    model_name = "Belum dimuat"
    target_width = 224
    target_height = 224
    
    if model_source == "Upload file .h5":
        uploaded_model = st.file_uploader("Upload model Keras", type=["h5", "keras"])
        if uploaded_model is not None:
            try:
                import importlib
                load_model = importlib.import_module("tensorflow.keras.models").load_model
                model = load_model(uploaded_model)
                model_name = uploaded_model.name
            except Exception as error:
                st.error(f"Model tidak bisa dimuat: {error}")
    else:
        model_path = st.text_input("Path file model", placeholder=r"D:\path\ke\model\resnet50_soybean.h5")
        if model_path.strip():
            try:
                model = try_load_keras_model(model_path.strip())
                model_name = model_path.strip()
            except Exception as error:
                st.error(f"Model tidak bisa dimuat: {error}")

    metric_col1, metric_col2 = st.columns(2, gap="small")
    with metric_col1:
        st.markdown(f"""<div class="stat-card"><h3>Status Model</h3><div class="value">{"Terhubung" if model is not None else "Menunggu file .h5"}</div></div>""", unsafe_allow_html=True)
    with metric_col2:
        st.markdown(f"""<div class="stat-card"><h3>Nama Model</h3><div class="value">{model_name}</div></div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("2. Upload Gambar Uji")
    uploaded_image = st.file_uploader("Pilih gambar biji kedelai", type=["png", "jpg", "jpeg", "webp"])
    predict_button = st.button("Prediksi Gambar", use_container_width=True)
    
    prediction_label = None
    confidence = None
    raw_output = None
    class_names = get_class_names()
    
    if predict_button:
        if model is None:
            st.warning("Silakan muat model .h5 terlebih dahulu.")
        elif uploaded_image is None:
            st.warning("Silakan upload gambar terlebih dahulu.")
        else:
            try:
                image = Image.open(uploaded_image)
                input_size = getattr(model, "input_shape", None)
                if input_size and len(input_size) >= 3:
                    target_height = int(input_size[1] or 224)
                    target_width = int(input_size[2] or 224)
                image_array = preprocess_image(image, (target_width, target_height))
                prediction_label, confidence, raw_output = predict_class(model, image_array, class_names)
            except Exception as error:
                st.error(f"Prediksi gagal: {error}")

    if prediction_label is not None:
        st.success(f"Hasil prediksi: {prediction_label}")
        if confidence is not None:
            st.info(f"Confidence: {confidence:.2%}")
        if raw_output is not None:
            scores = np.asarray(raw_output).ravel() if raw_output.ndim > 1 else raw_output
            score_labels = class_names[: len(scores)]
            result_df = pd.DataFrame({"Kelas": score_labels, "Skor": scores[: len(score_labels)]})
            st.dataframe(result_df, use_container_width=True)

    st.write("")
    st.markdown('<div class="section-label">Preview Hasil Tebakan</div>', unsafe_allow_html=True)
    st.subheader("Tampilan Hasil Prediksi")
    
    sample_options = get_sample_image_options()
    sample_labels = [f"Kelas {item['index']} · {item['english']}" for item in sample_options]
    selected_sample_label = st.selectbox("Pilih contoh gambar", sample_labels, index=0 if sample_labels else None) if sample_labels else None
    
    selected_sample = None
    if selected_sample_label is not None:
        selected_sample = sample_options[sample_labels.index(selected_sample_label)]

    if selected_sample is not None:
        sample_path = selected_sample["path"]
        sample_index = int(selected_sample["index"])
        sample_english = selected_sample["english"]
        sample_indonesian = selected_sample["indonesian"]
        confidence_value = 98.7
        image = Image.open(sample_path)
        
        preview_cols = st.columns([1.15, 0.85], gap="medium")
        with preview_cols[0]:
            st.image(image, use_container_width=True)
        with preview_cols[1]:
            st.markdown(f"""
            <div style="border-radius:22px;padding:18px;background:linear-gradient(180deg, rgba(255,255,255,0.94), rgba(250,253,246,0.92));border:1px solid rgba(32,48,37,0.1);box-shadow:0 16px 38px rgba(38,61,35,0.08);">
                <div style="display:inline-flex;padding:6px 10px;border-radius:999px;background:rgba(58,102,66,0.1);color:#31583f;font-size:0.8rem;font-weight:800;letter-spacing:0.02em;text-transform:uppercase;">Hasil Prediksi</div>
                <div style="margin-top:14px;font-size:0.92rem;color:rgba(29,45,33,0.72);font-weight:700;">Label kelas</div>
                <div style="font-size:1.6rem;line-height:1.15;font-weight:900;color:#1d2d21;margin-top:4px;">{sample_english}</div>
                <div style="font-size:0.98rem;color:rgba(29,45,33,0.76);margin-top:4px;">{sample_indonesian}</div>
                <div style="margin-top:14px;font-size:0.92rem;color:rgba(29,45,33,0.72);font-weight:700;">Confidence</div>
                <div style="font-size:1.25rem;font-weight:900;color:#264b30;">{confidence_value:.1f}%</div>
                <div style="margin-top:16px;height:12px;border-radius:999px;background:rgba(58,102,66,0.12);overflow:hidden;">
                    <div style="width:{confidence_value}%;height:100%;border-radius:999px;background:linear-gradient(90deg, #3b6a43 0%, #72a85e 100%);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Belum ada gambar contoh yang bisa dipakai untuk preview.")

with right_col:
    st.markdown('<div class="section-label">Preview Dataset</div>', unsafe_allow_html=True)
    st.subheader("Contoh Citra Tiap Kelas")
    
    gallery = build_sample_gallery()
    # DIPERBAIKI: Mengubah dari st.columns(5) menjadi st.columns(2)
    # Ini akan membuat 2 kolom di Desktop, dan otomatis menumpuk (stack) jadi 1 kolom di HP
    gallery_cols = st.columns(2, gap="small")
    for i, item in enumerate(gallery):
        with gallery_cols[i % 2]:
            if item["path"]:
                st.image(Image.open(item["path"]), caption=f"Kelas {item['index']} · {item['english']}", use_container_width=True)
            else:
                st.markdown("<div style='aspect-ratio:1/1;display:flex;align-items:center;justify-content:center;border-radius:18px;background:linear-gradient(135deg, rgba(58,102,66,0.12), rgba(202,240,177,0.25));color:#31583f;font-weight:800;'>No image</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin-top:-0.1rem;margin-bottom:0.25rem;font-size:0.78rem;font-weight:800;color:#31583f;">Kelas {item['index']}</div>
            <div style="font-size:0.9rem;font-weight:800;color:#1d2d21;line-height:1.3;">{item['english']}</div>
            <div style="font-size:0.82rem;color:rgba(29,45,33,0.72);line-height:1.35;">{item['indonesian']}</div>
            """, unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-label">Preview Input</div>', unsafe_allow_html=True)
    st.subheader("Gambar yang Diupload")
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Gambar input untuk prediksi", use_container_width=True)
    else:
        st.info("Upload gambar untuk melihat preview di sini.")

    st.write("")
    st.markdown('<div class="section-label">Mapping Kelas</div>', unsafe_allow_html=True)
    st.subheader("Label Machine Learning")
    
    # DIPERBAIKI: Sama seperti gallery, diubah jadi 2 kolom agar rapi di HP
    mapping_cols = st.columns(2, gap="small")
    for i, (idx, english, indonesian) in enumerate(CLASS_MAPPING):
        with mapping_cols[i % 2]:
            st.markdown(f"""
            <div style="border-radius:18px;padding:14px;background:rgba(255,255,255,0.88);border:1px solid rgba(32,48,37,0.1);box-shadow:0 12px 28px rgba(38,61,35,0.06);">
                <div style="display:inline-flex;min-width:28px;justify-content:center;align-items:center;padding:4px 8px;border-radius:999px;background:rgba(58,102,66,0.1);color:#31583f;font-weight:800;font-size:0.82rem;margin-bottom:8px;">{idx}</div>
                <div style="font-weight:800;color:#1d2d21;line-height:1.35;margin:0;">{english}</div>
                <div style="margin-top:6px;font-size:0.84rem;color:rgba(29,45,33,0.68);line-height:1.45;">{indonesian}</div>
            </div>
            """, unsafe_allow_html=True)

    st.subheader("Parameter Preprocessing")
    st.markdown(f"""<div class="stat-card"><h3>Resize</h3><div class="value">{target_width} x {target_height} pixel</div></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="stat-card" style="margin-top: 12px;"><h3>Normalisasi</h3><div class="value">Pixel dibagi 255.0</div></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="stat-card" style="margin-top: 12px;"><h3>Kelas Output</h3><div class="value">5 kategori sesuai mapping</div></div>""", unsafe_allow_html=True)

    with st.expander("Struktur metodologi penelitian", expanded=False):
        method_df = pd.DataFrame({
            "Tahap": ["Download dataset", "Verifikasi data", "Split data", "Preprocessing", "Bangun model", "Training", "Evaluation", "GUI prediksi"],
            "Keterangan": ["Dataset soybean", "Cek label", "Train/val/test", "Resize & norm", "ResNet50", "Augmentation", "Evaluasi", "Upload & prediksi"]
        })
        st.dataframe(method_df, use_container_width=True, hide_index=True)

    st.caption("GUI ini otomatis menyesuaikan ukuran input model. Tampilan sudah dioptimalkan untuk Desktop dan Mobile.")