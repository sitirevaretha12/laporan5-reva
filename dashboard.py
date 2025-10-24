import os
import warnings
warnings.filterwarnings("ignore")  # 🔇 Matikan semua warning Python
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 🔇 Matikan log TensorFlow
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ======================
# IMPORT LIBRARY
# ======================
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import glob

# Coba muat YOLO (tanpa crash)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# ======================
# KONFIGURASI PAGE
# ======================
st.set_page_config(
    page_title="🌙☀️ Celestial Vision Dashboard",
    layout="wide",
    page_icon="✨"
)

# ======================
# STYLE CSS
# ======================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0c1b3f 0%, #1f305e 50%, #f9d29d 100%);
    color: #fff;
    font-family: 'Poppins', sans-serif;
}
.title-box {
    background: rgba(0,0,0,0.45);
    padding: 20px;
    border-radius: 20px;
    text-align:center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}
.result-card {
    background: rgba(255,255,255,0.15);
    padding: 18px;
    border-radius: 16px;
    margin-bottom: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.result-card:hover {
    transform: scale(1.03);
}
footer {
    text-align:center;
    color:#ffec99;
    margin-top:40px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODEL FUNGSI
# ======================
MODEL_FOLDER = "model"

def find_first(pattern):
    files = glob.glob(os.path.join(MODEL_FOLDER, pattern))
    return files[0] if files else None

@st.cache_resource
def load_classifier():
    h5_path = find_first("*.h5")
    if not h5_path:
        return None, "❌ Model .h5 tidak ditemukan"
    try:
        model = tf.keras.models.load_model(h5_path, compile=False)
        return model, f"✅ Classifier dimuat dari: {os.path.basename(h5_path)}"
    except Exception as e:
        return None, f"⚠️ Gagal memuat .h5: {e}"

@st.cache_resource
def load_yolo():
    if not YOLO_AVAILABLE:
        return None, "⚠️ YOLO tidak tersedia"
    pt_path = find_first("*.pt")
    if not pt_path:
        return None, "❌ Model .pt tidak ditemukan"
    try:
        model = YOLO(pt_path)
        return model, f"✅ YOLO dimuat dari: {os.path.basename(pt_path)}"
    except Exception as e:
        return None, f"⚠️ Gagal memuat YOLO: {e}"

classifier, cls_info = load_classifier()
yolo, yolo_info = load_yolo()

# ======================
# LABEL DATA
# ======================
class_names = ["bulan", "matahari"]
celestial_info = {
    "bulan": {"nama":"🌙 Bulan", "deskripsi":"Satelit alami Bumi.", "fakta":"Bulan mengatur pasang surut air laut."},
    "matahari":{"nama":"☀️ Matahari", "deskripsi":"Bintang pusat tata surya.", "fakta":"Menyediakan cahaya & energi kehidupan."}
}

# ======================
# PREPROCESSING
# ======================
def preprocess_image(img, model):
    try:
        shape = model.input_shape[1:3]
    except:
        shape = (224,224)
    img_resized = img.resize(shape)
    arr = image.img_to_array(img_resized)
    arr = np.expand_dims(arr, axis=0)/255.0
    return arr

def predict_image(model, pil_img):
    arr = preprocess_image(pil_img, model)
    preds = model.predict(arr, verbose=0)
    idx = np.argmax(preds)
    label = class_names[idx] if idx < len(class_names) else "unknown"
    conf = float(np.max(preds))
    return label, conf

# ======================
# UI HEADER
# ======================
st.markdown("<div class='title-box'><h1>🌙☀️ Celestial Vision Dashboard</h1><h4>AI Analisis Bulan & Matahari</h4></div>", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.header("⚙️ Status Model")
st.sidebar.info(cls_info)
st.sidebar.info(yolo_info)

# Pilihan mode utama
mode = st.sidebar.radio(
    "Pilih Mode:",
    ["Klasifikasi", "Deteksi Objek", "Filter Gambar", "Analisis Warna"]
)

# ======================
# UPLOAD GAMBAR
# ======================
uploaded = st.file_uploader("📤 Unggah gambar Bulan atau Matahari", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="📷 Gambar diunggah", use_container_width=True)
    st.markdown("---")

    # === MODE 1: KLASIFIKASI ===
    if mode == "Klasifikasi":
        if classifier is None:
            st.error("Model .h5 belum dimuat.")
        else:
            label, conf = predict_image(classifier, img)
            info = celestial_info.get(label, {"nama": label})
            st.markdown(f"""
            <div class='result-card'>
            <h3>{info['nama']} ({conf*100:.2f}%)</h3>
            <b>Deskripsi:</b> {info.get('deskripsi','-')}<br>
            <b>Fakta:</b> {info.get('fakta','-')}
            </div>
            """, unsafe_allow_html=True)

    # === MODE 2: DETEKSI OBJEK ===
    elif mode == "Deteksi Objek":
        if yolo is None:
            st.error("Model YOLO belum dimuat.")
        else:
            results = yolo(img)
            st.image(results[0].plot(), caption="🔍 Hasil Deteksi Objek",
