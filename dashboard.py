import os
import glob
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# YOLO import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# ==========================
# UI & CSS
# ==========================
st.set_page_config(page_title="🌙☀️ Celestial Vision", layout="wide", page_icon="✨")
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0c1b3f 0%, #1f305e 50%, #f9d29d 100%); color: #fff; font-family: 'Poppins', sans-serif;}
.title-box { background: rgba(0,0,0,0.45); padding: 20px; border-radius: 20px; text-align:center; box-shadow:0 4px 12px rgba(0,0,0,0.5);}
.result-card { background: rgba(255,255,255,0.15); padding:20px; border-radius:16px; margin-bottom:15px; box-shadow:0 4px 12px rgba(0,0,0,0.3); transition: transform 0.3s;}
.result-card:hover { transform: scale(1.03);}
footer { text-align:center; color:#ffec99; margin-top:40px;}
</style>
""", unsafe_allow_html=True)

# ==========================
# Folder Model
# ==========================
MODEL_FOLDER = "model"

def find_first(pattern):
    files = glob.glob(os.path.join(MODEL_FOLDER, pattern))
    return files[0] if files else None

# ==========================
# Load YOLO
# ==========================
@st.cache_resource
def load_yolo():
    if not YOLO_AVAILABLE:
        return None, "⚠️ YOLO tidak tersedia"
    pt_path = find_first("*.pt")
    if not pt_path:
        return None, "❌ Model .pt tidak ditemukan"
    try:
        model = YOLO(pt_path)
        return model, f"✅ YOLO dimuat: {os.path.basename(pt_path)}"
    except Exception as e:
        return None, f"⚠️ Gagal memuat YOLO: {e}"

yolo_model, yolo_info = load_yolo()

# ==========================
# Load multiple classifiers
# ==========================
@st.cache_resource
def load_classifiers():
    classifiers = {}
    h5_files = glob.glob(os.path.join(MODEL_FOLDER, "*.h5"))
    for f in h5_files:
        try:
            model = tf.keras.models.load_model(f, compile=False)
            classifiers[os.path.basename(f)] = model
        except Exception as e:
            print(f"⚠️ Gagal memuat {f}: {e}")
    return classifiers

classifiers = load_classifiers()

# ==========================
# Class Labels & Info
# ==========================
class_names = ["Bulan", "Matahari"]
class_info = {
    "Bulan": {"deskripsi":"Satelit alami Bumi.", "fakta":"Bulan mengatur pasang surut air laut."},
    "Matahari": {"deskripsi":"Bintang pusat tata surya.", "fakta":"Menyediakan cahaya & energi kehidupan."}
}

# ==========================
# UI Header
# ==========================
st.markdown("<div class='title-box'><h1>🌙☀️ Celestial Vision Dashboard</h1><h4>AI Analisis Bulan & Matahari</h4></div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Status Model")
st.sidebar.info(f"Classifier: {len(classifiers)} model ditemukan")
st.sidebar.info(yolo_info)

menu = st.sidebar.radio("Pilih Mode:", ["Klasifikasi Gambar", "Deteksi Objek (YOLO)"])

# Pilih classifier jika ada
if classifiers:
    classifier_name = st.sidebar.selectbox("Pilih Model Klasifikasi:", list(classifiers.keys()))
    classifier = classifiers[classifier_name]
    st.sidebar.success(f"✅ Model dipilih: {classifier_name}")
else:
    st.sidebar.warning("❌ Tidak ada model .h5 ditemukan")
    classifier = None

# ==========================
# Upload Image
# ==========================
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Gambar diunggah", use_container_width=True)
    st.markdown("---")

    if menu == "Klasifikasi Gambar":
        if classifier is None:
            st.error("Model classifier belum dimuat")
        else:
            img_resized = img.resize((224,224))
            arr = image.img_to_array(img_resized)
            arr = np.expand_dims(arr, axis=0)/255.0

            pred = classifier.predict(arr, verbose=0)
            idx = np.argmax(pred)
            label = class_names[idx] if idx < len(class_names) else "Unknown"
            conf = np.max(pred)
            info = class_info.get(label, {"deskripsi":"-", "fakta":"-"})

            st.markdown(f"""
            <div class='result-card'>
            <h3>Hasil Prediksi: {label} ({conf*100:.2f}%)</h3>
            <b>Deskripsi:</b> {info['deskripsi']}<br>
            <b>Fakta Menarik:</b> {info['fakta']}
            </div>
            """, unsafe_allow_html=True)

    elif menu == "Deteksi Objek (YOLO)":
        if yolo_model is None:
            st.error("YOLO belum dimuat")
        else:
            results = yolo_model(img)
            st.image(results[0].plot(), caption="🔍 Hasil Deteksi Objek", use_container_width=True)

else:
    st.info("📁 Silakan unggah gambar untuk mulai analisis.")

# ==========================
# Footer
# ==========================
st.markdown("<footer>🌙☀️ Celestial Vision — by Reva 💜 | Streamlit & TensorFlow</footer>", unsafe_allow_html=True)
