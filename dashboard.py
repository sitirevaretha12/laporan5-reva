import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import glob, os

# ======================
# OPSIONAL: YOLO
# ======================
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="🌙☀️ Celestial Vision",
    layout="wide",
    page_icon="✨"
)

# ======================
# CSS CUSTOM
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
    transition: transform 0.3s;
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
# HELPER FUNCTIONS
# ======================
MODEL_FOLDER = "model"

def find_first(pattern):
    files = glob.glob(os.path.join(MODEL_FOLDER, pattern))
    return files[0] if files else None

@st.cache_resource
def load_classifier():
    path = find_first("*.h5")
    if not path:
        return None, "Model .h5 tidak ditemukan di folder 'model'"
    try:
        model = tf.keras.models.load_model(path)
        return model, f"✅ Classifier dimuat dari: {path}"
    except Exception as e:
        return None, f"❌ Gagal memuat model .h5: {e}"

@st.cache_resource
def load_yolo():
    if not ULTRALYTICS_AVAILABLE:
        return None, "❌ YOLO tidak tersedia (pastikan paket ultralytics terinstal)"
    path = find_first("*.pt")
    if not path:
        return None, "⚠️ Model YOLO (.pt) tidak ditemukan di folder 'model'"
    try:
        model = YOLO(path)
        return model, f"✅ YOLO dimuat dari: {path}"
    except Exception as e:
        return None, f"❌ Gagal memuat YOLO: {e}"

classifier, cls_info = load_classifier()
yolo, yolo_info = load_yolo()

# ======================
# DATA LABEL
# ======================
class_names = ["bulan", "matahari"]
celestial_info = {
    "bulan": {"nama": "🌙 Bulan", "deskripsi": "Satelit alami Bumi.", "fakta": "Bulan mengatur pasang surut air laut."},
    "matahari": {"nama": "☀️ Matahari", "deskripsi": "Bintang pusat tata surya.", "fakta": "Menyediakan cahaya & energi kehidupan."}
}

# ======================
# PREPROCESS
# ======================
def preprocess_image(img, model):
    try:
        shape = model.input_shape[1:3]
    except:
        shape = (224, 224)
    img_resized = img.resize(shape)
    arr = image.img_to_array(img_resized)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr

def predict_image(model, pil_img):
    arr = preprocess_image(pil_img, model)
    preds = model.predict(arr)
    idx = np.argmax(preds)
    label = class_names[idx] if idx < len(class_names) else "unknown"
    conf = float(np.max(preds))
    return label, conf

# ======================
# UI HEADER
# ======================
st.markdown("<div class='title-box'><h1>🌙☀️ Celestial Vision Dashboard</h1><h4>AI Analisis Bulan & Matahari</h4></div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Pengaturan")
if classifier:
    st.sidebar.success(cls_info)
else:
    st.sidebar.warning("⚠️ Model .h5 belum dimuat atau tidak ditemukan")

if yolo:
    st.sidebar.success(yolo_info)
else:
    st.sidebar.info("YOLO opsional — aktif jika file .pt tersedia")

mode = st.sidebar.radio("Pilih Mode:", ["Klasifikasi", "Deteksi Objek", "Filter Gambar", "Analisis Warna"])

# ======================
# UPLOAD IMAGE
# ======================
uploaded = st.file_uploader("📤 Unggah gambar Bulan/Matahari", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Gambar diunggah", use_container_width=True)
    st.markdown("---")

    # ===== KLASIFIKASI =====
    if mode == "Klasifikasi":
        if classifier is None:
            st.error("Model classifier belum dimuat.")
        else:
            label, conf = predict_image(classifier, img)
            if label in celestial_info:
                info = celestial_info[label]
                st.markdown(f"""
                <div class='result-card'>
                <h3>{info['nama']} ({conf*100:.2f}%)</h3>
                <b>Deskripsi:</b> {info['deskripsi']}<br>
                <b>Fakta:</b> {info['fakta']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"Hasil: {label} ({conf*100:.2f}%)")

    # ===== DETEKSI OBJEK =====
    elif mode == "Deteksi Objek":
        if yolo is None:
            st.error("Model YOLO belum aktif atau tidak ditemukan.")
        else:
            results = yolo(img)
            st.image(results[0].plot(), caption="Hasil Deteksi Objek (YOLO)", use_container_width=True)

    # ===== FILTER GAMBAR =====
    elif mode == "Filter Gambar":
        filter_opt = st.selectbox("Pilih filter:", ["Asli", "Grayscale", "Blur", "Sharpen", "Edge"])
        intensity = st.slider("Intensitas filter", 1, 10, 3)
        if filter_opt == "Grayscale":
            out = ImageOps.grayscale(img)
        elif filter_opt == "Blur":
            out = img.filter(ImageFilter.GaussianBlur(radius=intensity))
        elif filter_opt == "Sharpen":
            out = img.filter(ImageFilter.UnsharpMask(radius=intensity))
        elif filter_opt == "Edge":
            out = img.filter(ImageFilter.FIND_EDGES)
        else:
            out = img
        st.image(out, caption=f"Filter: {filter_opt}", use_container_width=True)

    # ===== ANALISIS WARNA =====
    elif mode == "Analisis Warna":
        small = img.resize((120, 120))
        arr = np.array(small).reshape(-1, 3)
        uniq, counts = np.unique((arr // 32) * 32, axis=0, return_counts=True)
        top = uniq[np.argsort(-counts)[:5]]
        st.write("🌈 Warna dominan:")
        cols = st.columns(5)
        for i, c in enumerate(top):
            hexc = '#%02x%02x%02x' % tuple(c)
            cols[i].markdown(f"<div style='background:{hexc};height:80px;border-radius:12px;'></div>", unsafe_allow_html=True)
            cols[i].write(hexc)
else:
    st.info("📁 Unggah gambar Bulan atau Matahari untuk mulai analisis.")

# ======================
# FOOTER
# ======================
st.markdown("<footer>🌙☀️ Celestial Vision — by Reva 💜 | Streamlit + TensorFlow + YOLO</footer>", unsafe_allow_html=True)
