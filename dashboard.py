import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps
import glob
import os

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="🌙 Sun & Moon Classifier", layout="wide")

# ==========================
# CSS THEME (Night Sky Style)
# ==========================
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(160deg, #0b132b 0%, #1c2541 40%, #3a506b 80%, #5bc0be 100%);
            color: #ffffff;
            font-family: 'Poppins', sans-serif;
        }
        .title { text-align:center; color:#f9f871; font-size:42px; font-weight:800; margin-top:10px; }
        .subtitle { text-align:center; color:#b5c7f3; margin-bottom:25px; }
        .result-box { background: rgba(255,255,255,0.12); padding:18px; border-radius:16px; box-shadow: 0 4px 16px rgba(0,0,0,0.3); }
        footer { text-align:center; color:#b5c7f3; margin-top:35px; padding:10px; border-top: 1px solid rgba(255,255,255,0.2); }
    </style>
""", unsafe_allow_html=True)

# ==========================
# FUNGSI: Temukan model .h5
# ==========================
MODEL_FOLDER = "model"

def find_first(pattern):
    files = glob.glob(os.path.join(MODEL_FOLDER, pattern))
    return files[0] if files else None

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    h5_path = find_first("*.h5")
    if not h5_path:
        return None, "no_model"
    try:
        model = tf.keras.models.load_model(h5_path)
        return model, h5_path
    except Exception as e:
        return None, f"error:{e}"

model, info = load_model()

# ==========================
# CLASS LABELS
# ==========================
class_names = ["bulan", "matahari"]

# ==========================
# INFO DATA
# ==========================
object_info = {
    "bulan": {
        "nama": "Bulan 🌕",
        "tipe": "Satelit alami Bumi",
        "ciri": "Berwarna putih keabu-abuan, tampak bersinar di malam hari",
        "fakta": "Bulan tidak memancarkan cahaya sendiri — ia memantulkan cahaya Matahari."
    },
    "matahari": {
        "nama": "Matahari ☀️",
        "tipe": "Bintang di pusat tata surya",
        "ciri": "Berwarna kuning terang, sangat panas dan memancarkan cahaya kuat",
        "fakta": "Cahaya Matahari membutuhkan sekitar 8 menit 20 detik untuk mencapai Bumi."
    }
}

# ==========================
# HEADER
# ==========================
st.markdown("<div class='title'>🌙 Sun & Moon Vision</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Klasifikasi Gambar Bulan & Matahari — Model Stabil dan Ringan</div>", unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("🔧 Status Model")
if model is None:
    if info == "no_model":
        st.sidebar.error("❌ Tidak ditemukan file .h5 di folder 'model/'.")
    else:
        st.sidebar.error(f"❌ Gagal memuat model: {info}")
else:
    st.sidebar.success(f"✅ Model berhasil dimuat dari:\n{info}")

# ==========================
# UNGGAH GAMBAR
# ==========================
uploaded_file = st.file_uploader("📤 Unggah gambar Bulan atau Matahari (.jpg .jpeg .png)", type=["jpg", "jpeg", "png"])

# ==========================
# PREPROCESS
# ==========================
def preprocess_image(pil_img, size=(224, 224)):
    img_resized = pil_img.resize(size)
    arr = image.img_to_array(img_resized)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr

def predict_image(model, pil_img):
    arr = preprocess_image(pil_img)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    label = class_names[idx] if idx < len(class_names) else "unknown"
    return label, confidence

# ==========================
# MAIN PROCESS
# ==========================
if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"❌ Gagal membuka gambar: {e}")
        st.stop()

    st.image(img, caption="📸 Gambar yang diunggah", use_container_width=True)
    st.markdown("---")

    if model is None:
        st.error("Model tidak tersedia. Letakkan file .h5 di folder 'model/'.")
    else:
        with st.spinner("🔮 Menganalisis gambar..."):
            try:
                label, conf = predict_image(model, img)
            except Exception as e:
                st.error(f"Error saat prediksi: {e}")
                st.stop()

        if label not in object_info:
            st.warning(f"Prediksi: {label} (data tidak lengkap). Confidence: {conf:.2%}")
        else:
            info_obj = object_info[label]
            st.success(f"🌟 Teridentifikasi: {info_obj['nama']} — Confidence: {conf*100:.2f}%")
            st.markdown(f"""
            <div class='result-box'>
                <h3>{info_obj['nama']}</h3>
                <b>🪐 Tipe:</b> {info_obj['tipe']}<br>
                <b>🌈 Ciri-ciri:</b> {info_obj['ciri']}<br>
                <b>💡 Fakta menarik:</b> {info_obj['fakta']}
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("📁 Unggah gambar untuk memulai klasifikasi. Pastikan file model (.h5) sudah ada di folder 'model/'.")

# ==========================
# FOOTER
# ==========================
st.markdown("""
<footer>
    🌌 <b>Sun & Moon Vision</b> • by Repa Cantikk 🪐<br>
    Letakkan model klasifikasi kamu di folder <code>model/</code> (format .h5)
</footer>
""", unsafe_allow_html=True)
