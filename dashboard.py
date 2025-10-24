import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# ==========================
# STYLE CSS
# ==========================
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
    padding: 20px;
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

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/2208108010063_siti reva retha_laporan 4_pemograman big data_shift p3.pt")
    classifier = tf.keras.models.load_model("model/model_reva_laporan 2.h5", compile=False)
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# LABEL DATA
# ==========================
class_names = ["Bulan", "Matahari"]
class_info = {
    "Bulan": {
        "deskripsi": "Satelit alami Bumi yang mengatur pasang surut laut.",
        "fakta": "Bulan mempengaruhi gravitasi dan rotasi Bumi."
    },
    "Matahari": {
        "deskripsi": "Bintang pusat tata surya yang menyediakan cahaya dan energi.",
        "fakta": "Matahari adalah sumber utama energi kehidupan di Bumi."
    }
}

# ==========================
# UI HEADER
# ==========================
st.markdown("<div class='title-box'><h1>🌙☀️ Celestial Vision Dashboard</h1><h4>AI Analisis Bulan & Matahari</h4></div>", unsafe_allow_html=True)

# Sidebar Menu
menu = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Gambar yang diunggah", use_container_width=True)
    st.markdown("---")

    if menu == "Deteksi Objek (YOLO)":
        results = yolo_model(img)
        st.image(results[0].plot(), caption="🔍 Hasil Deteksi Objek", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        # Preprocessing
        img_resized = img.resize((224,224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)/255.0

        # Prediksi
        prediction = classifier.predict(img_array, verbose=0)
        class_idx = np.argmax(prediction)
        class_label = class_names[class_idx]
        confidence = np.max(prediction)

        # Tampilkan hasil dalam card
        info = class_info[class_label]
        st.markdown(f"""
        <div class='result-card'>
        <h3>Hasil Prediksi: {class_label} ({confidence*100:.2f}%)</h3>
        <b>Deskripsi:</b> {info['deskripsi']}<br>
        <b>Fakta Menarik:</b> {info['fakta']}
        </div>
        """, unsafe_allow_html=True)

# ==========================
# FOOTER
# ==========================
st.markdown("<footer>🌙☀️ Celestial Vision — by Reva 💜 | Streamlit & TensorFlow</footer>", unsafe_allow_html=True)
