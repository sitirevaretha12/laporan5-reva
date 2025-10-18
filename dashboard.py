# ============================================
# 🧠 IMAGE CLASSIFICATION & OBJECT DETECTION APP
# ============================================

# ==== NONAKTIFKAN LOG & AKSES GITHUB ULTRALYTICS ====
import os
os.environ["YOLO_VERBOSE"] = "False"
os.environ["ULTRALYTICS_HUB"] = "False"

# ==== IMPORT LIBRARY ====
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ============================================
# KONFIGURASI DASAR
# ============================================
st.set_page_config(
    page_title="🧠 Image Classification & Object Detection",
    page_icon="🤖",
    layout="wide"
)

st.title("🧠 Image Classification & Object Detection Dashboard")
st.markdown("Aplikasi ini menggabungkan **YOLOv8** untuk deteksi objek dan **TensorFlow** untuk klasifikasi gambar.")
st.markdown("---")

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/2208108010063_siti reva retha_laporan 4_pemograman big data_shift p3.pt")  # Model YOLO
    classifier = tf.keras.models.load_model("model/model_reva_laporan 2.h5")  # Model klasifikasi
    return yolo_model, classifier

try:
    yolo_model, classifier = load_models()
    st.sidebar.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ============================================
# SIDEBAR NAVIGASI
# ============================================
st.sidebar.header("🧭 Navigasi")
menu = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.sidebar.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ============================================
# PROSES UTAMA
# ============================================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Gambar yang Diupload", use_container_width=True)
    st.markdown("---")

    # ==== MODE 1: DETEKSI OBJEK ====
    if menu == "Deteksi Objek (YOLO)":
        st.subheader("🔍 Hasil Deteksi Objek (YOLOv8)")
        with st.spinner("Sedang mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()
        st.image(result_img, caption="🧩 Hasil Deteksi Objek", use_container_width=True)
        st.success("✅ Deteksi selesai!")

    # ==== MODE 2: KLASIFIKASI ====
    elif menu == "Klasifikasi Gambar":
        st.subheader("🧾 Hasil Klasifikasi Gambar")
        with st.spinner("Sedang mengklasifikasi gambar..."):
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

        st.success(f"**Prediksi:** {class_index}")
        st.info(f"**Probabilitas:** {confidence:.2%}")
else:
    st.warning("📁 Silakan unggah gambar terlebih dahulu di sidebar untuk mulai.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("👩‍💻 Dibuat oleh **Siti Reva Retha** — Menggabungkan YOLOv8 & TensorFlow untuk analisis gambar cerdas.")
