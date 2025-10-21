import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# KONFIGURASI DASAR
# ==========================
st.set_page_config(
    page_title="🧠 Image Classification & Object Detection",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model/2208108010063_siti reva retha_Laporan 4_pemograman big data_shift p3.pt")
        classifier = tf.keras.models.load_model("model/model_reva_Laporan 2.h5")
        return yolo_model, classifier
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

yolo_model, classifier = load_models()
st.sidebar.success("✅ Model berhasil dimuat.")

# ==========================
# ANTARMUKA UTAMA
# ==========================
st.title("🧠 Image Classification & Object Detection App")
st.write("Aplikasi ini menggunakan *YOLOv8* untuk deteksi objek dan *TensorFlow* untuk klasifikasi gambar.")

menu = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# PROSES GAMBAR YANG DI-UPLOAD
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Gambar yang Diupload", use_container_width=True)
    st.markdown("---")

    if menu == "Deteksi Objek (YOLO)":
        st.subheader("🔍 Hasil Deteksi Objek (YOLOv8)")
        with st.spinner("Sedang mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()  # hasil deteksi (gambar dengan bounding box)
        st.image(result_img, caption="🧩 Hasil Deteksi Objek", use_container_width=True)

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
    st.warning("📁 Silakan unggah gambar terlebih dahulu untuk mulai.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption("👩‍💻 Dibuat oleh *siti reva retha* — Menggabungkan YOLOv8 & TensorFlow untuk Analisis Gambar.")
