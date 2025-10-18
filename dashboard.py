import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# ==========================
# KONFIGURASI DASAR
# ==========================
st.set_page_config(
    page_title="🧩 YOLOv8 Object Detection Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #F9FAFB;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    model_path = "model/2208108010063_siti_reva_retha_laporan4_pemograman_big_data_shift_p3.pt"
    model = YOLO(model_path)
    return model

try:
    yolo_model = load_model()
    st.sidebar.success("✅ Model YOLOv8 berhasil dimuat.")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ==========================
# UI
# ==========================
st.title("🧠 Smart Vision Dashboard (YOLOv8)")
st.write("Aplikasi ini menggunakan **YOLOv8** untuk mendeteksi objek secara otomatis dari gambar yang diunggah.")
st.sidebar.header("⚙️ Pengaturan")

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# PROSES DETEKSI
# ==========================
if uploaded_file is not None:
    # Baca gambar
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="📷 Gambar Asli", use_container_width=True)

    with col2:
        st.subheader("🔍 Hasil Deteksi (YOLOv8)")
        with st.spinner("🚀 Sedang mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()  # Hasil dengan bounding box

        # Tampilkan hasil
        st.image(result_img, caption="🧩 Gambar dengan Bounding Box", use_container_width=True)

        # Tampilkan daftar objek yang terdeteksi
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            st.markdown("### 📊 Objek yang Terdeteksi:")
            for i, box in enumerate(boxes):
                label = yolo_model.names[int(box.cls)]
                conf = float(box.conf)
                st.write(f"{i+1}. **{label}** ({conf:.2%})")
        else:
            st.warning("Tidak ada objek terdeteksi.")

else:
    st.info("📁 Silakan unggah gambar untuk mulai deteksi objek.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption("👩‍💻 Dibuat oleh **Siti Reva Retha** — Dashboard YOLOv8 Object Detection dengan Streamlit.")
