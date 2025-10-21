import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import glob, os

# Optional YOLO import
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="🌙☀️ Bulan & Matahari Vision",
    page_icon="🌞🌙",
    layout="wide"
)

# ======================
# CUSTOM CSS (Night & Day theme + Glassmorphism)
# ======================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1d51 0%, #1a2b64 40%, #f9e6d3 100%);
    font-family: 'Poppins', sans-serif;
    color: #ffffff;
}
h1, h2, h3, h4 { color: #ffdd59; font-weight: 700; }
.block-container { padding-top: 1rem; }
.title-box {
    background: rgba(0,0,0,0.55);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
}
.mode-box {
    background: rgba(255,255,255,0.1);
    padding: 15px;
    border-radius: 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.result-box {
    background: rgba(255,255,255,0.15);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.2);
}
footer {
    text-align: center;
    color: #ffdd59;
    margin-top: 40px;
    font-size: 14px;
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
        return None, "Model .h5 tidak ditemukan"
    try:
        model = tf.keras.models.load_model(path)
        return model, f"Dimuat dari {path}"
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_yolo():
    if not ULTRALYTICS_AVAILABLE:
        return None, "Ultralytics tidak tersedia"
    path = find_first("*.pt")
    if not path:
        return None, "Model YOLO (.pt) tidak ditemukan"
    try:
        model = YOLO(path)
        return model, f"Dimuat dari {path}"
    except Exception as e:
        return None, str(e)

classifier, cls_info = load_classifier()
yolo, yolo_info = load_yolo()

# ======================
# CLASS LABELS & INFO
# ======================
class_names = ["bulan", "matahari"]

celestial_info = {
    "bulan": {"nama": "🌙 Bulan", "deskripsi": "Satelit alami Bumi, memantulkan cahaya Matahari.",
              "fakta": "Bulan mengatur pasang surut air laut dan muncul dalam berbagai fase."},
    "matahari": {"nama": "☀️ Matahari", "deskripsi": "Bintang pusat Tata Surya, sumber energi utama Bumi.",
                 "fakta": "Matahari menyediakan cahaya dan panas yang memungkinkan kehidupan di Bumi."}
}

def preprocess_image(img, model):
    try:
        input_shape = model.input_shape[1:3]
    except:
        input_shape = (224, 224)
    img_resized = img.resize(input_shape)
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
# HEADER
# ======================
st.markdown("<div class='title-box'><h1>🌙☀️ Bulan & Matahari Vision</h1><h4>AI Cosmic Classification Dashboard</h4></div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Pengaturan Model")
if classifier: st.sidebar.success(f"✅ Classifier aktif: {cls_info}")
else: st.sidebar.warning("⚠️ Model .h5 belum ditemukan")

if yolo: st.sidebar.success(f"✅ YOLO aktif: {yolo_info}")
else: st.sidebar.info("YOLO tidak aktif (opsional)")

mode = st.sidebar.radio("Pilih Mode:", ["Klasifikasi Kosmik", "Deteksi Objek", "Filter Gambar", "Analisis Warna"])

# ======================
# MAIN
# ======================
uploaded = st.file_uploader("📤 Unggah gambar bulan atau matahari", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Gambar diunggah", use_container_width=True)
    st.markdown("---")

    # ===== KLASIFIKASI =====
    if mode == "Klasifikasi Kosmik":
        if classifier is None:
            st.error("Model classifier belum dimuat.")
        else:
            with st.spinner("🔎 Mendeteksi objek kosmik..."):
                label, conf = predict_image(classifier, img)
            if label in celestial_info:
                info = celestial_info[label]
                st.success(f"🎯 Hasil: {info['nama']} ({conf*100:.2f}%)")
                st.markdown(f"""
                <div class='result-box'>
                    <b>🌌 Deskripsi:</b> {info['deskripsi']}<br>
                    <b>💡 Fakta menarik:</b> {info['fakta']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"Hasil: {label} ({conf:.2%}) — tidak ditemukan di database.")

    # ===== YOLO =====
    elif mode == "Deteksi Objek":
        if yolo is None:
            st.error("Model YOLO tidak aktif.")
        else:
            with st.spinner("🧠 Mendeteksi objek kosmik..."):
                result = yolo(img)
                st.image(result[0].plot(), caption="Hasil Deteksi", use_container_width=True)

    # ===== FILTER =====
    elif mode == "Filter Gambar":
        opt = st.selectbox("Pilih filter:", ["Asli", "Grayscale", "Blur", "Sharpen", "Edge"])
        if opt == "Grayscale": out = ImageOps.grayscale(img)
        elif opt == "Blur": out = img.filter(ImageFilter.BLUR)
        elif opt == "Sharpen": out = img.filter(ImageFilter.SHARPEN)
        elif opt == "Edge": out = img.filter(ImageFilter.FIND_EDGES)
        else: out = img
        st.image(out, caption=f"Hasil filter: {opt}", use_container_width=True)

    # ===== WARNA =====
    elif mode == "Analisis Warna":
        small = img.resize((120, 120))
        arr = np.array(small).reshape(-1, 3)
        uniq, counts = np.unique((arr//32)*32, axis=0, return_counts=True)
        top = uniq[np.argsort(-counts)[:5]]
        st.write("🌈 Warna dominan:")
        cols = st.columns(5)
        for i, c in enumerate(top):
            hexc = '#%02x%02x%02x' % tuple(c)
            cols[i].markdown(f"<div style='background:{hexc};height:80px;border-radius:10px;'></div>", unsafe_allow_html=True)
            cols[i].write(hexc)

else:
    st.info("📁 Unggah gambar bulan atau matahari untuk mulai menganalisis.")

# ======================
# FOOTER
# ======================
st.markdown("<footer>🌙☀️ Bulan & Matahari Vision — by Reva 💜 | Built with Streamlit & TensorFlow</footer>", unsafe_allow_html=True)
