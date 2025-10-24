import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import glob, os

# ======================
# YOLO Setup
# ======================
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

# Disable GPU if error occurs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ======================
# Page Config
# ======================
st.set_page_config(
    page_title="🌙☀️ Celestial Vision",
    layout="wide",
    page_icon="✨"
)

# ======================
# CSS Custom
# ======================
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #0c1b3f 0%, #1f305e 50%, #f9d29d 100%); color: #fff; font-family: 'Poppins', sans-serif;}
.title-box {background: rgba(0,0,0,0.45); padding: 20px; border-radius: 20px; text-align:center; box-shadow: 0 4px 12px rgba(0,0,0,0.5);}
.result-card {background: rgba(255,255,255,0.15); padding: 18px; border-radius: 16px; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); transition: transform 0.3s;}
.result-card:hover {transform: scale(1.03);}
footer {text-align:center; color:#ffec99; margin-top:40px;}
</style>
""", unsafe_allow_html=True)

# ======================
# Helper Functions
# ======================
MODEL_FOLDER = "model"

def find_first(pattern):
    files = glob.glob(os.path.join(MODEL_FOLDER, pattern))
    return files[0] if files else None

@st.cache_resource
def load_classifier():
    path = find_first("*.h5")
    if not path: return None, "Model .h5 tidak ditemukan"
    try:
        model = tf.keras.models.load_model(path)
        return model, f"Dimuat dari {path}"
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_yolo():
    if not ULTRALYTICS_AVAILABLE: return None, "YOLO tidak tersedia"
    path = find_first("*.pt")
    if not path: return None, "Model .pt tidak ditemukan"
    try:
        model = YOLO(path)
        return model, f"Dimuat dari {path}"
    except Exception as e:
        return None, str(e)

classifier, cls_info = load_classifier()
yolo, yolo_info = load_yolo()

# ======================
# Class Names & Info
# ======================
class_names = ["bulan","matahari"]
celestial_info = {
    "bulan": {"nama":"🌙 Bulan", "deskripsi":"Satelit alami Bumi.", "fakta":"Bulan mengatur pasang surut air laut."},
    "matahari":{"nama":"☀️ Matahari", "deskripsi":"Bintang pusat tata surya.", "fakta":"Menyediakan cahaya & energi kehidupan."}
}

# ======================
# Image Preprocess
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
    preds = model.predict(arr)
    idx = np.argmax(preds)
    label = class_names[idx] if idx < len(class_names) else "unknown"
    conf = float(np.max(preds))
    return label, conf

# ======================
# UI Header
# ======================
st.markdown("<div class='title-box'><h1>🌙☀️ Celestial Vision Dashboard</h1><h4>AI Analisis Bulan & Matahari</h4></div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Pengaturan")
if classifier: st.sidebar.success(f"✅ Classifier aktif: {cls_info}")
else: st.sidebar.warning("⚠️ Classifier belum dimuat")

if yolo: st.sidebar.success(f"✅ YOLO aktif: {yolo_info}")
else: st.sidebar.info("YOLO opsional")

mode = st.sidebar.radio("Pilih Mode:", ["Klasifikasi", "Deteksi Objek", "Filter Gambar", "Analisis Warna"])

# ======================
# Upload Image
# ======================
uploaded = st.file_uploader("📤 Unggah gambar Bulan/Matahari", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Gambar diunggah", use_container_width=True)
    st.markdown("---")

    # Klasifikasi
    if mode=="Klasifikasi":
        if classifier is None: st.error("Model classifier belum dimuat.")
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

    # Deteksi Objek
    elif mode=="Deteksi Objek":
        if yolo is None: st.error("YOLO belum dimuat.")
        else:
            results = yolo(img)
            st.image(results[0].plot(), caption="Hasil Deteksi", use_container_width=True)

    # Filter Gambar
    elif mode=="Filter Gambar":
        filter_opt = st.selectbox("Pilih filter:", ["Asli","Grayscale","Blur","Sharpen","Edge"])
        intensity = st.slider("Intensitas filter", 1, 10, 3)
        if filter_opt=="Grayscale": out = ImageOps.grayscale(img)
        elif filter_opt=="Blur": out = img.filter(ImageFilter.GaussianBlur(radius=intensity))
        elif filter_opt=="Sharpen": out = img.filter(ImageFilter.UnsharpMask(radius=intensity))
        elif filter_opt=="Edge": out = img.filter(ImageFilter.FIND_EDGES)
        else: out = img
        st.image(out, caption=f"Filter: {filter_opt}", use_container_width=True)

    # Analisis Warna
    elif mode=="Analisis Warna":
        small = img.resize((120,120))
        arr = np.array(small).reshape(-1,3)
        uniq, counts = np.unique((arr//32)*32, axis=0, return_counts=True)
        top = uniq[np.argsort(-counts)[:5]]
        st.write("🌈 Warna dominan:")
        cols = st.columns(5)
        for i,c in enumerate(top):
            hexc = '#%02x%02x%02x'%tuple(c)
            cols[i].markdown(f"<div style='background:{hexc};height:80px;border-radius:12px;'></div>", unsafe_allow_html=True)
            cols[i].write(hexc)

else:
    st.info("📁 Unggah gambar Bulan atau Matahari untuk mulai analisis.")

# Footer
st.markdown("<footer>🌙☀️ Celestial Vision — by Reva 💜 | Streamlit & TensorFlow</footer>", unsafe_allow_html=True)
