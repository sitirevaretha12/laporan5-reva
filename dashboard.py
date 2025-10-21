import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import glob
import os

# Optional YOLO import
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="🌙☀️ Celestial Vision", layout="wide")

# ==========================
# CSS: Langit malam + Glass
# ==========================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        font-family: 'Poppins', sans-serif;
        color: #ffffff;
    }
    .title { text-align:center; color:#ffd700; font-size:40px; font-weight:700; margin-top:10px; }
    .subtitle { text-align:center; color:#f0e68c; margin-bottom:20px; }
    .result-box { background: rgba(0,0,0,0.6); padding:16px; border-radius:14px; box-shadow: 0 6px 18px rgba(0,0,0,0.5); }
    footer { text-align:center; color:#f0e68c; margin-top:30px; padding:8px; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

# ==========================
# HELPER: find model files
# ==========================
MODEL_FOLDER = "model"
def find_first(pattern):
    files = glob.glob(os.path.join(MODEL_FOLDER, pattern))
    return files[0] if files else None

# ==========================
# LOAD CLASSIFIER & YOLO
# ==========================
@st.cache_resource
def load_classifier_model():
    h5_path = find_first("*.h5")
    if not h5_path:
        return None, "no_h5"
    try:
        model = tf.keras.models.load_model(h5_path)
        return model, h5_path
    except Exception as e:
        return None, f"error:{e}"

@st.cache_resource
def load_yolo_model():
    if not ULTRALYTICS_AVAILABLE:
        return None, "ultralytics_missing"
    pt_path = find_first("*.pt")
    if not pt_path:
        return None, "no_pt"
    try:
        yolo = YOLO(pt_path)
        return yolo, f"Dimuat dari {pt_path}"
    except Exception as e:
        return None, f"error:{e}"

classifier, cls_load_info = load_classifier_model()
yolo_model, yolo_load_info = load_yolo_model()

# ==========================
# CLASS NAMES
# ==========================
class_names = ["Bulan", "Matahari"]

# ==========================
# INFO DATABASE
# ==========================
celestial_info = {
    "Bulan": {
        "nama": "🌙 Bulan",
        "jenis": "Bulan Purnama, Bulan Sabit, Bulan Baru",
        "fakta": "Bulan mengorbit Bumi setiap 27.3 hari dan mempengaruhi pasang surut air laut."
    },
    "Matahari": {
        "nama": "☀️ Matahari",
        "jenis": "Matahari Utama",
        "fakta": "Matahari adalah bintang terdekat dengan Bumi dan sumber energi utama bagi kehidupan."
    }
}

# ==========================
# UI HEADER
# ==========================
st.markdown("<div class='title'>🌙☀️ Celestial Vision</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Klasifikasi & Deteksi Bulan / Matahari</div>", unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("🔧 Status Model & Pengaturan")
if classifier is None:
    if cls_load_info=="no_h5": st.sidebar.error("❌ Classifier .h5 tidak ditemukan di folder 'model/'")
    else: st.sidebar.error(f"❌ Gagal load classifier: {cls_load_info}")
else:
    st.sidebar.success(f"✅ Classifier dimuat: {cls_load_info}")

if yolo_model is None:
    if yolo_load_info=="ultralytics_missing": st.sidebar.info("⚠️ Ultralytics tidak terpasang — YOLO mati")
    elif yolo_load_info=="no_pt": st.sidebar.info("⚠️ Model YOLO (.pt) tidak ditemukan — Deteksi nonaktif")
    else: st.sidebar.warning(f"⚠️ YOLO load error: {yolo_load_info}")
else:
    st.sidebar.success(f"✅ YOLO dimuat: {yolo_load_info}")

# Feature selection
features = []
if yolo_model: features.append("Deteksi Objek (YOLO)")
if classifier: features.append("Klasifikasi & Info")
features += ["Filter Gambar", "Analisis Warna"]

mode = st.sidebar.selectbox("Pilih Mode:", features)

# ==========================
# UPLOAD IMAGE
# ==========================
uploaded_file = st.file_uploader("📤 Unggah gambar Bulan / Matahari", type=["jpg","jpeg","png"])

# ==========================
# HELPER FUNCTIONS
# ==========================
def preprocess_for_classifier(pil_img, size=(224,224)):
    img_resized = pil_img.resize(size)
    arr = image.img_to_array(img_resized)/255.0
    arr = np.expand_dims(arr,0)
    return arr

def predict_label(model, pil_img):
    arr = preprocess_for_classifier(pil_img)
    pred = model.predict(arr)
    idx = int(np.argmax(pred))
    score = float(np.max(pred))
    label = class_names[idx] if idx < len(class_names) else "unknown"
    return label, score

def get_dominant_colors_simple(pil_img, n_colors=5):
    small = pil_img.resize((160,160))
    arr = np.array(small).reshape(-1,3)
    arr = (arr//16)*16
    uniq, counts = np.unique(arr, axis=0, return_counts=True)
    idx_sorted = np.argsort(-counts)
    colors = uniq[idx_sorted][:n_colors]
    return [tuple(map(int,c)) for c in colors]

# ==========================
# PROCESS IMAGE
# ==========================
if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"❌ Gagal membuka gambar: {e}")
        st.stop()

    st.image(img, caption="📷 Gambar diunggah", use_container_width=True)
    st.markdown("---")

    # MODE YOLO
    if mode=="Deteksi Objek (YOLO)":
        if yolo_model is None:
            st.error("YOLO tidak tersedia")
        else:
            with st.spinner("🔍 Deteksi YOLO..."):
                try:
                    results = yolo_model(img)
                    result_img = results[0].plot()
                    st.image(result_img, caption="🧩 Hasil Deteksi YOLO", use_container_width=True)
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes)>0:
                        st.write("Deteksi (label, confidence):")
                        for b in boxes:
                            lbl = int(b.cls.cpu().numpy()) if hasattr(b,"cls") else None
                            conf = float(b.conf.cpu().numpy()) if hasattr(b,"conf") else None
                            st.write(f"- Label: {lbl}, Confidence: {conf:.2f}" if lbl is not None else f"- Confidence: {conf:.2f}")
                except Exception as e:
                    st.error(f"YOLO gagal: {e}")

    # MODE KLASIFIKASI
    elif mode=="Klasifikasi & Info":
        if classifier is None:
            st.error("Classifier tidak tersedia")
        else:
            with st.spinner("🔎 Memprediksi..."):
                try:
                    label, score = predict_label(classifier,img)
                except Exception as e:
                    st.error(f"Prediksi gagal: {e}")
                    st.stop()
            info = celestial_info.get(label)
            if info:
                st.success(f"🎯 Terdeteksi: {info['nama']} — Confidence: {score*100:.2f}%")
                st.markdown(f"""
                <div class='result-box'>
                    <h3>{info['nama']}</h3>
                    <b>Jenis:</b> {info['jenis']}<br>
                    <b>Fakta:</b> {info['fakta']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"Hasil prediksi: {label} — Confidence: {score*100:.2f}%")

    # FILTER GAMBAR
    elif mode=="Filter Gambar":
        option = st.selectbox("Pilih filter:", ["Asli","Grayscale","Blur","Sharpen","Edge"])
        if option=="Grayscale": out = ImageOps.grayscale(img)
        elif option=="Blur": out = img.filter(ImageFilter.BLUR)
        elif option=="Sharpen": out = img.filter(ImageFilter.SHARPEN)
        elif option=="Edge": out = img.filter(ImageFilter.FIND_EDGES)
        else: out = img
        st.image(out, caption=f"Hasil filter: {option}", use_container_width=True)

    # ANALISIS WARNA
    elif mode=="Analisis Warna":
        colors = get_dominant_colors_simple(img, n_colors=5)
        st.write("🌈 Warna dominan (hex):")
        cols = st.columns(len(colors))
        for i,c in enumerate(colors):
            hexc = '#%02x%02x%02x' % c
            cols[i].markdown(f"<div style='background:{hexc}; height
