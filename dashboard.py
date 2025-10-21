import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import glob
import os

# Optional YOLO import (only if ultralytics available)
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="🐾 Animal Vision Pro (Stable)", layout="wide")

# ==========================
# SIMPLE CSS (pastel + glass)
# ==========================
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #fde2e4 0%, #fad2e1 30%, #e2ece9 70%, #bee1e6 100%);
            font-family: 'Poppins', sans-serif;
        }
        .title { text-align:center; color:#5a189a; font-size:40px; font-weight:700; margin-top:10px; }
        .subtitle { text-align:center; color:#6d597a; margin-bottom:20px; }
        .result-box { background: rgba(255,255,255,0.78); padding:16px; border-radius:14px; box-shadow: 0 6px 18px rgba(0,0,0,0.08); }
        footer { text-align:center; color:#5c4d7d; margin-top:30px; padding:8px; border-radius:12px; }
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
# LOAD MODELS (with caching)
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
        return yolo, pt_path
    except Exception as e:
        return None, f"error:{e}"

# Try load classifier (mandatory for classification features)
classifier, cls_load_info = load_classifier_model()

# Try load YOLO (optional; detection feature only available if loaded)
yolo_model, yolo_load_info = load_yolo_model()

# ==========================
# CLASS NAMES (must match your training order)
# ==========================
class_names = ["butterfly", "cat", "chicken", "dog", "fish", "horse", "spider"]

# ==========================
# ANIMAL INFO DATABASE
# ==========================
animal_info = {
    "cat": {
        "nama": "Kucing 🐱",
        "jenis": "Domestik, Persia, Maine Coon, Siam, Bengal",
        "makanan": "Daging, ikan, makanan kucing kering",
        "habitat": "Rumah, taman, lingkungan manusia",
        "fakta": "Kucing bisa tidur 12–16 jam per hari dan punya refleks melompat luar biasa."
    },
    "dog": {
        "nama": "Anjing 🐶",
        "jenis": "Labrador, Bulldog, German Shepherd, Pomeranian",
        "makanan": "Daging, tulang, makanan anjing komersial",
        "habitat": "Rumah dan lingkungan manusia",
        "fakta": "Hidung anjing bisa mencium sangat tajam."
    },
    "fish": {
        "nama": "Ikan 🐟",
        "jenis": "Mas, Guppy, Koi, Salmon, Lele",
        "makanan": "Plankton, serangga air, pelet ikan",
        "habitat": "Sungai, laut, danau, akuarium",
        "fakta": "Beberapa ikan seperti hiu hanya punya tulang rawan."
    },
    "chicken": {
        "nama": "Ayam 🐔",
        "jenis": "Kampung, Broiler, Petelur, Silkie",
        "makanan": "Biji-bijian, serangga kecil, dedak",
        "habitat": "Kandang dan area peternakan",
        "fakta": "Ayam bisa mengenali banyak wajah."
    },
    "horse": {
        "nama": "Kuda 🐴",
        "jenis": "Arab, Poni, Thoroughbred",
        "makanan": "Rumput, jerami, biji-bijian",
        "habitat": "Padang rumput, peternakan, istal",
        "fakta": "Kuda bisa tidur sambil berdiri."
    },
    "butterfly": {
        "nama": "Kupu-kupu 🦋",
        "jenis": "Monarch, Swallowtail, Morpho, Sulphur",
        "makanan": "Nektar bunga",
        "habitat": "Taman, padang rumput, hutan",
        "fakta": "Merasakan rasa lewat kaki."
    },
    "spider": {
        "nama": "Laba-laba 🕷️",
        "jenis": "Tarantula, Black Widow, Jumping Spider",
        "makanan": "Serangga kecil",
        "habitat": "Sudut rumah, taman, hutan",
        "fakta": "Jaring laba-laba sangat kuat relatif terhadap ukurannya."
    }
}

# ==========================
# UI HEADER
# ==========================
st.markdown("<div class='title'>🐾 Animal Vision Pro (Stable)</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Klasifikasi Hewan + Info Lengkap — Tahan error & mudah dipakai</div>", unsafe_allow_html=True)

# ==========================
# Sidebar: show model load status and options
# ==========================
st.sidebar.header("🔧 Status Model & Pengaturan")

if classifier is None:
    if cls_load_info == "no_h5":
        st.sidebar.error("❌ Classifier .h5 tidak ditemukan di folder 'model/'. Tambahkan file .h5 model kamu.")
    else:
        st.sidebar.error(f"❌ Gagal memuat classifier: {cls_load_info}")
else:
    st.sidebar.success(f"✅ Classifier dimuat dari:\n{cls_load_info}")

if yolo_model is None:
    if yolo_load_info == "ultralytics_missing":
        st.sidebar.info("⚠️ Ultralytics (YOLO) tidak terpasang — fitur deteksi objek dimatikan.")
    elif yolo_load_info == "no_pt":
        st.sidebar.info("⚠️ Model YOLO (.pt) tidak ditemukan di folder 'model/'. Deteksi objek nonaktif.")
    else:
        st.sidebar.warning(f"⚠️ Gagal muat YOLO: {yolo_load_info}")
else:
    st.sidebar.success(f"✅ YOLO dimuat dari:\n{yolo_load_info}")

# Feature selection
features = []
if yolo_model is not None:
    features.append("Deteksi Objek (YOLO)")
if classifier is not None:
    features.append("Klasifikasi & Info Hewan")
features += ["Filter Gambar (opsional)", "Analisis Warna (opsional)"]

if not features:
    st.warning("Tidak ada model siap. Silakan letakkan file model di folder 'model/'. Lihat sidebar untuk detail.")
    st.stop()

mode = st.sidebar.selectbox("Pilih Mode:", features)

uploaded_file = st.file_uploader("📤 Unggah gambar hewan (.jpg .jpeg .png)", type=["jpg", "jpeg", "png"])

# ==========================
# Helper: preprocess for classifier
# ==========================
def preprocess_for_classifier(pil_img, size=(224,224)):
    img_resized = pil_img.resize(size)
    arr = image.img_to_array(img_resized)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr

# Helper: get top prediction safely
def predict_label(model, pil_img):
    arr = preprocess_for_classifier(pil_img)
    pred = model.predict(arr)
    idx = int(np.argmax(pred))
    score = float(np.max(pred))
    label = class_names[idx] if idx < len(class_names) else "unknown"
    return label, score

# Simple dominant color extractor (kmeans-free: use sampling + unique)
def get_dominant_colors_simple(pil_img, n_colors=5):
    small = pil_img.resize((160, 160))
    arr = np.array(small).reshape(-1, 3)
    # sample unique colors by rounding
    arr_rounded = (arr // 16) * 16
    uniq, counts = np.unique(arr_rounded, axis=0, return_counts=True)
    idx_sorted = np.argsort(-counts)
    colors = uniq[idx_sorted][:n_colors]
    return [tuple(map(int, c)) for c in colors]

# ==========================
# MAIN: process uploaded image
# ==========================
if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"❌ Gagal membuka file gambar: {e}")
        st.stop()

    st.image(img, caption="📷 Gambar yang diunggah", use_container_width=True)
    st.markdown("---")

    if mode == "Deteksi Objek (YOLO)":
        if yolo_model is None:
            st.error("Fitur deteksi tidak aktif karena model YOLO tidak tersedia.")
        else:
            with st.spinner("🔍 Menjalankan deteksi YOLO..."):
                try:
                    results = yolo_model(img)  # ultralytics accepts PIL
                    result_img = results[0].plot()
                    st.image(result_img, caption="🧩 Hasil Deteksi (YOLO)", use_container_width=True)
                    # show textual boxes if available
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        st.write("Deteksi (label, confidence):")
                        for b in boxes:
                            lbl = b.cls.cpu().numpy() if hasattr(b, "cls") else None
                            conf = float(b.conf.cpu().numpy()) if hasattr(b, "conf") else None
                            # ultralytics box label mapping depends on model; show raw if needed
                            st.write(f"- {lbl} — {conf:.2f}" if lbl is not None else f"- conf {conf:.2f}")
                except Exception as e:
                    st.error(f"Gagal menjalankan deteksi YOLO: {e}")

    elif mode == "Klasifikasi & Info Hewan":
        if classifier is None:
            st.error("Fitur klasifikasi tidak aktif karena model classifier (.h5) tidak tersedia.")
        else:
            with st.spinner("🔎 Memprediksi kelas hewan..."):
                try:
                    label, score = predict_label(classifier, img)
                except Exception as e:
                    st.error(f"Error saat prediksi: {e}")
                    st.stop()

            if label not in animal_info:
                st.warning(f"Hasil prediksi: {label} (tidak ada data info). Confidence: {score:.2%}")
            else:
                info = animal_info[label]
                st.success(f"🎯 Hewan terdeteksi: {info['nama']} — Confidence: {score*100:.2f}%")
                st.markdown(f"""
                <div class='result-box'>
                    <h3>{info['nama']}</h3>
                    <b>🌿 Jenis-jenis:</b> {info['jenis']}<br>
                    <b>🍽️ Makanan:</b> {info['makanan']}<br>
                    <b>🏞️ Habitat:</b> {info['habitat']}<br>
                    <b>💡 Fakta menarik:</b> {info['fakta']}
                </div>
                """, unsafe_allow_html=True)

    elif mode == "Filter Gambar (opsional)":
        option = st.selectbox("Pilih filter:", ["Asli", "Grayscale", "Blur", "Sharpen", "Edge"])
        if option == "Grayscale":
            out = ImageOps.grayscale(img)
        elif option == "Blur":
            out = img.filter(ImageFilter.BLUR)
        elif option == "Sharpen":
            out = img.filter(ImageFilter.SHARPEN)
        elif option == "Edge":
            out = img.filter(ImageFilter.FIND_EDGES)
        else:
            out = img
        st.image(out, caption=f"Hasil filter: {option}", use_container_width=True)

    elif mode == "Analisis Warna (opsional)":
        colors = get_dominant_colors_simple(img, n_colors=5)
        st.write("🌈 Warna dominan (hex):")
        cols = st.columns(len(colors))
        for i, c in enumerate(colors):
            hexc = '#%02x%02x%02x' % c
            cols[i].markdown(f"<div style='background:{hexc}; height:80px; border-radius:8px;'></div>", unsafe_allow_html=True)
            cols[i].write(hexc)

else:
    st.info("📁 Silakan unggah gambar untuk dianalisis. Pastikan kamu sudah menaruh model classifier (.h5) di folder 'model/' jika ingin fitur klasifikasi bekerja.")

# ==========================
# FOOTER
# ==========================
st.markdown("""
<footer>
    🌸 Animal Vision Pro — Stable Build • By Repa Cantikk<br>
    Letakkan model di folder <code>model/</code> (file .h5 untuk classifier, .pt untuk YOLO jika diperlukan).
</footer>
""", unsafe_allow_html=True)
