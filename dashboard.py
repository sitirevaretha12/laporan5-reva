import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
import numpy as np
from PIL import Image
import glob, os

# ======================
# KONFIGURASI PATH
# ======================
MODEL_FOLDER = "model"

def find_first(pattern):
    """Cari file pertama yang cocok dengan pola (di folder model)."""
    files = glob.glob(os.path.join(MODEL_FOLDER, pattern))
    return files[0] if files else None

# ======================
# LOAD CLASSIFIER (.h5)
# ======================
@st.cache_resource
def load_classifier():
    h5_path = find_first("*.h5")
    if not h5_path:
        return None, "❌ File .h5 tidak ditemukan di folder 'model'"
    try:
        model = tf.keras.models.load_model(h5_path)
        return model, f"✅ Classifier dimuat dari: {os.path.basename(h5_path)}"
    except Exception as e:
        return None, f"⚠️ Gagal memuat model .h5: {e}"

# ======================
# LOAD YOLO (.pt)
# ======================
@st.cache_resource
def load_yolo():
    pt_path = find_first("*.pt")
    if not pt_path:
        return None, "❌ File .pt tidak ditemukan di folder 'model'"
    try:
        model = YOLO(pt_path)
        return model, f"✅ YOLO dimuat dari: {os.path.basename(pt_path)}"
    except Exception as e:
        return None, f"⚠️ Gagal memuat YOLO: {e}"

# ======================
# MUAT MODEL
# ======================
classifier, cls_info = load_classifier()
yolo, yolo_info = load_yolo()

# ======================
# STATUS DI SIDEBAR
# ======================
st.sidebar.header("⚙️ Status Model")

if classifier:
    st.sidebar.success(cls_info)
else:
    st.sidebar.error(cls_info)

if yolo:
    st.sidebar.success(yolo_info)
else:
    st.sidebar.warning(yolo_info)
