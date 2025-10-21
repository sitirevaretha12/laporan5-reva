# =========================================================
# DASHBOARD PENJUALAN — Tema Langit Malam ✨
# =========================================================
import streamlit as st
import pandas as pd

# coba import Plotly (interaktif), fallback ke Matplotlib
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    import matplotlib.pyplot as plt
    PLOTLY_AVAILABLE = False

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="🌙 NightSky Dashboard Penjualan",
    page_icon="🌌",
    layout="wide"
)

# -------------------------------
# CSS (tema langit malam)
# -------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #2c2c54 50%, #202040 100%);
        color: #e0e0e0;
        font-family: 'Poppins', sans-serif;
    }
    .main-title {
        text-align: center;
        color: #cddafd;
        font-size: 38px;
        font-weight: 700;
        padding: 10px 0;
    }
    .sub {
        text-align: center;
        color: #9abaff;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .metric-box {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.markdown("<div class='main-title'>🌌 NightSky Dashboard Penjualan</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Visualisasi Penjualan, Profit, dan Tren Bulanan dengan Gaya Langit Malam</div>", unsafe_allow_html=True)

# -------------------------------
# LOAD DATASET
# -------------------------------
try:
    df = pd.read_csv("superstore_sales.csv")
except FileNotFoundError:
    st.error("⚠️ File `superstore_sales.csv` tidak ditemukan. Letakkan file CSV di folder yang sama dengan `dashboard.py`.")
    st.stop()

# cek kolom penting
if not {'Sales','Profit'}.issubset(df.columns):
    st.warning("Kolom 'Sales' atau 'Profit' tidak ditemukan. Pastikan dataset kamu sesuai format Superstore.")
    st.stop()

# -------------------------------
# SIDEBAR FILTER
# -------------------------------
st.sidebar.header("🔎 Filter Data")

if 'Region' in df.columns:
    region_list = df['Region'].dropna().unique().tolist()
    selected_regions = st.sidebar.multiselect("Pilih Region", region_list, default=region_list)
    df = df[df['Region'].isin(selected_regions)]

if 'Category' in df.columns:
    cat_list = df['Category'].dropna().unique().tolist()
    selected_cats = st.sidebar.multiselect("Pilih Kategori", cat_list, default=cat_list)
    df = df[df['Category'].isin(selected_cats)]

# -------------------------------
# KPI Cards
# -------------------------------
total_sales = df['Sales'].sum()
total_profit = df['Profit'].sum()
avg_discount = df['Discount'].mean() if 'Discount' in df.columns else 0

col1, col2, col3 = st.columns(3)
col1.markdown(f"<div class='metric-box'><h4>Total Penjualan</h4><h2>${total_sales:,.0f}</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-box'><h4>Total Profit</h4><h2>${total_profit:,.0f}</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-box'><h4>Rata-rata Diskon</h4><h2>{avg_discount*100:.1f}%</h2></div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# CHART 1: Sales per Category
# -------------------------------
st.subheader("📊 Penjualan per Kategori")
if 'Category' in df.columns:
    data_cat = df.groupby('Category', as_index=False)['Sales'].sum().sort_values('Sales', ascending=False)
    if PLOTLY_AVAILABLE:
        fig1 = px.bar(
            data_cat, x='Category', y='Sales',
            color='Category',
            title="Total Penjualan per Kategori",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        fig, ax = plt.subplots()
        ax.bar(data_cat['Category'], data_cat['Sales'], color='skyblue')
        ax.set_title("Total Penjualan per Kategori")
        st.pyplot(fig)

# -------------------------------
# CHART 2: Profit vs Sales Scatter
# -------------------------------
st.subheader("💸 Hubungan Profit vs Sales")
if PLOTLY_AVAILABLE:
    fig2 = px.scatter(
        df, x='Sales', y='Profit',
        size='Quantity' if 'Quantity' in df.columns else None,
        color='Region' if 'Region' in df.columns else None,
        title="Profit vs Sales",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    fig, ax = plt.subplots()
    ax.scatter(df['Sales'], df['Profit'], color='lightblue')
    ax.set_xlabel("Sales")
    ax.set_ylabel("Profit")
    ax.set_title("Profit vs Sales")
    st.pyplot(fig)

# -------------------------------
# CHART 3: Tren Penjualan Bulanan
# -------------------------------
st.subheader("🕐 Tren Penjualan Bulanan")
date_col = None
for col in df.columns:
    if 'date' in col.lower():
        date_col = col
        break

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df_month = df.groupby(pd.Grouper(key=date_col, freq='M'))['Sales'].sum().reset_index()
    if PLOTLY_AVAILABLE:
        fig3 = px.line(
            df_month, x=date_col, y='Sales',
            markers=True,
            title="Tren Penjualan Bulanan",
            line_shape="spline",
            color_discrete_sequence=["#9abaff"]
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        fig, ax = plt.subplots()
        ax.plot(df_month[date_col], df_month['Sales'], marker='o', color='#9abaff')
        ax.set_title("Tren Penjualan Bulanan")
        ax.set_xlabel("Bulan")
        ax.set_ylabel("Penjualan")
        plt.xticks(rotation=45)
        st.pyplot(fig)
else:
    st.info("📅 Kolom tanggal tidak ditemukan. Tambahkan kolom seperti 'Order Date' agar tren bisa ditampilkan.")

st.markdown("---")
st.caption("✨ NightSky Dashboard • Dibuat dengan Streamlit, Plotly, dan cinta 🌙")
