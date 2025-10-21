import streamlit as st
import pandas as pd
import plotly.express as px

# Judul
st.set_page_config(page_title="Dashboard Penjualan", layout="wide")
st.title("📈 Dashboard Penjualan Interaktif - Superstore")

# Load data
df = pd.read_csv("superstore_sales.csv")

# Filter interaktif
st.sidebar.header("Filter Data")
region = st.sidebar.multiselect("Pilih Wilayah:", df['Region'].unique(), default=df['Region'].unique())
category = st.sidebar.multiselect("Pilih Kategori:", df['Category'].unique(), default=df['Category'].unique())

# Filter dataframe
filtered_df = df[(df['Region'].isin(region)) & (df['Category'].isin(category))]

# KPI (indikator utama)
total_sales = filtered_df['Sales'].sum()
total_profit = filtered_df['Profit'].sum()
avg_discount = filtered_df['Discount'].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total Penjualan", f"${total_sales:,.0f}")
col2.metric("Total Keuntungan", f"${total_profit:,.0f}")
col3.metric("Rata-rata Diskon", f"{avg_discount*100:.2f}%")

# Grafik penjualan per kategori
fig1 = px.bar(filtered_df, x='Category', y='Sales', color='Category', title="Penjualan per Kategori")
st.plotly_chart(fig1, use_container_width=True)

# Grafik profit per wilayah
fig2 = px.pie(filtered_df, values='Profit', names='Region', title="Distribusi Profit per Wilayah", hole=0.4)
st.plotly_chart(fig2, use_container_width=True)

# Tren penjualan bulanan
df['Order Date'] = pd.to_datetime(df['Order Date'])
df_month = filtered_df.groupby(df['Order Date'].dt.to_period("M")).sum().reset_index()
df_month['Order Date'] = df_month['Order Date'].astype(str)
fig3 = px.line(df_month, x='Order Date', y='Sales', title="Tren Penjualan Bulanan")
st.plotly_chart(fig3, use_container_width=True)
