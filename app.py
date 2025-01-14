import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import requests

# Page config
st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")
st.title("COVID-19 Küresel Veri Analizi ve Tahmin Dashboard'u")

# Data loading
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df['date'])
    return df

# Load data
df = load_data()

# Sidebar filters
st.sidebar.header("Filtreleme Seçenekleri")
selected_country = st.sidebar.selectbox(
    "Ülke Seçin",
    options=sorted(df['location'].unique())
)

# Filter data for selected country
country_data = df[df['location'] == selected_country].copy()

# Main metrics
st.header(f"{selected_country} COVID-19 İstatistikleri")
col1, col2, col3, col4 = st.columns(4)

with col1:
    last_total_cases = country_data['total_cases'].iloc[-1]
    st.metric("Toplam Vaka", f"{int(last_total_cases):,}")

with col2:
    last_total_deaths = country_data['total_deaths'].iloc[-1]
    st.metric("Toplam Ölüm", f"{int(last_total_deaths):,}")

with col3:
    last_vaccinations = country_data['people_vaccinated'].iloc[-1]
    st.metric("Aşılanan Kişi", f"{int(last_vaccinations):,}")

with col4:
    mortality_rate = (last_total_deaths / last_total_cases) * 100
    st.metric("Ölüm Oranı", f"{mortality_rate:.2f}%")

# Time series plot
st.subheader("Zaman Serisi Analizi")
metric = st.selectbox("Metrik Seçin", ['new_cases', 'new_deaths', 'total_cases', 'total_deaths'])

fig = px.line(country_data, x='date', y=metric, title=f"{selected_country} - {metric}")
st.plotly_chart(fig, use_container_width=True)

# Simple prediction
st.subheader("Gelecek 30 Gün Tahmin")
X = np.array(range(len(country_data))).reshape(-1, 1)
y = country_data['total_cases'].values

model = LinearRegression()
model.fit(X, y)

# Predict next 30 days
future_dates = pd.date_range(
    start=country_data['date'].iloc[-1], 
    periods=31, 
    freq='D'
)[1:]

X_future = np.array(range(len(country_data), len(country_data) + 30)).reshape(-1, 1)
predictions = model.predict(X_future)

future_df = pd.DataFrame({
    'date': future_dates,
    'predicted_cases': predictions
})

# Plot predictions
fig_pred = px.line(title=f"{selected_country} - Gelecek 30 Gün Vaka Tahmini")
fig_pred.add_scatter(x=country_data['date'], y=country_data['total_cases'], name='Gerçek Veriler')
fig_pred.add_scatter(x=future_df['date'], y=future_df['predicted_cases'], name='Tahmin')
st.plotly_chart(fig_pred, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Not:** Bu dashboard örnek amaçlı hazırlanmıştır. Tahminler basit doğrusal regresyon kullanılarak yapılmıştır.")
