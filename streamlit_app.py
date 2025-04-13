import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import random
from catboost import CatBoostRegressor
import joblib
import plotly.express as px

st.set_page_config(page_title="GHG Emission Forecaster", layout="wide")

# Загрузка модели
@st.cache_resource
def load_model():
    return joblib.load("cat_model.pkl")

model = load_model()

st.markdown("## 🌍 Платформа прогнозирования выбросов парниковых газов")
st.markdown("Интеллектуальная система, имитирующая поток данных с промышленных объектов и предсказывающая выбросы CO₂/CH₄ в реальном времени.")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📡 Входные данные (имитация в реальном времени)")

    def generate_realtime_data():
        return {
            'site_id': random.choice([0, 1, 2, 3, 4]),
            'temp': np.random.normal(10, 5),
            'pressure': np.random.normal(1013, 5),
            'humidity': np.random.uniform(30, 90),
            'load': np.random.uniform(50, 100),
            'maintenance_flag': np.random.choice([0, 1], p=[0.95, 0.05]),
            'gas_type': random.choice([0, 1]),  # 0 = CH4, 1 = CO2
            'operational_hours': np.random.randint(16, 24)
        }

    data_log = []
    
    placeholder = st.empty()
    with placeholder.container():
        for _ in range(20):
            timestamp_now = datetime.now().strftime('%H:%M:%S')
            for gas_type in [0, 1]:
                data = generate_realtime_data()
                data['gas_type'] = gas_type
                df_input = pd.DataFrame([data])
                prediction = model.predict(df_input)[0]
                data['prediction'] = prediction
                data['timestamp'] = timestamp_now
                data_log.append(data)
    
                st.write(f"**⏱️ {timestamp_now} — {'CH₄' if gas_type == 0 else 'CO₂'}**")
                st.dataframe(df_input, use_container_width=True)
                st.success(f"💨 Прогноз выброса: **{prediction:.2f}** единиц")
                st.divider()
    
            time.sleep(1.5)

with col2:
    st.subheader("Динамика прогнозов")

    if data_log:
        df_log = pd.DataFrame(data_log)

        df_avg = (
            df_log.groupby(['timestamp', 'gas_type'])
            .agg({'prediction': 'mean'})
            .reset_index()
        )
        df_avg['gas_type'] = df_avg['gas_type'].map({0: "CH₄", 1: "CO₂"})
        
        fig = px.line(df_avg, x='timestamp', y='prediction', color='gas_type',
                      title="📈 Средние прогнозы выбросов по времени (CH₄ vs CO₂)",
                      labels={'prediction': 'Прогноз выбросов', 'timestamp': 'Время'})
        st.plotly_chart(fig, use_container_width=True)

        
        # Таблица последних 5 точек
        st.markdown("### Последние наблюдения")
        st.dataframe(df_log.tail(5)[['timestamp', 'temp', 'pressure', 'load', 'gas_type', 'prediction']],
                     use_container_width=True)
