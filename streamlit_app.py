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
            data = generate_realtime_data()
            df_input = pd.DataFrame([data])
            prediction = model.predict(df_input)[0]
            data['prediction'] = prediction
            data['timestamp'] = datetime.now().strftime('%H:%M:%S')
            data_log.append(data)
            
            st.write(f"**⏱️ {data['timestamp']}**")
            st.dataframe(df_input, use_container_width=True)
            st.success(f"💨 Прогноз выброса: **{prediction:.2f}** единиц")
            st.divider()
            time.sleep(1.5)

with col2:
    st.subheader("Динамика прогнозов")

    if data_log:
        df_log = pd.DataFrame(data_log)
        
        # График прогноза по времени
        fig = px.line(df_log, x="timestamp", y="prediction",
                      color=df_log["gas_type"].map({0: "CH₄", 1: "CO₂"}),
                      labels={"timestamp": "Время", "prediction": "Прогноз выбросов"},
                      title="Прогноз выбросов во времени")
        st.plotly_chart(fig, use_container_width=True)
        
        # Таблица последних 5 точек
        st.markdown("### Последние наблюдения")
        st.dataframe(df_log.tail(5)[['timestamp', 'temp', 'pressure', 'load', 'gas_type', 'prediction']],
                     use_container_width=True)
