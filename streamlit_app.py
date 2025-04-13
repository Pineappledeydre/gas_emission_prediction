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

@st.cache_resource
def load_model():
    return joblib.load("cat_model.pkl")

model = load_model()

st.markdown("## 🌍 Платформа прогнозирования выбросов парниковых газов")
st.markdown("Интеллектуальная система, имитирующая поток данных с промышленных объектов и предсказывающая выбросы CO₂/CH₄ в реальном времени.")
st.divider()

col1, col2 = st.columns([1, 2])

# Logging data for plotting
data_log = []

# Function to simulate a single record
def generate_realtime_data():
    return {
        'site_id': random.choice([0, 1, 2, 3, 4]),
        'temp': np.random.normal(10, 5),
        'pressure': np.random.normal(1013, 5),
        'humidity': np.random.uniform(30, 90),
        'load': np.random.uniform(50, 100),
        'maintenance_flag': np.random.choice([0, 1], p=[0.95, 0.05]),
        'gas_type': None,  # Will set inside loop
        'operational_hours': np.random.randint(16, 24)
    }

# Real-time loop
placeholder1 = col1.empty()
placeholder2 = col2.empty()

for _ in range(20):
    timestamp_now = datetime.now().strftime('%H:%M:%S')
    
    new_data_block = []  # Temporarily store this timestamp's data for visualization

    for gas_type in [0, 1]:
        data = generate_realtime_data()
        data['gas_type'] = gas_type
        df_input = pd.DataFrame([data])
        prediction = model.predict(df_input)[0]
        data['prediction'] = prediction
        data['timestamp'] = timestamp_now
        data_log.append(data)
        new_data_block.append((gas_type, df_input, prediction))

    # Live update input + prediction in col1
    with placeholder1.container():
        for gas_type, df_input, prediction in new_data_block:
            gas_name = "CH₄" if gas_type == 0 else "CO₂"
            st.write(f"**⏱️ {timestamp_now} — {gas_name}**")
            st.dataframe(df_input, use_container_width=True)
            st.success(f"💨 Прогноз выброса: **{prediction:.2f}** единиц")
        st.divider()

    # Live update graph in col2
    with placeholder2.container():
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

        st.markdown("### Последние наблюдения")
        st.dataframe(df_log.tail(6)[['timestamp', 'temp', 'pressure', 'load', 'gas_type', 'prediction']],
                     use_container_width=True)

    time.sleep(1.5)
