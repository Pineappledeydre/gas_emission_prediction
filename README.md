## 🌍 GHG Emission Forecaster — Streamlit Dashboard

This project simulates **real-time prediction of greenhouse gas emissions (CO₂ and CH₄)** using a machine learning model trained on environmental and industrial parameters.

Built with **Streamlit**, **CatBoost**, and **Plotly**, the app imitates the live data collection process and provides dynamic visualization of forecasted emissions over time.

---

### Features

- **Real-time data simulation** for two gas types: CO₂ and CH₄
- **CatBoost-based regression model** for predicting emissions
- **Live updating charts** of emissions over time
- Interactive dashboard with real-time input parameters and forecast output
- Displays the latest prediction inputs and results in a readable, scientific format

---

### Technologies

- [Streamlit](https://streamlit.io/) — UI and data dashboarding  
- [CatBoost](https://catboost.ai/) — Regression model for emissions  
- [Plotly](https://plotly.com/) — Interactive visualizations  
- Python (Pandas, NumPy, Joblib)

---

### Installation

1. Clone the repository or download the app file:
```bash
git clone https://github.com/your-username/ghg-forecast-dashboard.git
cd ghg-forecast-dashboard
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

3. Make sure your CatBoost model file is present as:
```
cat_model.pkl
```

If you don’t have one, train it separately and save it using `joblib.dump(model, "cat_model.pkl")`.

4. Run the app:
```bash
streamlit run enhanced_streamlit_emission_app.py
```

---

### File Structure

```
│
├── streamlit_app.py                     # Main Streamlit app
├── cat_model.pkl                        # Trained CatBoost model
|── gas_emissions.ipynb                  # Colab Notebook - Model Trainig
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

---
### Notes

- This app simulates **new data every 1.5 seconds** and predicts emissions for both gases.
- You can integrate real external APIs (e.g., OpenWeatherMap) to replace the synthetic input.

---

###  License

MIT License. Use freely with attribution.
