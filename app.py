import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

st.title("📈 Time Series Forecasting App (ARIMA Model)")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your CSV file (Date column as index)", type=["csv"])

# Load Data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
    st.success("✅ Custom dataset uploaded successfully!")
else:
    st.info("ℹ️ No file uploaded. Using default dataset (Daily Minimum Temperatures).")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    df = pd.read_csv(url, parse_dates=['Date'], index_col='Date')

# Preview Data
st.subheader("📊 Raw Data Preview")
st.line_chart(df.iloc[:, 0])

# Forecast horizon
n = st.number_input("⏳ Enter number of days to forecast", min_value=1, max_value=30, value=7)

# ARIMA Forecasting
try:
    series = df.iloc[:, 0]  # Use first numeric column
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=n)

    # Generate future dates and assign as index
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n)
    forecast.index = future_dates

    # Show forecast
    st.subheader("📈 Forecasted Values")
    st.line_chart(forecast)
    st.write(forecast)

except Exception as e:
    st.error(f"❌ Error: {e}")
