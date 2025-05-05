import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.title("📈 Time Series Forecasting App (ARIMA Model)")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your CSV file (Date column as index)", type=["csv"])

# Load dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
    st.success("✅ Custom dataset uploaded successfully!")
else:
    st.info("ℹ️ No file uploaded. Using default dataset (Daily Minimum Temperatures).")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    df = pd.read_csv(url, parse_dates=['Date'], index_col='Date')

# Show raw data
st.subheader("📊 Raw Data Preview")
st.line_chart(df.iloc[:, 0])

# User input for number of days
n = st.number_input("⏳ Enter number of days to forecast", min_value=1, max_value=30, value=7)

# ARIMA forecasting
try:
    series = df.iloc[:, 0]  # First numeric column
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=n)

    # Plot forecast
    st.subheader("📈 Forecasted Values")
    st.line_chart(forecast)
    st.write(forecast)
except Exception as e:
    st.error(f"❌ Error: {e}")
