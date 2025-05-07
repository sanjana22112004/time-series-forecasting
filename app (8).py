import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# User input for number of days to forecast
n = st.number_input("⏳ Enter number of days to forecast", min_value=1, max_value=30, value=7)

# User inputs for ARIMA parameters
st.subheader("⚙️ ARIMA Model Parameters")
p = st.number_input("Enter AR term (p)", min_value=0, max_value=10, value=5)
d = st.number_input("Enter I term (d)", min_value=0, max_value=2, value=1)
q = st.number_input("Enter MA term (q)", min_value=0, max_value=10, value=0)

# Split data into training and testing sets (80% training, 20% testing)
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Train ARIMA model on training data
try:
    model = ARIMA(train.iloc[:, 0], order=(p, d, q))
    model_fit = model.fit()

    # Forecast on test data
    forecast = model_fit.forecast(steps=len(test))

    # Plot forecast
    st.subheader("📈 Forecasted Values")
    st.line_chart(forecast)
    st.write(forecast)

    # Calculate MAE and RMSE
    mae = mean_absolute_error(test.iloc[:, 0], forecast)
    rmse = np.sqrt(mean_squared_error(test.iloc[:, 0], forecast))

    # Display accuracy metrics
    st.subheader("📊 Model Accuracy")
    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

except Exception as e:
    st.error(f"❌ Error: {e}")

