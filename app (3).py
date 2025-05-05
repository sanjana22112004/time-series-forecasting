
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Time Series Forecasting App", layout="centered")

st.title("📈 Time Series Forecasting App")
st.markdown("Upload your time series dataset and forecast future values using the ARIMA model.")

uploaded_file = st.file_uploader("Upload CSV file with a Date column", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        st.success("File uploaded successfully!")

        date_col = st.selectbox("Select the Date column", df.columns)
        value_col = st.selectbox("Select the target column (e.g., temperature, sales)", df.columns)

        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df = df[[value_col]]

        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        st.subheader("📈 Time Series Plot")
        st.line_chart(df)

        forecast_days = st.slider("How many future steps to forecast?", 1, 30, 7)

        st.sidebar.title("ARIMA Parameters")
        p = st.sidebar.number_input("AR term (p)", 0, 10, 5)
        d = st.sidebar.number_input("Differencing term (d)", 0, 2, 1)
        q = st.sidebar.number_input("MA term (q)", 0, 10, 0)

        train = df[:-forecast_days]
        test = df[-forecast_days:]

        st.subheader("🔧 Training ARIMA Model...")
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=forecast_days)
        forecast.index = test.index

        st.subheader("📈 Forecast vs Actual")
        fig, ax = plt.subplots()
        ax.plot(train.index, train[value_col], label="Train")
        ax.plot(test.index, test[value_col], label="Actual")
        ax.plot(forecast.index, forecast, label="Forecast")
        ax.legend()
        st.pyplot(fig)

        mse = mean_squared_error(test[value_col], forecast)
        mae = mean_absolute_error(test[value_col], forecast)

        st.subheader("📊 Model Evaluation")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")

else:
    st.info("Please upload a CSV file to get started.")
