
import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

st.title("Time Series Forecasting App (ARIMA)")

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
df = pd.read_csv(url, parse_dates=['Date'], index_col='Date')

st.subheader("Daily Minimum Temperatures")
st.line_chart(df['Temp'])

n = st.number_input("Enter number of days to forecast", min_value=1, max_value=30, value=7)

model = ARIMA(df['Temp'], order=(5, 1, 0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=n)

st.subheader("Forecasted Temperatures")
st.write(forecast)
st.line_chart(forecast)
