import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import datetime
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("irctc_model.pkl")
scaler = joblib.load("irctc_scaler.pkl")

st.title("🚆 IRCTC Stock Price Movement Prediction App")
st.markdown("Predict whether the **next trading day's closing price** will go **Up** 📈 or **Down** 📉.")

# Sidebar inputs
st.sidebar.header("Data Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="IRCTC.NS")
start_date = st.sidebar.date_input("Start Date", value=datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.date.today())

# Download stock data
st.subheader(f"Historical Data for {ticker}")
df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

if df.empty:
    st.error("No data found for this ticker/date range.")
    st.stop()

# Feature engineering (must match training features)
df["Return"] = df["Close"].pct_change()
df["MA5"] = df["Close"].rolling(window=5).mean()
df["MA10"] = df["Close"].rolling(window=10).mean()
df["MA20"] = df["Close"].rolling(window=20).mean()
df["Vol_MA5"] = df["Volume"].rolling(window=5).mean()
df.dropna(inplace=True)

st.write(df.tail())

# Plot closing price
st.subheader("Closing Price Trend")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["Close"], label="Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price (INR)")
ax.legend()
st.pyplot(fig)

# Prepare latest data for prediction
latest_data = df[["Open", "High", "Low", "Volume", "Return", "MA5", "MA10", "MA20", "Vol_MA5"]].iloc[-1].values.reshape(1, -1)
latest_scaled = scaler.transform(latest_data)
predicted_class = model.predict(latest_scaled)[0]

movement = "📈 Up" if predicted_class == 1 else "📉 Down"
st.subheader("Prediction")
st.write(f"*Predicted movement for the next trading day:* **{movement}**")

# Manual input prediction
st.sidebar.header("Manual Prediction Input")
open_price = st.sidebar.number_input("Open Price", value=float(df['Open'].iloc[-1]))
high_price = st.sidebar.number_input("High Price", value=float(df['High'].iloc[-1]))
low_price = st.sidebar.number_input("Low Price", value=float(df['Low'].iloc[-1]))
volume = st.sidebar.number_input("Volume", value=float(df['Volume'].iloc[-1]))
ret = st.sidebar.number_input("Return (%)", value=float(df['Return'].iloc[-1]))
ma5 = st.sidebar.number_input("MA5", value=float(df['MA5'].iloc[-1]))
ma10 = st.sidebar.number_input("MA10", value=float(df['MA10'].iloc[-1]))
ma20 = st.sidebar.number_input("MA20", value=float(df['MA20'].iloc[-1]))
vol_ma5 = st.sidebar.number_input("Vol_MA5", value=float(df['Vol_MA5'].iloc[-1]))

if st.sidebar.button("Predict from Manual Input"):
    manual_data = np.array([[open_price, high_price, low_price, volume, ret, ma5, ma10, ma20, vol_ma5]])
    manual_scaled = scaler.transform(manual_data)
    manual_pred_class = model.predict(manual_scaled)[0]
    manual_movement = "📈 Up" if manual_pred_class == 1 else "📉 Down"
    st.write(f"Manual Input Prediction: **{manual_movement}**")

st.success("✅ App is ready. Use the sidebar to adjust settings.")
