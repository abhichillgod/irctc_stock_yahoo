# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import datetime
import matplotlib.pyplot as plt

# =====================
# Load Model & Scaler
# =====================
model = joblib.load("irctc_model.pkl")      # Exported from your notebook
scaler = joblib.load("irctc_scaler.pkl")    # Exported from your notebook

# =====================
# App Title & Info
# =====================
st.title("🚆 IRCTC Stock Price Prediction App")
st.markdown("Predict the **next trading day's closing price** and trend using historical market data.")

# =====================
# Sidebar Inputs
# =====================
st.sidebar.header("Data Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="IRCTC.NS")
start_date = st.sidebar.date_input("Start Date", value=datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.date.today())

# Graph selection
graph_type = st.sidebar.selectbox(
    "Select Graph Type",
    ["Closing Price Trend", "Candlestick Chart", "Volume Chart", "Moving Average (50/200)"]
)

# =====================
# Download Stock Data
# =====================
df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

if df.empty:
    st.error("No data found for this ticker/date range.")
    st.stop()

st.subheader(f"📊 Historical Data for {ticker}")
st.write(df.tail())

# =====================
# Plot Graph Based on Selection
# =====================
st.subheader(f"Graph: {graph_type}")

fig, ax = plt.subplots(figsize=(10, 5))

if graph_type == "Closing Price Trend":
    ax.plot(df["Close"], label="Closing Price", color="blue")
    ax.set_ylabel("Price (INR)")

elif graph_type == "Volume Chart":
    ax.bar(df.index, df["Volume"], color="orange")
    ax.set_ylabel("Volume")

elif graph_type == "Moving Average (50/200)":
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["MA200"] = df["Close"].rolling(window=200).mean()
    ax.plot(df["Close"], label="Close", color="blue")
    ax.plot(df["MA50"], label="50-Day MA", color="green")
    ax.plot(df["MA200"], label="200-Day MA", color="red")
    ax.set_ylabel("Price (INR)")

elif graph_type == "Candlestick Chart":
    try:
        import mplfinance as mpf
        mpf.plot(df, type='candle', style='charles', volume=True)
        st.pyplot(plt)
    except ImportError:
        st.error("Please install mplfinance for candlestick chart: pip install mplfinance")

ax.set_xlabel("Date")
ax.legend()
if graph_type != "Candlestick Chart":
    st.pyplot(fig)

# =====================
# Prepare Latest Data for Prediction
# =====================
latest_data = df[['Open', 'High', 'Low', 'Volume']].iloc[-1].values.reshape(1, -1)
latest_scaled = scaler.transform(latest_data)
predicted_price = model.predict(latest_scaled)[0].item()

# Ensure last_close is a scalar float
last_close = float(df["Close"].iloc[-1])

# Determine trend (up/down)
trend = "📈 UP" if predicted_price > last_close else "📉 DOWN"

st.subheader("Prediction")
st.write(f"*Predicted closing price for the next trading day:* ₹{predicted_price:,.2f}")
st.write(f"**Expected Trend:** {trend}")

# =====================
# Manual Prediction Input
# =====================
st.sidebar.header("Manual Prediction Input")
open_price = st.sidebar.number_input("Open Price", value=float(df['Open'].iloc[-1]))
high_price = st.sidebar.number_input("High Price", value=float(df['High'].iloc[-1]))
low_price = st.sidebar.number_input("Low Price", value=float(df['Low'].iloc[-1]))
volume = st.sidebar.number_input("Volume", value=float(df['Volume'].iloc[-1]))

if st.sidebar.button("Predict from Manual Input"):
    manual_data = np.array([[open_price, high_price, low_price, volume]])
    manual_scaled = scaler.transform(manual_data)
    manual_pred = model.predict(manual_scaled)[0].item()
    manual_trend = "📈 UP" if manual_pred > last_close else "📉 DOWN"
    st.write(f"Manual Input Prediction: ₹{manual_pred:,.2f}")
    st.write(f"**Expected Trend:** {manual_trend}")

st.success("✅ App is ready. Adjust settings in the sidebar.")
