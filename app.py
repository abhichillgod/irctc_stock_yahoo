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
st.markdown("Predict whether the **next trading day's closing price** will go **Up** 📈 or **Down** 📉 and visualize trends.")

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

st.write(df.tail())

# Prepare latest data for prediction (4 features only)
latest_data = df[["Open", "High", "Low", "Volume"]].iloc[-1].values.reshape(1, -1)
latest_scaled = scaler.transform(latest_data)
predicted_class = model.predict(latest_scaled)[0]
movement = "📈 Up" if predicted_class == 1 else "📉 Down"

# Display prediction
st.subheader("Prediction")
st.write(f"*Predicted movement for the next trading day:* **{movement}**")

# Graph 1: Closing Price Trend with prediction marker
st.subheader("Closing Price Trend with Prediction Marker")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index.to_list(), df["Close"].astype(float).to_list(), label="Closing Price", color="blue")
ax.scatter([df.index[-1]], [float(df["Close"].iloc[-1])],
           color="green" if predicted_class == 1 else "red", s=100, label="Prediction")
ax.set_xlabel("Date")
ax.set_ylabel("Price (INR)")
ax.legend()
st.pyplot(fig)

# Graph 2: Volume Trend
st.subheader("Trading Volume Trend")
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(df.index.to_list(), df["Volume"].astype(float).to_list(), color="orange")
ax.set_xlabel("Date")
ax.set_ylabel("Volume")
st.pyplot(fig)

# Graph 3: Moving Averages
st.subheader("Moving Averages (MA5, MA10, MA20)")
df["MA5"] = df["Close"].rolling(window=5).mean()
df["MA10"] = df["Close"].rolling(window=10).mean()
df["MA20"] = df["Close"].rolling(window=20).mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index.to_list(), df["Close"].astype(float).to_list(), label="Close Price", color="blue")
ax.plot(df.index.to_list(), df["MA5"].astype(float).to_list(), label="MA5", color="green")
ax.plot(df.index.to_list(), df["MA10"].astype(float).to_list(), label="MA10", color="orange")
ax.plot(df.index.to_list(), df["MA20"].astype(float).to_list(), label="MA20", color="red")
ax.set_xlabel("Date")
ax.set_ylabel("Price (INR)")
ax.legend()
st.pyplot(fig)

# Graph 4: Outcome vs Prediction (last 10 days)
st.subheader("Outcome vs Prediction (Last 10 Days)")
df["Actual_Movement"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
last_10 = df.tail(10).copy()

preds = []
for i in range(len(last_10)):
    row = last_10[["Open", "High", "Low", "Volume"]].iloc[i].values.reshape(1, -1)
    row_scaled = scaler.transform(row)
    preds.append(model.predict(row_scaled)[0])
last_10["Predicted"] = preds

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(last_10.index.to_list(), last_10["Actual_Movement"].astype(int).to_list(),
        marker="o", label="Actual", color="blue")
ax.plot(last_10.index.to_list(), last_10["Predicted"].astype(int).to_list(),
        marker="x", label="Predicted", color="red")
ax.set_yticks([0, 1])
ax.set_yticklabels(["📉 Down", "📈 Up"])
ax.set_xlabel("Date")
ax.set_ylabel("Movement")
ax.legend()
st.pyplot(fig)

# Manual input prediction (4 features only)
st.sidebar.header("Manual Prediction Input")
open_price = st.sidebar.number_input("Open Price", value=float(df['Open'].iloc[-1]))
high_price = st.sidebar.number_input("High Price", value=float(df['High'].iloc[-1]))
low_price = st.sidebar.number_input("Low Price", value=float(df['Low'].iloc[-1]))
volume = st.sidebar.number_input("Volume", value=float(df['Volume'].iloc[-1]))

if st.sidebar.button("Predict from Manual Input"):
    manual_data = np.array([[open_price, high_price, low_price, volume]])
    manual_scaled = scaler.transform(manual_data)
    manual_pred_class = model.predict(manual_scaled)[0]
    manual_movement = "📈 Up" if manual_pred_class == 1 else "📉 Down"
    st.write(f"Manual Input Prediction: **{manual_movement}**")

st.success("✅ App is ready. Use the sidebar to adjust settings.")
