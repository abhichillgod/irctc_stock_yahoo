import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import matplotlib.pyplot as plt

# Load saved model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("IRCTC Stock Price Prediction App 📈")
st.markdown("Predict the price movement of IRCTC stocks based on historical data.")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter Stock Ticker", "IRCTC.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Download stock data
df = yf.download(ticker, start=start_date, end=end_date)

if df.empty:
    st.error("No data found for the given ticker and date range.")
else:
    # Feature Engineering
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["Return"] = df["Close"].pct_change()

    df.dropna(inplace=True)

    features = ["Close", "MA5", "MA10", "MA20", "Volume", "Return"]
    latest_data = df[features].iloc[-1:].values

    # Scale features
    latest_scaled = scaler.transform(latest_data)

    # Make prediction
    prediction = model.predict(latest_scaled)
    predicted_class = int(prediction[0])

    movement = "📈 UP" if predicted_class == 1 else "📉 DOWN"
    st.subheader(f"Predicted Price Movement: {movement}")

    # Graph 1: Closing Price Trend with prediction marker
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index.tolist(), df["Close"].astype(float).tolist(),
            label="Closing Price", color="blue")
    ax.scatter([df.index[-1]], [float(df["Close"].iloc[-1])],
               color="green" if predicted_class == 1 else "red",
               s=100, label="Prediction")
    ax.set_title("Closing Price Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # Graph 2: Volume Trend
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df.index.tolist(), df["Volume"].astype(float).tolist(), color="orange")
    ax.set_title("Trading Volume Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    st.pyplot(fig)

    # Graph 3: Moving Averages
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index.tolist(), df["Close"].astype(float).tolist(), label="Close Price", color="blue")
    ax.plot(df.index.tolist(), df["MA5"].astype(float).tolist(), label="MA5", color="green")
    ax.plot(df.index.tolist(), df["MA10"].astype(float).tolist(), label="MA10", color="orange")
    ax.plot(df.index.tolist(), df["MA20"].astype(float).tolist(), label="MA20", color="red")
    ax.set_title("Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # Dummy outcome vs prediction for demonstration (replace with real values if available)
    last_10 = df.tail(10).copy()
    last_10["Actual_Movement"] = np.random.randint(0, 2, size=len(last_10))
    last_10["Predicted"] = np.random.randint(0, 2, size=len(last_10))

    # Graph 4: Outcome vs Prediction
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(last_10.index.tolist(),
            last_10["Actual_Movement"].astype(int).tolist(),
            marker="o", label="Actual", color="blue")
    ax.plot(last_10.index.tolist(),
            last_10["Predicted"].astype(int).tolist(),
            marker="x", label="Predicted", color="red")
    ax.set_title("Actual vs Predicted Movements (Last 10 Days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Movement (0 = Down, 1 = Up)")
    ax.legend()
    st.pyplot(fig)
