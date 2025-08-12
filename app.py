import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import date

# ------------------------------
# Load trained model & preprocessing
# ------------------------------
with open("irctc_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("irctc_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ------------------------------
# App Title
# ------------------------------
st.set_page_config(page_title="Stock Prediction App", layout="wide")
st.title("📈 Stock Price Prediction App")

# ------------------------------
# Sidebar for user input
# ------------------------------
st.sidebar.header("User Input Parameters")

stock_list = ["IRCTC", "TCS", "RELIANCE", "INFY", "HDFC"]  # Update based on your dataset
stock_name = st.sidebar.selectbox("Select Stock", stock_list)

start_date = st.sidebar.date_input("Start Date", date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())

graph_type = st.sidebar.selectbox(
    "Select Graph Type",
    [
        "Line Chart",
        "Candlestick",
        "Bar Chart",
        "Area Chart",
        "Scatter Plot",
        "Histogram"
    ]
)

# ------------------------------
# Load Stock Data
# ------------------------------
@st.cache_data
def load_stock_data(stock):
    file_path = f"data/{stock}.csv"  # Adjust path if needed
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
    return df

try:
    df = load_stock_data(stock_name)
except FileNotFoundError:
    st.error(f"No data file found for {stock_name}. Please ensure 'data/{stock_name}.csv' exists.")
    st.stop()

# ------------------------------
# Prepare Data for Prediction
# ------------------------------
features = df.drop(columns=["Date", "Close"])
scaled_features = scaler.transform(features)
predictions = model.predict(scaled_features)

# Add predictions to dataframe
df["Prediction"] = predictions

# ------------------------------
# Display Graph
# ------------------------------
st.subheader(f"{stock_name} Stock Price Prediction ({graph_type})")

if graph_type == "Line Chart":
    fig = px.line(df, x="Date", y=["Close", "Prediction"], labels={"value": "Price"})
elif graph_type == "Candlestick":
    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )])
elif graph_type == "Bar Chart":
    fig = px.bar(df, x="Date", y="Close")
elif graph_type == "Area Chart":
    fig = px.area(df, x="Date", y="Close")
elif graph_type == "Scatter Plot":
    fig = px.scatter(df, x="Date", y="Close")
elif graph_type == "Histogram":
    fig = px.histogram(df, x="Close")

st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Display Prediction Outcome
# ------------------------------
final_pred = df["Prediction"].iloc[-1]
last_actual = df["Close"].iloc[-1]

if final_pred > last_actual:
    st.success("📈 Prediction: The stock is likely to go UP.")
else:
    st.error("📉 Prediction: The stock is likely to go DOWN.")
