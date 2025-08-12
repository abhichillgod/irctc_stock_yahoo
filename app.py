import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf

# ---------------------
# App Configuration
# ---------------------
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Stock Price Prediction Dashboard")
st.markdown(
    "<p style='font-size:18px;'>Select your stock, choose a date range, and view predictions with interactive charts.</p>",
    unsafe_allow_html=True
)

# ---------------------
# Sidebar for Inputs
# ---------------------
st.sidebar.header("⚙️ Settings")

# Stock options (You can extend this list)
stock_options = {
    "IRCTC": "IRCTC.NS",
    "TCS": "TCS.NS",
    "Reliance": "RELIANCE.NS",
    "Infosys": "INFY.NS"
}

stock_name = st.sidebar.selectbox("Select Stock", list(stock_options.keys()))
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now())

graph_type = st.sidebar.selectbox(
    "Select Graph Type",
    ["Line Chart", "Candlestick"]
)

# ---------------------
# Data Fetching
# ---------------------
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    return df

df = load_data(stock_options[stock_name], start_date, end_date)

if df.empty:
    st.warning("⚠️ No data found for the selected date range.")
else:
    st.success(f"✅ Data loaded for **{stock_name}** from {start_date} to {end_date}")

    # ---------------------
    # Display Graph
    # ---------------------
    if graph_type == "Line Chart":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name='Closing Price',
            line=dict(color='royalblue', width=2)
        ))
        fig.update_layout(title=f"{stock_name} Closing Price Over Time", xaxis_title="Date", yaxis_title="Price (₹)")
        st.plotly_chart(fig, use_container_width=True)

    elif graph_type == "Candlestick":
        fig = go.Figure(data=[go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='green',
            decreasing_line_color='red'
        )])
        fig.update_layout(title=f"{stock_name} Candlestick Chart", xaxis_title="Date", yaxis_title="Price (₹)")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------
    # Show Predictions Placeholder
    # ---------------------
    st.subheader("🔮 Stock Price Predictions")
    st.info("Prediction model integration coming soon — will display ML-based forecasts here.")

    # ---------------------
    # Show Data Table
    # ---------------------
    with st.expander("📄 View Raw Data"):
        st.dataframe(df)
