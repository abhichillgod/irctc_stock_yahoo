import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="IRCTC Stock Dashboard", layout="wide")

st.title("📈 IRCTC Stock Dashboard")

# Input for ticker (default IRCTC.NS for NSE India)
ticker = st.text_input("Enter Stock Ticker Symbol", "IRCTC.NS")

# Date range
period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
period = st.selectbox("Select Time Period", period_options, index=3)

if st.button("Fetch Stock Data"):
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            st.error("No data found. Please check the ticker symbol.")
        else:
            st.subheader(f"Last 5 Rows of {ticker} Data")
            st.dataframe(data.tail())

            # Plot Closing Price
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data.index, data['Close'], label="Close Price", color="blue")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (INR)")
            ax.set_title(f"{ticker} Closing Price")
            ax.legend()
            st.pyplot(fig)

            # Show summary statistics
            st.subheader("📊 Summary Statistics")
            st.write(data.describe())

    except Exception as e:
        st.error(f"Error fetching data: {e}")
