import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# Streamlit App Title
# -------------------------------
st.title("📈 IRCTC Stock Analysis & Prediction App")

# -------------------------------
# User Inputs
# -------------------------------
st.sidebar.header("User Input Parameters")

ticker = st.sidebar.text_input("Stock Ticker", value="IRCTC.NS")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")
show_graph_type = st.sidebar.selectbox("Select Graph Type", ["Line Chart", "Moving Averages", "Return Distribution", "Target Countplot"])

# -------------------------------
# Download Stock Data
# -------------------------------
if st.sidebar.button("Run Analysis"):
    if not start_date or not end_date:
        st.error("Please select start and end dates.")
    else:
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            st.error("No data found. Check ticker or date range.")
        else:
            irctc = pd.DataFrame(data)

            # Calculate additional columns
            irctc['Return'] = irctc['Close'].pct_change()
            irctc['MA5'] = irctc['Close'].rolling(window=5).mean()
            irctc['MA10'] = irctc['Close'].rolling(window=10).mean()
            irctc['MA20'] = irctc['Close'].rolling(window=20).mean()
            irctc['Vol_MA5'] = irctc['Volume'].rolling(window=5).mean()
            irctc['Target'] = (irctc['Close'].shift(-1) > irctc['Close']).astype(int)

            irctc.dropna(inplace=True)

            st.subheader("First 5 Rows of Data")
            st.dataframe(irctc.head())

            # Graphs
            if show_graph_type == "Line Chart":
                st.line_chart(irctc[['Close', 'MA5', 'MA10', 'MA20']])
            
            elif show_graph_type == "Moving Averages":
                fig, ax = plt.subplots()
                ax.plot(irctc.index, irctc['Close'], label='Close Price')
                ax.plot(irctc.index, irctc['MA5'], label='MA5')
                ax.plot(irctc.index, irctc['MA10'], label='MA10')
                ax.plot(irctc.index, irctc['MA20'], label='MA20')
                ax.legend()
                ax.set_title("Moving Averages")
                st.pyplot(fig)

            elif show_graph_type == "Return Distribution":
                fig, ax = plt.subplots()
                sns.histplot(irctc['Return'], bins=50, kde=True, ax=ax)
                ax.set_title("Return Distribution")
                st.pyplot(fig)

            elif show_graph_type == "Target Countplot":
                fig, ax = plt.subplots()
                sns.countplot(x='Target', data=irctc, ax=ax)
                ax.set_title("Target Value Counts")
                st.pyplot(fig)

# -------------------------------
# Footer
# -------------------------------
st.sidebar.markdown("**Built with ❤️ using Streamlit**")
