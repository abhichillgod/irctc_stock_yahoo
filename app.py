
import streamlit as st

st.set_page_config(page_title="IRCTC Stock Prediction", layout="wide")

# Original Notebook Code (adapted to Streamlit)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

# 1️⃣ User inputs
ticker = input("Enter stock ticker (default: IRCTC.NS): ") or "IRCTC.NS"
start_date = input("Enter start date (YYYY-MM-DD): ")
end_date = input("Enter end date (YYYY-MM-DD): ")

# 2️⃣ Download IRCTC data
data = yf.download('IRCTC.NS', start=start_date, end=end_date)
irctc = pd.DataFrame(data)

# 3️⃣ Calculate additional columns
irctc['Return'] = irctc['Close'].pct_change()  # Daily return
irctc['MA5'] = irctc['Close'].rolling(window=5).mean()
irctc['MA10'] = irctc['Close'].rolling(window=10).mean()
irctc['MA20'] = irctc['Close'].rolling(window=20).mean()
irctc['Vol_MA5'] = irctc['Volume'].rolling(window=5).mean()

# Target: 1 if next day's close > today's close, else 0
irctc['Target'] = (irctc['Close'].shift(-1) > irctc['Close']).astype(int)

# 4️⃣ Drop NaN values created by rolling calculations
irctc.dropna(inplace=True)

# 5️⃣ Display first few rows
print(irctc.head())

# 6️⃣ Plot Target count
sns.countplot(x='Target', data=irctc)
plt.title("Target Value Counts (IRCTC)")
plt.show()


!pip install yfinance


irctc.head(10)

irctc.tail(10)

print("\nDataset Info:")
print(irctc.info())

# 📊 Visualization Section
import matplotlib.pyplot as plt
import seaborn as sns

# Make plots look nicer
sns.set(style="whitegrid")

# 1. Distribution of each feature
irctc.hist(figsize=(12, 10), bins=20, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

irctc.describe()


irctc.corr()


# 3. Correlation heatmap
plt.figure(figsize=(10, 8))
corr = irctc.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure(figsize=(12,6))
sns.histplot(data=irctc[['Open','High','Low','Close']], kde=True)
plt.show()

sns.pairplot(irctc[['Open', 'High', 'Low', 'Close', 'Volume', 'Return']])
plt.suptitle("Pairwise Relationship between IRCTC Stock Attributes", y=1.02)
plt.show()


plt.figure(figsize=(12,6))
plt.plot(irctc.index, irctc['Close'], label='Close Price', color='blue')
plt.plot(irctc.index, irctc['MA5'], label='MA5', color='orange')
plt.plot(irctc.index, irctc['MA10'], label='MA10', color='green')
plt.plot(irctc.index, irctc['MA20'], label='MA20', color='red')
plt.title("Close Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler# Using Open, High, Low, Volume as features to predict Close
X = irctc[['Open', 'High', 'Low', 'Volume']]
y = irctc['Close']

print("Feature set shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled training data shape:", X_train_scaled.shape)
print("Scaled test data shape:", X_test_scaled.shape)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("Model trained successfully.")

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Closing Prices")
plt.show()

# Example: Predict using the latest available data
latest_data = irctc[['Open', 'High', 'Low', 'Volume']].iloc[-1].values.reshape(1, -1)
latest_scaled = scaler.transform(latest_data)
predicted_price = model.predict(latest_scaled)[0]  # returns a numpy array or scalar
predicted_price_value = predicted_price.item()     # safely get scalar from array
print(f"Predicted closing price for {ticker} on the next trading day: {predicted_price_value:.2f} INR")

import joblib
joblib.dump(model, "irctc_model.pkl")
joblib.dump(scaler, "irctc_scaler.pkl")
print("Model and scaler saved.")

loaded_model = joblib.load("irctc_model.pkl")
loaded_scaler = joblib.load("irctc_scaler.pkl")

# Example with loaded model
sample_data = latest_data
sample_scaled = loaded_scaler.transform(sample_data)
sample_pred = loaded_model.predict(sample_scaled)[0].item()
print(f"Loaded model prediction: {sample_pred:.2f} INR")


