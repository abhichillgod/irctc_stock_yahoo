from flask import Flask, request, render_template_string
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from functools import lru_cache
import matplotlib.pyplot as plt
import io, base64
from datetime import datetime, timedelta

app = Flask(__name__)

# HTML template
TEMPLATE = """
<!doctype html>
<title>IRCTC Stock Predictor</title>
<h1>IRCTC Stock Prediction App</h1>
<form method=post>
  Start date: <input type=date name=start_date value="{{ start_date }}">
  End date: <input type=date name=end_date value="{{ end_date }}"><br><br>
  Prediction horizon (days): <input type=number name=horizon value="{{ horizon }}"><br><br>
  <input type=submit value="Predict">
</form>

{% if error %}
<p style="color:red">{{ error }}</p>
{% endif %}

{% if summary %}
<h2>Prediction Summary</h2>
<p>Data range: {{ min_date }} to {{ max_date }}</p>
<p>Test RMSE: {{ rmse }}</p>
<img src="data:image/png;base64,{{ history_img }}">
<img src="data:image/png;base64,{{ pred_img }}">
{% endif %}
"""

# Cache Yahoo Finance data
@lru_cache(maxsize=10)
def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

def prepare_lag_features(series, window):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i-window:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def train_and_predict(series, window, horizon):
    X, y = prepare_lag_features(series.values, window)
    if len(X) < 10:
        raise ValueError("Not enough data to train the model.")

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    last_window = list(series.values[-window:])
    future_preds = []
    for _ in range(horizon):
        arr = np.array(last_window[-window:]).reshape(1, -1)
        p = model.predict(arr)[0]
        future_preds.append(p)
        last_window.append(p)

    return rmse, future_preds

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    summary = None
    today = datetime.today().date()
    default_start = today - timedelta(days=365)
    start_date = default_start.isoformat()
    end_date = today.isoformat()
    horizon = 7

    if request.method == "POST":
        try:
            start_date = request.form.get("start_date", start_date)
            end_date = request.form.get("end_date", end_date)
            horizon = int(request.form.get("horizon", horizon))

            sd = pd.to_datetime(start_date)
            ed = pd.to_datetime(end_date) + timedelta(days=1)  # include end day

            df = get_stock_data("IRCTC.NS", sd.strftime("%Y-%m-%d"), ed.strftime("%Y-%m-%d"))
            if df.empty:
                raise ValueError("No data found for IRCTC in this date range.")

            df = df.sort_index()
            closes = df["Close"]

            rmse, preds = train_and_predict(closes, window=10, horizon=horizon)

            # Plot history
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(closes.index, closes.values, label="Close Price")
            ax1.set_title("IRCTC Historical Close Price")
            ax1.legend()
            history_img = fig_to_base64(fig1)

            # Plot prediction
            last_date = closes.index[-1]
            future_dates = [last_date + timedelta(days=i+1) for i in range(len(preds))]
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(closes.index, closes.values, label="Historical Close")
            ax2.plot(future_dates, preds, "r--o", label="Predicted")
            ax2.set_title("Predicted Future Prices")
            ax2.legend()
            pred_img = fig_to_base64(fig2)

            summary = True
            return render_template_string(TEMPLATE,
                                          start_date=start_date,
                                          end_date=end_date,
                                          horizon=horizon,
                                          summary=summary,
                                          min_date=df.index.min().date(),
                                          max_date=df.index.max().date(),
                                          rmse=round(rmse, 2),
                                          history_img=history_img,
                                          pred_img=pred_img,
                                          error=None)

        except Exception as e:
            error = str(e)

    return render_template_string(TEMPLATE,
                                  start_date=start_date,
                                  end_date=end_date,
                                  horizon=horizon,
                                  summary=summary,
                                  error=error)

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
