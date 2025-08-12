"""
Stock Prediction Web App (app.py)

Single-file Flask app that:
- Takes user input: stock ticker, start date, end date, prediction horizon (days), model window size
- Lets user choose which graphs to display (historical price, moving average, predicted future)
- Uses yfinance to download history, trains a RandomForestRegressor on lag features, and predicts next N days
- Renders the plots inline by converting matplotlib figures to base64 PNGs

Requirements:
pip install flask yfinance pandas numpy scikit-learn matplotlib

Run:
python app.py
Open http://127.0.0.1:5000 in your browser

Note: This is a simple demonstration model. For production or higher accuracy, use more advanced time-series models and more careful backtesting.
"""

from flask import Flask, request, render_template_string, redirect, url_for
import io, base64
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

app = Flask(__name__)

TEMPLATE = '''
<!doctype html>
<title>Stock Predictor</title>
<h1>Simple Stock Prediction App</h1>
<form method=post>
  Ticker (e.g. AAPL): <input type=text name=ticker value="{{ ticker }}"><br>
  Start date: <input type=date name=start_date value="{{ start_date }}"> &nbsp;
  End date: <input type=date name=end_date value="{{ end_date }}"><br>
  Prediction horizon (days): <input type=number name=horizon min=1 max=365 value="{{ horizon }}"><br>
  Window size (lag days to use as features): <input type=number name=window min=1 max=60 value="{{ window }}"><br>
  Graphs to show:<br>
  <input type=checkbox name=graphs value="history" {% if 'history' in graphs %}checked{% endif %}> Historical Close Price<br>
  <input type=checkbox name=graphs value="ma" {% if 'ma' in graphs %}checked{% endif %}> Moving Average (window=20)<br>
  <input type=checkbox name=graphs value="pred" {% if 'pred' in graphs %}checked{% endif %}> Predicted Future<br>
  <br>
  <input type=submit value=Predict>
</form>

{% if error %}
  <p style="color:red">{{ error }}</p>
{% endif %}

{% if summary %}
  <h2>Summary for {{ ticker.upper() }}</h2>
  <p>Data range: {{ min_date }} to {{ max_date }} ({{ n_days }} trading days)</p>
  <p>Model: RandomForestRegressor with window={{ window }}; Test RMSE: {{ rmse }}</p>

  {% for img in images %}
    <div style="margin-bottom:20px">
      <img src="data:image/png;base64,{{ img }}" style="max-width:900px; width:100%">
    </div>
  {% endfor %}
{% endif %}

<hr>
<p>Notes: This app downloads historical data from Yahoo Finance using the <code>yfinance</code> package. The prediction is a simple ML model trained on lagged close prices, intended for demo and educational purposes only.</p>
'''


def prepare_lag_features(series, window):
    """Create lag features for a series. Returns DataFrame of features and target."""
    X = []
    y = []
    for i in range(window, len(series)):
        X.append(series[i - window:i].values)
        y.append(series[i])
    X = np.array(X)
    y = np.array(y)
    return X, y


def train_and_predict(close_series, window, horizon, random_state=42):
    """Train a RandomForest on lag features and forecast horizon days iteratively."""
    # Build lag features
    X, y = prepare_lag_features(close_series, window)
    if len(X) < 10:
        raise ValueError("Not enough data to build model with the selected window. Try expanding date range or reducing window.")

    # Use the last 20% as test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # Forecast iteratively
    last_window = close_series.values[-window:].tolist()
    future_preds = []
    for _ in range(horizon):
        arr = np.array(last_window[-window:]).reshape(1, -1)
        p = float(model.predict(arr)[0])
        future_preds.append(p)
        last_window.append(p)

    return model, rmse, future_preds


def plot_history(dates, closes):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, closes, label='Close')
    ax.set_title('Historical Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_moving_average(dates, closes, ma_window=20):
    s = pd.Series(closes, index=pd.to_datetime(dates))
    ma = s.rolling(window=ma_window).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, closes, label='Close')
    ax.plot(ma.index, ma.values, label=f'{ma_window}-day MA')
    ax.set_title(f'Close Price and {ma_window}-day Moving Average')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_prediction(dates, closes, future_preds):
    last_date = pd.to_datetime(dates[-1])
    future_dates = [last_date + timedelta(days=i + 1) for i in range(len(future_preds))]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, closes, label='Historical Close')
    ax.plot(future_dates, future_preds, label='Predicted Close', linestyle='--', marker='o')
    ax.set_title('Historical and Predicted Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_b64


@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    images = []
    summary = None
    ticker = 'AAPL'
    today = datetime.today().date()
    default_start = (today - timedelta(days=365)).isoformat()
    default_end = today.isoformat()
    start_date = default_start
    end_date = default_end
    horizon = 7
    window = 10
    graphs = ['history', 'ma', 'pred']

    if request.method == 'POST':
        try:
            ticker = request.form.get('ticker', 'AAPL').strip().upper()
            start_date = request.form.get('start_date', default_start)
            end_date = request.form.get('end_date', default_end)
            horizon = int(request.form.get('horizon', 7))
            window = int(request.form.get('window', 10))
            # graphs may be multiple checkboxes; Flask returns last if name same; handle differently
            raw_graphs = request.form.getlist('graphs')
            graphs = raw_graphs if raw_graphs else []

            # Validate dates
            sd = pd.to_datetime(start_date)
            ed = pd.to_datetime(end_date)
            if sd >= ed:
                raise ValueError('Start date must be before end date')

            # Download data
            df = yf.download(ticker, start=sd.strftime('%Y-%m-%d'), end=(ed + timedelta(days=1)).strftime('%Y-%m-%d'))
            if df.empty:
                raise ValueError('No data returned for this ticker and date range. Check ticker symbol and dates.')

            df = df.sort_index()
            closes = df['Close']

            # Train and predict
            model, rmse, future_preds = train_and_predict(closes, window, horizon)

            # Build plots as requested
            if 'history' in graphs:
                fig = plot_history(closes.index, closes.values)
                images.append(fig_to_base64(fig))
            if 'ma' in graphs:
                fig = plot_moving_average(closes.index, closes.values, ma_window=20)
                images.append(fig_to_base64(fig))
            if 'pred' in graphs:
                fig = plot_prediction(closes.index, closes.values, future_preds)
                images.append(fig_to_base64(fig))

            summary = True
            min_date = df.index.min().date()
            max_date = df.index.max().date()
            n_days = len(df)

            return render_template_string(TEMPLATE,
                                          ticker=ticker,
                                          start_date=start_date,
                                          end_date=end_date,
                                          horizon=horizon,
                                          window=window,
                                          graphs=graphs,
                                          images=images,
                                          summary=summary,
                                          min_date=min_date,
                                          max_date=max_date,
                                          n_days=n_days,
                                          rmse=round(rmse, 4),
                                          error=None)

        except Exception as e:
            error = str(e)

    return render_template_string(TEMPLATE,
                                  ticker=ticker,
                                  start_date=start_date,
                                  end_date=end_date,
                                  horizon=horizon,
                                  window=window,
                                  graphs=graphs,
                                  images=images,
                                  summary=summary,
                                  min_date=None,
                                  max_date=None,
                                  n_days=0,
                                  rmse=None,
                                  error=error)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

