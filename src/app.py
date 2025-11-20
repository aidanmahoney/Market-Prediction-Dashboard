import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Page config
st.set_page_config(
    page_title="Market Predictor",
    page_icon=":chart_increasing:",
    layout="wide"
)

# Cache downloading of historical data
@st.cache_data
def load_data(ticker: str) -> pd.DataFrame:
    end_date = dt.datetime.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start="2010-01-01", end=end_date)
    if data.empty:
        raise ValueError(f"No data available for {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

# Cache feature engineering
@st.cache_data
def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df[['Close']].dropna().copy()
    if len(df) < 200:
        raise ValueError("Not enough data to compute indicators.")
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Momentum'] = df['Close'].diff(periods=5)
    df.dropna(subset=['SMA_50', 'SMA_200', 'Momentum'], inplace=True)
    X = df[['SMA_50', 'SMA_200', 'Momentum']]
    y = df['Close']
    return X, y

# Cache the trained model and reuse
@st.cache_resource
def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[RandomForestRegressor, float, pd.DatetimeIndex, np.ndarray, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse, X_test.index, y_pred, y_test

# Generate future feature matrix and predictions
def create_future_features(
    df: pd.DataFrame,
    model: RandomForestRegressor,
    last_close: float,
    days: int = 30
) -> tuple[pd.DataFrame, list[float]]:
    features: list[list[float]] = []
    predictions: list[float] = []

    # Get recent historical prices for SMA calculations
    recent_prices = df['Close'].tolist()

    for i in range(days):
        # Calculate SMAs from actual prices (historical + predicted so far)
        all_prices = recent_prices + predictions

        if len(all_prices) >= 50:
            sma50 = float(np.mean(all_prices[-50:]))
        else:
            sma50 = float(np.mean(all_prices))

        if len(all_prices) >= 200:
            sma200 = float(np.mean(all_prices[-200:]))
        else:
            sma200 = float(np.mean(all_prices))

        # Calculate momentum as 5-day price change
        if len(all_prices) >= 5:
            momentum = float(all_prices[-1] - all_prices[-5])
        else:
            momentum = 0.0

        # Create feature vector and predict
        feat_vec = np.array([[sma50, sma200, momentum]])
        pred = float(model.predict(feat_vec)[0])

        features.append([sma50, sma200, momentum])
        predictions.append(pred)

    idx = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=days
    )
    future_df = pd.DataFrame(
        features,
        columns=['SMA_50', 'SMA_200', 'Momentum'],
        index=idx
    )
    return future_df, predictions

# Main dashboard
def main() -> None:
    st.title("Market Recovery Prediction Dashboard")

    ticker = st.text_input(
        "Enter stock symbol (e.g., SPY):", value="SPY"
    )
    if not ticker:
        return

    try:
        data = load_data(ticker)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    st.subheader(f"{ticker} Historical Close Price")
    st.line_chart(data['Close'])

    try:
        X, y = create_features(data)
    except Exception as e:
        st.error(str(e))
        return

    model, mse, test_idx, y_pred, y_test = train_model(X, y)
    st.subheader("Model Performance on Test Set")
    st.write(f"Mean Squared Error: {mse:.2f}")
    perf_df = pd.DataFrame(
        {'Actual': y_test, 'Predicted': y_pred},
        index=test_idx
    )
    st.line_chart(perf_df)

    years = st.slider(
        "Select forecast horizon (years):", 1, 3, 1
    )
    future_days = years * 252

    processed = X.copy()
    processed['Close'] = y

    future_df, future_prices = create_future_features(
        processed,
        model,
        last_close=processed['Close'].iloc[-1],
        days=future_days
    )
    future_df['Predicted'] = future_prices

    combined_df = pd.concat(
        [data['Close'], future_df['Predicted']],
        axis=1
    )
    combined_df.columns = ['Actual', 'Predicted']

    st.subheader("Combined Historical and Forecasted Prices")
    st.line_chart(combined_df)

if __name__ == "__main__":
    main()
