import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(
    page_title="Market Predictor",
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
def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = df[['Close']].dropna().copy()
    if len(df) < 200:
        raise ValueError("Not enough data to compute indicators.")

    # Calculate returns (percentage change)
    df['Returns'] = df['Close'].pct_change()

    # Target: Next day's return (what we're trying to predict)
    df['Target_Return'] = df['Returns'].shift(-1)

    # Lag features (previous day returns - more stationary than prices)
    df['Return_Lag_1'] = df['Returns'].shift(1)
    df['Return_Lag_2'] = df['Returns'].shift(2)
    df['Return_Lag_5'] = df['Returns'].shift(5)

    # Moving averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # Price position relative to SMAs (percentage)
    df['Price_to_SMA10'] = (df['Close'] - df['SMA_10']) / df['SMA_10'] * 100
    df['Price_to_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50'] * 100
    df['Price_to_SMA200'] = (df['Close'] - df['SMA_200']) / df['SMA_200'] * 100

    # Volatility (rolling std of returns)
    df['Volatility_20'] = df['Returns'].rolling(window=20).std()
    df['Volatility_50'] = df['Returns'].rolling(window=50).std()

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD as percentage of price
    df['MACD'] = (df['EMA_12'] - df['EMA_26']) / df['Close'] * 100

    # Bollinger Bands position
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # Rate of Change
    df['ROC'] = df['Close'].pct_change(periods=10) * 100

    df.dropna(inplace=True)

    X = df[['Return_Lag_1', 'Return_Lag_2', 'Return_Lag_5',
            'Price_to_SMA10', 'Price_to_SMA50', 'Price_to_SMA200',
            'Volatility_20', 'Volatility_50', 'RSI', 'MACD',
            'BB_position', 'ROC']]
    y = df['Target_Return']
    close_prices = df['Close']
    return X, y, close_prices

# Cache the trained model and reuse
@st.cache_resource
def train_model(X: pd.DataFrame, y: pd.Series, close_prices: pd.Series) -> tuple:
    # Use time-based split (no shuffle for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use Gradient Boosting with simpler parameters
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
        verbose=0
    )

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Calculate multiple metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Convert returns back to prices for visualization
    test_close_prices = close_prices.loc[X_test.index]
    pred_prices = test_close_prices * (1 + y_pred)
    actual_next_prices = test_close_prices * (1 + y_test)

    return model, scaler, mse, rmse, mae, r2, X_test.index, pred_prices, actual_next_prices

# Generate future feature matrix and predictions
def create_future_features(
    df: pd.DataFrame,
    model: GradientBoostingRegressor,
    scaler: StandardScaler,
    last_close: float,
    days: int = 30
) -> tuple[pd.DataFrame, list[float]]:
    features: list[list[float]] = []
    predictions: list[float] = []

    # Get recent historical prices and returns
    recent_prices = df['Close'].tolist()
    recent_returns = df['Returns'].dropna().tolist()

    for i in range(days):
        # Combine historical and predicted prices
        all_prices = recent_prices + predictions

        # Current price
        current_price = all_prices[-1]

        # Calculate returns for lag features
        all_returns = recent_returns.copy()
        if len(predictions) > 0:
            for j in range(len(predictions)):
                if j == 0:
                    ret = (predictions[0] - recent_prices[-1]) / recent_prices[-1]
                else:
                    ret = (predictions[j] - predictions[j-1]) / predictions[j-1]
                all_returns.append(ret)

        # Return lag features
        return_lag_1 = all_returns[-1] if len(all_returns) >= 1 else 0.0
        return_lag_2 = all_returns[-2] if len(all_returns) >= 2 else 0.0
        return_lag_5 = all_returns[-5] if len(all_returns) >= 5 else 0.0

        # Moving averages
        sma_10 = float(np.mean(all_prices[-10:])) if len(all_prices) >= 10 else current_price
        sma_50 = float(np.mean(all_prices[-50:])) if len(all_prices) >= 50 else current_price
        sma_200 = float(np.mean(all_prices[-200:])) if len(all_prices) >= 200 else current_price

        # Price position relative to SMAs
        price_to_sma10 = float((current_price - sma_10) / sma_10 * 100) if sma_10 != 0 else 0.0
        price_to_sma50 = float((current_price - sma_50) / sma_50 * 100) if sma_50 != 0 else 0.0
        price_to_sma200 = float((current_price - sma_200) / sma_200 * 100) if sma_200 != 0 else 0.0

        # Volatility (std of recent returns)
        volatility_20 = float(np.std(all_returns[-20:])) if len(all_returns) >= 20 else 0.0
        volatility_50 = float(np.std(all_returns[-50:])) if len(all_returns) >= 50 else 0.0

        # RSI calculation
        if len(all_prices) >= 15:
            price_changes = np.diff(all_prices[-15:])
            gains = price_changes[price_changes > 0]
            losses = -price_changes[price_changes < 0]
            avg_gain = gains.mean() if len(gains) > 0 else 0.0
            avg_loss = losses.mean() if len(losses) > 0 else 0.0
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = float(100 - (100 / (1 + rs)))
            else:
                rsi = 100.0
        else:
            rsi = 50.0

        # EMAs for MACD
        if len(all_prices) >= 12:
            weights_12 = np.exp(np.linspace(-1., 0., 12))
            weights_12 /= weights_12.sum()
            ema_12 = float(np.average(all_prices[-12:], weights=weights_12))
        else:
            ema_12 = current_price

        if len(all_prices) >= 26:
            weights_26 = np.exp(np.linspace(-1., 0., 26))
            weights_26 /= weights_26.sum()
            ema_26 = float(np.average(all_prices[-26:], weights=weights_26))
        else:
            ema_26 = current_price

        # MACD as percentage
        macd = (ema_12 - ema_26) / current_price * 100 if current_price != 0 else 0.0

        # Bollinger Bands
        if len(all_prices) >= 20:
            bb_middle = float(np.mean(all_prices[-20:]))
            bb_std = float(np.std(all_prices[-20:]))
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else 0.5
        else:
            bb_position = 0.5

        # Rate of Change
        roc = float((all_prices[-1] - all_prices[-10]) / all_prices[-10] * 100) if len(all_prices) >= 10 else 0.0

        # Create feature vector matching training order
        feat_vec = np.array([[return_lag_1, return_lag_2, return_lag_5,
                             price_to_sma10, price_to_sma50, price_to_sma200,
                             volatility_20, volatility_50, rsi, macd,
                             bb_position, roc]])

        # Scale features
        feat_vec_scaled = scaler.transform(feat_vec)

        # Predict next day return
        pred_return = float(model.predict(feat_vec_scaled)[0])

        # Apply reasonable bounds for daily returns (±5%)
        pred_return = np.clip(pred_return, -0.05, 0.05)

        # Convert return to price
        pred_price = current_price * (1 + pred_return)

        features.append([return_lag_1, return_lag_2, return_lag_5,
                        price_to_sma10, price_to_sma50, price_to_sma200,
                        volatility_20, volatility_50, rsi, macd,
                        bb_position, roc])
        predictions.append(float(pred_price))

    idx = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=days
    )
    future_df = pd.DataFrame(
        features,
        columns=['Return_Lag_1', 'Return_Lag_2', 'Return_Lag_5',
                 'Price_to_SMA10', 'Price_to_SMA50', 'Price_to_SMA200',
                 'Volatility_20', 'Volatility_50', 'RSI', 'MACD',
                 'BB_position', 'ROC'],
        index=idx
    )
    return future_df, predictions

# Main dashboard
def main() -> None:
    st.title("Market Prediction Dashboard")

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
        X, y, close_prices = create_features(data)
    except Exception as e:
        st.error(str(e))
        return

    model, scaler, mse, rmse, mae, r2, test_idx, y_pred, y_test = train_model(X, y, close_prices)
    st.subheader("Model Performance on Test Set")

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("RMSE", f"${rmse:.2f}")
    with col3:
        st.metric("MAE", f"${mae:.2f}")
    with col4:
        # Calculate directional accuracy
        correct_direction = sum((y_pred - close_prices.loc[test_idx]) * (y_test - close_prices.loc[test_idx]) > 0)
        dir_accuracy = correct_direction / len(y_test) * 100
        st.metric("Direction Accuracy", f"{dir_accuracy:.1f}%")

    perf_df = pd.DataFrame(
        {'Actual': y_test, 'Predicted': y_pred},
        index=test_idx
    )
    st.line_chart(perf_df)

    # Show explanation of metrics
    with st.expander("Understanding Model Metrics"):
        st.write("""
        - **R² Score**: Measures how well the model explains variance (1.0 is perfect, closer to 1 is better)
        - **RMSE**: Root Mean Squared Error - average prediction error in dollars for next-day prices
        - **MAE**: Mean Absolute Error - average absolute prediction error in dollars
        - **Direction Accuracy**: Percentage of times the model correctly predicted the direction of price movement

        Note: The model predicts next-day returns (percentage changes) which are then converted to prices.
        """)

    years = st.slider(
        "Select forecast horizon (years):", 1, 3, 1
    )
    future_days = years * 252

    # Create a dataframe with Close and Returns for future predictions
    processed = pd.DataFrame({
        'Close': close_prices,
        'Returns': close_prices.pct_change()
    }, index=close_prices.index)

    future_df, future_prices = create_future_features(
        processed,
        model,
        scaler,
        last_close=processed['Close'].iloc[-1],
        days=future_days
    )
    future_df['Predicted'] = future_prices

    # Calculate prediction confidence intervals (widening over time)
    hist_volatility = processed['Close'].pct_change().std()
    confidence_lower = []
    confidence_upper = []

    for i, price in enumerate(future_prices):
        # Uncertainty grows with prediction horizon
        days_ahead = i + 1
        uncertainty = price * hist_volatility * np.sqrt(days_ahead) * 1.96  # 95% confidence
        confidence_lower.append(price - uncertainty)
        confidence_upper.append(price + uncertainty)

    future_df['Lower_95'] = confidence_lower
    future_df['Upper_95'] = confidence_upper

    combined_df = pd.concat(
        [data['Close'], future_df['Predicted']],
        axis=1
    )
    combined_df.columns = ['Actual', 'Predicted']

    st.subheader("Combined Historical and Forecasted Prices")
    st.line_chart(combined_df)

    # Show prediction with confidence intervals
    st.subheader("Forecast with 95% Confidence Interval")
    forecast_display = future_df[['Predicted', 'Lower_95', 'Upper_95']].copy()
    forecast_display.columns = ['Forecast', '95% CI Lower', '95% CI Upper']
    st.line_chart(forecast_display)

    st.info(
        "**Understanding the forecast:**\n\n"
        "- The model predicts next-day returns using technical indicators (RSI, MACD, price/SMA ratios, volatility)\n"
        "- Returns are converted to prices for visualization\n"
        "- Daily return predictions are capped at ±5% for stability\n"
        "- Confidence intervals widen over time, showing increased uncertainty\n"
        "- Longer-term predictions should be taken with caution\n\n"
        "**Note:** This is a simplified model for educational purposes. Real trading requires much more sophisticated analysis."
    )

if __name__ == "__main__":
    main()
