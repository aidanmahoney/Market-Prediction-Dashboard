import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Data requirements
MIN_DATA_LENGTH = 200

# Technical indicator windows
SMA_SHORT_WINDOW = 10
SMA_MEDIUM_WINDOW = 50
SMA_LONG_WINDOW = 200
EMA_FAST_SPAN = 12
EMA_SLOW_SPAN = 26
VOLATILITY_SHORT_WINDOW = 20
VOLATILITY_LONG_WINDOW = 50
RSI_WINDOW = 14
BOLLINGER_WINDOW = 20
BOLLINGER_STD_MULTIPLIER = 2
ROC_PERIODS = 10

# Model configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Gradient Boosting parameters
N_ESTIMATORS = 200
LEARNING_RATE = 0.1
MAX_DEPTH = 3
MIN_SAMPLES_SPLIT = 20
MIN_SAMPLES_LEAF = 10
SUBSAMPLE = 0.8

# Prediction constraints
MAX_DAILY_RETURN = 0.05
MIN_DAILY_RETURN = -0.05

# Trading calendar
TRADING_DAYS_PER_YEAR = 252

# Confidence intervals
CONFIDENCE_95_Z_SCORE = 1.96

# Technical indicator calculation functions
def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return prices.rolling(window=window).mean()

def calculate_ema(prices: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return prices.ewm(span=span, adjust=False).mean()

def calculate_rsi(prices: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: pd.Series, fast: int = EMA_FAST_SPAN, slow: int = EMA_SLOW_SPAN) -> pd.Series:
    """Calculate MACD as percentage of price."""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    return (ema_fast - ema_slow) / prices * 100

def calculate_bollinger_bands(prices: pd.Series, window: int = BOLLINGER_WINDOW, num_std: float = BOLLINGER_STD_MULTIPLIER) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands and position. Returns (middle, upper, lower, position)."""
    middle = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    position = (prices - lower) / (upper - lower)
    return middle, upper, lower, position

def calculate_roc(prices: pd.Series, periods: int = ROC_PERIODS) -> pd.Series:
    """Calculate Rate of Change."""
    return prices.pct_change(periods=periods) * 100

def calculate_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Calculate rolling volatility (standard deviation of returns)."""
    return returns.rolling(window=window).std()

def calculate_price_to_sma_ratios(prices: pd.Series, sma: pd.Series) -> pd.Series:
    """Calculate price position relative to SMA as percentage."""
    return (prices - sma) / sma * 100

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
    if len(df) < MIN_DATA_LENGTH:
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
    df['SMA_10'] = calculate_sma(df['Close'], SMA_SHORT_WINDOW)
    df['SMA_50'] = calculate_sma(df['Close'], SMA_MEDIUM_WINDOW)
    df['SMA_200'] = calculate_sma(df['Close'], SMA_LONG_WINDOW)

    # Price position relative to SMAs (percentage)
    df['Price_to_SMA10'] = calculate_price_to_sma_ratios(df['Close'], df['SMA_10'])
    df['Price_to_SMA50'] = calculate_price_to_sma_ratios(df['Close'], df['SMA_50'])
    df['Price_to_SMA200'] = calculate_price_to_sma_ratios(df['Close'], df['SMA_200'])

    # Volatility (rolling std of returns)
    df['Volatility_20'] = calculate_volatility(df['Returns'], VOLATILITY_SHORT_WINDOW)
    df['Volatility_50'] = calculate_volatility(df['Returns'], VOLATILITY_LONG_WINDOW)

    # Technical indicators
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    df['ROC'] = calculate_roc(df['Close'])

    # Bollinger Bands position
    df['BB_middle'], df['BB_upper'], df['BB_lower'], df['BB_position'] = calculate_bollinger_bands(df['Close'])

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
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
    )

    # Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use Gradient Boosting with simpler parameters
    model = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        subsample=SUBSAMPLE,
        random_state=RANDOM_STATE,
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

    return model, scaler, mse, rmse, mae, r2, X_test.index, y_test, y_pred, pred_prices, actual_next_prices

# Helper functions for future feature calculation
def calculate_future_rsi(prices: list[float]) -> float:
    """Calculate RSI for a list of prices (used in future predictions)."""
    if len(prices) < 15:
        return 50.0

    price_changes = np.diff(prices[-15:])
    gains = price_changes[price_changes > 0]
    losses = -price_changes[price_changes < 0]
    avg_gain = gains.mean() if len(gains) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    if avg_loss != 0:
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))
    return 100.0

def calculate_future_ema(prices: list[float], span: int) -> float:
    """Calculate EMA for a list of prices using exponential weights."""
    if len(prices) < span:
        return prices[-1]

    weights = np.exp(np.linspace(-1., 0., span))
    weights /= weights.sum()
    return float(np.average(prices[-span:], weights=weights))

def calculate_lag_features(all_returns: list[float]) -> tuple[float, float, float]:
    """Calculate return lag features."""
    return_lag_1 = all_returns[-1] if len(all_returns) >= 1 else 0.0
    return_lag_2 = all_returns[-2] if len(all_returns) >= 2 else 0.0
    return_lag_5 = all_returns[-5] if len(all_returns) >= 5 else 0.0
    return return_lag_1, return_lag_2, return_lag_5

def calculate_sma_features(all_prices: list[float], current_price: float) -> tuple[float, float, float, float, float, float]:
    """Calculate SMA and price-to-SMA ratio features."""
    sma_10 = float(np.mean(all_prices[-SMA_SHORT_WINDOW:])) if len(all_prices) >= SMA_SHORT_WINDOW else current_price
    sma_50 = float(np.mean(all_prices[-SMA_MEDIUM_WINDOW:])) if len(all_prices) >= SMA_MEDIUM_WINDOW else current_price
    sma_200 = float(np.mean(all_prices[-SMA_LONG_WINDOW:])) if len(all_prices) >= SMA_LONG_WINDOW else current_price

    price_to_sma10 = float((current_price - sma_10) / sma_10 * 100) if sma_10 != 0 else 0.0
    price_to_sma50 = float((current_price - sma_50) / sma_50 * 100) if sma_50 != 0 else 0.0
    price_to_sma200 = float((current_price - sma_200) / sma_200 * 100) if sma_200 != 0 else 0.0

    return sma_10, sma_50, sma_200, price_to_sma10, price_to_sma50, price_to_sma200

def calculate_volatility_features(all_returns: list[float]) -> tuple[float, float]:
    """Calculate volatility features."""
    volatility_20 = float(np.std(all_returns[-VOLATILITY_SHORT_WINDOW:])) if len(all_returns) >= VOLATILITY_SHORT_WINDOW else 0.0
    volatility_50 = float(np.std(all_returns[-VOLATILITY_LONG_WINDOW:])) if len(all_returns) >= VOLATILITY_LONG_WINDOW else 0.0
    return volatility_20, volatility_50

def calculate_macd_feature(all_prices: list[float], current_price: float) -> float:
    """Calculate MACD feature."""
    ema_12 = calculate_future_ema(all_prices, EMA_FAST_SPAN)
    ema_26 = calculate_future_ema(all_prices, EMA_SLOW_SPAN)
    return (ema_12 - ema_26) / current_price * 100 if current_price != 0 else 0.0

def calculate_bollinger_feature(all_prices: list[float], current_price: float) -> float:
    """Calculate Bollinger Bands position feature."""
    if len(all_prices) < BOLLINGER_WINDOW:
        return 0.5

    bb_middle = float(np.mean(all_prices[-BOLLINGER_WINDOW:]))
    bb_std = float(np.std(all_prices[-BOLLINGER_WINDOW:]))
    bb_upper = bb_middle + (bb_std * BOLLINGER_STD_MULTIPLIER)
    bb_lower = bb_middle - (bb_std * BOLLINGER_STD_MULTIPLIER)

    if (bb_upper - bb_lower) != 0:
        return (current_price - bb_lower) / (bb_upper - bb_lower)
    return 0.5

def calculate_roc_feature(all_prices: list[float]) -> float:
    """Calculate Rate of Change feature."""
    if len(all_prices) >= ROC_PERIODS:
        return float((all_prices[-1] - all_prices[-ROC_PERIODS]) / all_prices[-ROC_PERIODS] * 100)
    return 0.0

def update_returns_list(recent_returns: list[float], predictions: list[float], recent_prices: list[float]) -> list[float]:
    """Update the returns list with predicted returns."""
    all_returns = recent_returns.copy()
    if len(predictions) > 0:
        for j in range(len(predictions)):
            if j == 0:
                ret = (predictions[0] - recent_prices[-1]) / recent_prices[-1]
            else:
                ret = (predictions[j] - predictions[j-1]) / predictions[j-1]
            all_returns.append(ret)
    return all_returns

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

    for _ in range(days):
        # Combine historical and predicted prices
        all_prices = recent_prices + predictions
        current_price = all_prices[-1]

        # Calculate returns for lag features
        all_returns = update_returns_list(recent_returns, predictions, recent_prices)

        # Calculate all features using helper functions
        return_lag_1, return_lag_2, return_lag_5 = calculate_lag_features(all_returns)
        _, _, _, price_to_sma10, price_to_sma50, price_to_sma200 = calculate_sma_features(all_prices, current_price)
        volatility_20, volatility_50 = calculate_volatility_features(all_returns)
        rsi = calculate_future_rsi(all_prices)
        macd = calculate_macd_feature(all_prices, current_price)
        bb_position = calculate_bollinger_feature(all_prices, current_price)
        roc = calculate_roc_feature(all_prices)

        # Create feature vector matching training order
        feat_vec = np.array([[return_lag_1, return_lag_2, return_lag_5,
                             price_to_sma10, price_to_sma50, price_to_sma200,
                             volatility_20, volatility_50, rsi, macd,
                             bb_position, roc]])

        # Scale features and predict
        feat_vec_scaled = scaler.transform(feat_vec)
        pred_return = float(model.predict(feat_vec_scaled)[0])

        # Apply reasonable bounds for daily returns
        pred_return = np.clip(pred_return, MIN_DAILY_RETURN, MAX_DAILY_RETURN)

        # Convert return to price
        pred_price = current_price * (1 + pred_return)

        # Store features and prediction
        features.append([return_lag_1, return_lag_2, return_lag_5,
                        price_to_sma10, price_to_sma50, price_to_sma200,
                        volatility_20, volatility_50, rsi, macd,
                        bb_position, roc])
        predictions.append(float(pred_price))

    # Create future dataframe
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

    model, scaler, mse, rmse, mae, r2, test_idx, y_test, y_pred, pred_prices, actual_next_prices = train_model(X, y, close_prices)
    st.subheader("Model Performance on Test Set")

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("RMSE", f"{rmse*100:.4f}%")
    with col3:
        st.metric("MAE", f"{mae*100:.4f}%")
    with col4:
        # Calculate directional accuracy (using returns, not prices)
        correct_direction = sum((y_pred > 0) == (y_test > 0))
        dir_accuracy = correct_direction / len(y_test) * 100
        st.metric("Direction Accuracy", f"{dir_accuracy:.1f}%")

    perf_df = pd.DataFrame(
        {'Actual': actual_next_prices, 'Predicted': pred_prices},
        index=test_idx
    )
    st.line_chart(perf_df)

    # Show explanation of metrics
    with st.expander("Understanding Model Metrics"):
        st.write("""
        - **R² Score**: Measures how well the model explains variance (1.0 is perfect, closer to 1 is better)
        - **RMSE**: Root Mean Squared Error - average prediction error in percentage returns
        - **MAE**: Mean Absolute Error - average absolute prediction error in percentage returns
        - **Direction Accuracy**: Percentage of times the model correctly predicted the direction of price movement (up vs down)

        Note: The model predicts next-day returns (percentage changes) which are then converted to prices.
        """)

    years = st.slider(
        "Select forecast horizon (years):", 1, 3, 1
    )
    future_days = years * TRADING_DAYS_PER_YEAR

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
        uncertainty = price * hist_volatility * np.sqrt(days_ahead) * CONFIDENCE_95_Z_SCORE
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
