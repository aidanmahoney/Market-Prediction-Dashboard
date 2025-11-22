"""
Feature engineering for market prediction model.
"""
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from .config import (
    MIN_DATA_LENGTH, SMA_SHORT_WINDOW, SMA_MEDIUM_WINDOW, SMA_LONG_WINDOW,
    VOLATILITY_SHORT_WINDOW, VOLATILITY_LONG_WINDOW, FEATURE_COLUMNS,
    EMA_FAST_SPAN, EMA_SLOW_SPAN, BOLLINGER_WINDOW, BOLLINGER_STD_MULTIPLIER,
    ROC_PERIODS, MAX_DAILY_RETURN, MIN_DAILY_RETURN
)
from .technical_indicators import (
    calculate_sma, calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_roc, calculate_volatility, calculate_price_to_sma_ratios,
    calculate_future_rsi, calculate_future_ema
)


@st.cache_data
def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Create features from historical stock data.

    Args:
        df: DataFrame with historical stock data

    Returns:
        Tuple of (X features, y target, close prices)

    Raises:
        ValueError: If not enough data to compute indicators
    """
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

    X = df[FEATURE_COLUMNS]
    y = df['Target_Return']
    close_prices = df['Close']
    return X, y, close_prices


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


def create_future_features(
    df: pd.DataFrame,
    model: GradientBoostingRegressor,
    scaler: StandardScaler,
    last_close: float,
    days: int = 30
) -> tuple[pd.DataFrame, list[float]]:
    """
    Generate future feature matrix and predictions.

    Args:
        df: DataFrame with Close and Returns columns
        model: Trained prediction model
        scaler: Fitted feature scaler
        last_close: Last known closing price
        days: Number of days to predict

    Returns:
        Tuple of (future features DataFrame, predicted prices list)
    """
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
        columns=FEATURE_COLUMNS,
        index=idx
    )
    return future_df, predictions
