"""
Technical indicator calculation functions.
"""
import pandas as pd
import numpy as np
from .config import (
    RSI_WINDOW, EMA_FAST_SPAN, EMA_SLOW_SPAN,
    BOLLINGER_WINDOW, BOLLINGER_STD_MULTIPLIER, ROC_PERIODS
)


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


def calculate_bollinger_bands(
    prices: pd.Series,
    window: int = BOLLINGER_WINDOW,
    num_std: float = BOLLINGER_STD_MULTIPLIER
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
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
