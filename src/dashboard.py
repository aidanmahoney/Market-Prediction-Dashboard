"""
Main dashboard UI for the Market Prediction application.
"""
import streamlit as st
import pandas as pd
import numpy as np

from .config import TRADING_DAYS_PER_YEAR, CONFIDENCE_95_Z_SCORE
from .data_loader import load_data
from .feature_engineering import create_features, create_future_features
from .model import train_model


def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Market Predictor",
        layout="wide"
    )


def display_metrics(rmse: float, mae: float, r2: float, y_test: pd.Series, y_pred: np.ndarray):
    """
    Display model performance metrics.

    Args:
        rmse: Root mean squared error
        mae: Mean absolute error
        r2: R-squared score
        y_test: Actual test returns
        y_pred: Predicted test returns
    """
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


def display_metrics_explanation():
    """Display explanation of model metrics."""
    with st.expander("Understanding Model Metrics"):
        st.write("""
        - **R² Score**: Measures how well the model explains variance (1.0 is perfect, closer to 1 is better)
        - **RMSE**: Root Mean Squared Error - average prediction error in percentage returns
        - **MAE**: Mean Absolute Error - average absolute prediction error in percentage returns
        - **Direction Accuracy**: Percentage of times the model correctly predicted the direction of price movement (up vs down)

        Note: The model predicts next-day returns (percentage changes) which are then converted to prices.
        """)


def calculate_confidence_intervals(
    future_prices: list[float],
    hist_volatility: float
) -> tuple[list[float], list[float]]:
    """
    Calculate 95% confidence intervals for predictions.

    Args:
        future_prices: List of predicted prices
        hist_volatility: Historical volatility

    Returns:
        Tuple of (lower bounds, upper bounds)
    """
    confidence_lower = []
    confidence_upper = []

    for i, price in enumerate(future_prices):
        # Uncertainty grows with prediction horizon
        days_ahead = i + 1
        uncertainty = price * hist_volatility * np.sqrt(days_ahead) * CONFIDENCE_95_Z_SCORE
        confidence_lower.append(price - uncertainty)
        confidence_upper.append(price + uncertainty)

    return confidence_lower, confidence_upper


def display_forecast_info():
    """Display information about the forecast."""
    st.info(
        "**Understanding the forecast:**\n\n"
        "- The model predicts next-day returns using technical indicators (RSI, MACD, price/SMA ratios, volatility)\n"
        "- Returns are converted to prices for visualization\n"
        "- Daily return predictions are capped at ±5% for stability\n"
        "- Confidence intervals widen over time, showing increased uncertainty\n"
        "- Longer-term predictions should be taken with caution\n\n"
        "**Note:** This is a simplified model for educational purposes. Real trading requires much more sophisticated analysis."
    )


def main() -> None:
    """Main dashboard application."""
    setup_page_config()
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
    display_metrics(rmse, mae, r2, y_test, y_pred)

    perf_df = pd.DataFrame(
        {'Actual': actual_next_prices, 'Predicted': pred_prices},
        index=test_idx
    )
    st.line_chart(perf_df)

    display_metrics_explanation()

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
    confidence_lower, confidence_upper = calculate_confidence_intervals(future_prices, hist_volatility)

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

    display_forecast_info()


if __name__ == "__main__":
    main()
