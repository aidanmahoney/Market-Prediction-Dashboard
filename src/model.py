"""
Model training and prediction functionality.
"""
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from .config import (
    TEST_SIZE, RANDOM_STATE, N_ESTIMATORS, LEARNING_RATE,
    MAX_DEPTH, MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF, SUBSAMPLE
)


@st.cache_resource
def train_model(X: pd.DataFrame, y: pd.Series, close_prices: pd.Series) -> tuple:
    """
    Train the prediction model and evaluate performance.

    Args:
        X: Feature matrix
        y: Target variable (returns)
        close_prices: Historical closing prices

    Returns:
        Tuple containing:
        - model: Trained model
        - scaler: Fitted feature scaler
        - mse: Mean squared error
        - rmse: Root mean squared error
        - mae: Mean absolute error
        - r2: R-squared score
        - test_idx: Test set indices
        - y_test: Actual test returns
        - y_pred: Predicted test returns
        - pred_prices: Predicted prices
        - actual_next_prices: Actual next day prices
    """
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
