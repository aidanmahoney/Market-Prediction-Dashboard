"""
Data loading functionality for fetching historical stock data.
"""
import yfinance as yf
import pandas as pd
import datetime as dt
import streamlit as st


@st.cache_data
def load_data(ticker: str) -> pd.DataFrame:
    """
    Load historical stock data for a given ticker.

    Args:
        ticker: Stock symbol (e.g., 'SPY', 'AAPL')

    Returns:
        DataFrame with historical stock data

    Raises:
        ValueError: If no data is available for the ticker
    """
    end_date = dt.datetime.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start="2010-01-01", end=end_date)

    if data.empty:
        raise ValueError(f"No data available for {ticker}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data
