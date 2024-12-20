from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt

st.set_page_config(page_title="Market Predictor", page_icon=":chart_increasing:", layout="wide")

def create_future_features(df, last_close, days=30, growth_rate=0.07):
    future_features = []

    for i in range(days):
        year_adjustment = (1 + growth_rate) ** (i // 252)

        adjusted_close = last_close * year_adjustment

        future_sma_50 = np.nan
        future_sma_200 = np.nan

        previous_sma_50 = [adjusted_close] + [x[0] for x in future_features[-49:]]
        previous_sma_200 = [adjusted_close] + [x[1] for x in future_features[-199:]]

        if len(previous_sma_50) > 0:
            future_sma_50 = np.nanmean(previous_sma_50)

        if len(previous_sma_200) > 0:
            future_sma_200 = np.nanmean(previous_sma_200)

        future_features.append([future_sma_50, future_sma_200])

        last_close = adjusted_close

    return pd.DataFrame(future_features, columns=['SMA_50', 'SMA_200'],
                        index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days))

def load_data(ticker):
    try:
        end_date = dt.datetime.today().strftime('%Y-%m-%d')

        data = yf.download(ticker, start="2010-01-01", end=end_date)
        if data.empty:
            raise ValueError("No data available for this ticker.")
        return data
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def create_features(df):
    df = df[['Close']].copy()

    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    df = df.dropna(subset=['SMA_50', 'SMA_200'])

    X = df[['SMA_50', 'SMA_200']]
    y = df['Close']

    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse, X_test.index, y_pred, y_test

def main():
    st.title("Market Recovery Prediction Dashboard")

    ticker = st.text_input("Enter stock symbol (e.g., SPY for S&P 500):", value="SPY")

    data_load_state = st.text('Loading data...')
    data = load_data(ticker)
    data_load_state.text('Loading data... done!')

    if data is not None:
        st.subheader(f'{ticker} Historical Data')
        st.line_chart(data['Close'])

        X, y = create_features(data)

        if not X.empty:
            model, mse, test_dates, y_pred, y_test = train_model(X, y)

            st.subheader('Model Performance')
            st.write(f'Mean Squared Error: {mse}')

            results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=test_dates)
            st.line_chart(results_df)

            years_to_predict = st.slider("Select number of years for prediction:", 1, 3, 1)

            future_days = years_to_predict * 252
            initial_future_features = create_future_features(data, last_close=data['Close'].iloc[-1], days=50, growth_rate=0.07)
            initial_future_predictions = model.predict(initial_future_features)

            future_features = create_future_features(data, last_close=initial_future_predictions[-1], days=future_days, growth_rate=0.07)
            future_predictions = model.predict(future_features)

            future_dates = future_features.index
            future_results_df = pd.DataFrame({'Predicted': future_predictions}, index=future_dates)

            last_historical_price = data['Close'].iloc[-1]
            future_results_df = pd.DataFrame({'Predicted': np.concatenate(([last_historical_price], future_predictions))}, 
                                            index=pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days + 1))

            combined_results_df = pd.concat([data[['Close']], future_results_df], axis=1)
            combined_results_df.columns = ['Actual', 'Predicted']

            st.subheader('Combined Historical and Future Predictions')
            st.line_chart(combined_results_df)

        else:
            st.write("Not enough data to make predictions. Try another stock or adjust date range.")
    else:
        st.write("Failed to load data. Please check the ticker symbol or try again later.")

if __name__ == "__main__":
    main()
