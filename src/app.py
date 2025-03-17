from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt

st.set_page_config(page_title="Market Predictor", page_icon=":chart_increasing:", layout="wide")

def create_future_features(df, model, last_close, days=30):
    future_features = []
    future_prices = []
    
    last_momentum = float(df['Momentum'].iloc[-1]) if 'Momentum' in df.columns else 0.0
    historical_growth = df['Close'].pct_change().mean()
    damping_factor = 0.05

    for i in range(days):
        if future_features:
            last_features = np.array(future_features[-1], dtype=np.float64).reshape(1, -1)
        else:
            if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
                last_features = df[['SMA_50', 'SMA_200']].iloc[-1:].copy()
                last_features['Momentum'] = last_momentum 
                last_features = last_features.to_numpy(dtype=np.float64).reshape(1, -1)
            else:
                raise ValueError(f"SMA_50 and SMA_200 missing from dataset. Available columns: {df.columns.tolist()}")

        predicted_price = model.predict(last_features)
        if isinstance(predicted_price, (list, np.ndarray)):
            predicted_price = float(predicted_price[0])

        if isinstance(last_close, (list, np.ndarray)):
            last_close = float(last_close[0])

        if i == 0:
            predicted_price = last_close
        else:
            predicted_price += predicted_price * (historical_growth if historical_growth > 0 else 0.0005)

        future_sma_50 = np.mean([predicted_price] + [float(x[0]) for x in future_features[-49:]]) if len(future_features) >= 49 else predicted_price
        future_sma_200 = np.mean([predicted_price] + [float(x[1]) for x in future_features[-199:]]) if len(future_features) >= 199 else predicted_price

        momentum_decay = max(0.01, 1 - (i / (days * 3))) 
        future_momentum = float(predicted_price - last_close) * (0.02 * momentum_decay)
        future_momentum = np.clip(future_momentum, -1, 1) 

        future_features.append([future_sma_50, future_sma_200, future_momentum])
        future_prices.append(predicted_price)

        last_close = predicted_price 

    future_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days)
    future_df = pd.DataFrame(future_features, columns=['SMA_50', 'SMA_200', 'Momentum'], index=future_index)

    return future_df, future_prices

def load_data(ticker):
    try:
        end_date = dt.datetime.today().strftime('%Y-%m-%d')
        data = yf.download(ticker, start="2010-01-01", end=end_date)

        if data.empty:
            raise ValueError(f"No data available for {ticker}. Please try another ticker.")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def create_features(df):
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Close']].copy()

    if len(df) < 200:
        raise ValueError("Not enough data to compute SMA_50 and SMA_200. Please try another stock.")

    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['SMA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
    df['Momentum'] = df['Close'].diff(periods=5)

    if 'SMA_50' not in df.columns or 'SMA_200' not in df.columns:
        raise ValueError("SMA_50 or SMA_200 not created properly.")

    df.dropna(subset=['SMA_50', 'SMA_200'], inplace=True)
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("After removing NaN values, no data remains.")

    X = df[['SMA_50', 'SMA_200', 'Momentum']]
    y = df['Close']

    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor()
    
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

        processed_data = data.copy()
        X, y = create_features(processed_data)
        processed_data['SMA_50'] = X['SMA_50']
        processed_data['SMA_200'] = X['SMA_200']

        if not X.empty:
            model, mse, test_dates, y_pred, y_test = train_model(X, y)

            st.subheader('Model Performance')
            st.write(f'Mean Squared Error: {mse}')

            results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=test_dates)
            st.line_chart(results_df)

            years_to_predict = st.slider("Select number of years for prediction:", 1, 3, 1)

            future_days = years_to_predict * 252

            initial_future_predictions = create_future_features(
                processed_data, model, last_close=processed_data['Close'].iloc[-1], days=50
            )

            future_features, future_predictions = create_future_features(
                processed_data, model, last_close=initial_future_predictions[-1], days=future_days
            )

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
