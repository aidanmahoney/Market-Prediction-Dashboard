# Market Prediction Dashboard

A market predictor for day traders using machine learning.

## Overview

This is a stock market prediction tool that uses Scikit-learn to train a machine learning model with Gradient Boosting Regression. It fetches historical stock price data from Yahoo Finance and trains the model to make future stock price predictions based on technical indicators.

## Features

- Real-time stock data fetching from Yahoo Finance
- Technical indicator calculations (RSI, MACD, Bollinger Bands, etc.)
- Machine learning model using Gradient Boosting
- Interactive Streamlit dashboard
- Model performance metrics (R², RMSE, MAE, Direction Accuracy)
- Future price predictions with confidence intervals
- Configurable forecast horizons (1-3 years)

## Project Structure

```
Market-Prediction-Dashboard/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── app.py                   # Main entry point
│   ├── dashboard.py             # Streamlit UI and main application
│   ├── config.py                # Configuration constants
│   ├── data_loader.py           # Data fetching from Yahoo Finance
│   ├── feature_engineering.py   # Feature creation and processing
│   ├── model.py                 # Model training and evaluation
│   └── technical_indicators.py  # Technical indicator calculations
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit dashboard:
```bash
streamlit run src/app.py
```

Or from the src directory:
```bash
cd src
streamlit run app.py
```

## Requirements

- Python 3.x
- streamlit
- yfinance
- pandas
- scikit-learn
- numpy

See `requirements.txt` for specific versions.
## Credits
- Aidan Mahoney
- Yahoo Finance
