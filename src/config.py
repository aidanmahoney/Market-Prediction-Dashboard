"""
Configuration constants for the Market Prediction Dashboard.
"""

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

# Feature columns
FEATURE_COLUMNS = [
    'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_5',
    'Price_to_SMA10', 'Price_to_SMA50', 'Price_to_SMA200',
    'Volatility_20', 'Volatility_50', 'RSI', 'MACD',
    'BB_position', 'ROC'
]
