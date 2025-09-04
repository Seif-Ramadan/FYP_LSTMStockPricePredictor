import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

@st.cache_data
def fetch_data(symbol: str, start_date='2010-01-01', end_date=None):
    """
    Fetch historical stock data from Yahoo Finance.
    Caches the result to avoid re-downloading on each run.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
    if data.empty:
        raise ValueError(f"No data fetched for symbol: {symbol}")
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data = data.dropna()
    
    # Add technical indicators
    data = add_technical_indicators(data)
    
    return data

def add_technical_indicators(data):
    """
    Add technical indicators to the dataframe
    """
    df = data.copy()
    
    # Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Volatility
    df['Volatility5'] = df['Close'].rolling(window=5).std()
    df['Volatility20'] = df['Close'].rolling(window=20).std()
    
    # Price changes
    df['PriceChange1d'] = df['Close'].pct_change()
    df['PriceChange5d'] = df['Close'].pct_change(periods=5)
    df['PriceChange20d'] = df['Close'].pct_change(periods=20)
    
    # Volume indicators
    df['VolumeSMA5'] = df['Volume'].rolling(window=5).mean()
    df['VolumeChange'] = df['Volume'].pct_change()
    
    # Price Momentum
    df['Momentum5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum10'] = df['Close'] - df['Close'].shift(10)
    
    # RSI (Relative Strength Index)
    # Calculate daily price changes
    delta = df['Close'].diff()
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss over 14 days
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    # Calculate 12-day and 26-day EMA
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    
    # Calculate MACD line
    df['MACD'] = ema12 - ema26
    
    # Calculate signal line (9-day EMA of MACD)
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate MACD histogram
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    middle_band = df['Close'].rolling(window=20).mean()
    std_dev = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = middle_band + (std_dev * 2)
    df['BB_Middle'] = middle_band
    df['BB_Lower'] = middle_band - (std_dev * 2)
    
    # Drop missing values
    df = df.dropna()
    
    return df

def preprocess_daily_data(data: pd.DataFrame, use_features=None, sequence_length=90):
    """
    Prepare data for LSTM.
    """
    # Select the columns to use
    if use_features is None:
        # Default to use all features
        columns_to_use = data.columns
    else:
        # Use only specified features
        columns_to_use = use_features
        
    selected_data = data[columns_to_use]
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(selected_data)
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        # We want to predict the 'Close' price
        close_idx = list(columns_to_use).index('Close') if 'Close' in columns_to_use else 0
        y.append(scaled_data[i + sequence_length, close_idx])
        
    X, y = np.array(X), np.array(y)
    return X, y, scaler, columns_to_use

def create_sequences(data: pd.DataFrame, scaler, columns_to_use, sequence_length=90):
    """
    Create sequences from data using an already-fitted scaler.
    """
    # Select the features from the test data
    selected_data = data[columns_to_use]
    # Transform the data using the scaler fitted on training data
    scaled_data = scaler.transform(selected_data)
    
    X, y = [], []
    # Determine the index for the 'Close' price
    close_idx = list(columns_to_use).index('Close') if 'Close' in columns_to_use else 0

    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length, close_idx])
        
    return np.array(X), np.array(y)
