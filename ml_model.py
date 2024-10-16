import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Create models directory if not present
os.makedirs('models', exist_ok=True)


# Function to download stock data using yfinance
def load_stock_data(stock_name):
    # Download stock data from Yahoo Finance (last 5 years)
    stock_data = yf.download(stock_name, period="5y")

    # Only keep the 'Close' price and normalize it for model training
    close_prices = stock_data[['Close']]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    return stock_data, scaled_data, scaler


# Function to prepare the data for LSTM model training
def prepare_data(data, window_size=60):
    x_train, y_train = [], []
    for i in range(window_size, len(data)):
        x_train.append(data[i - window_size:i, 0])
        y_train.append(data[i, 0])
    return np.array(x_train), np.array(y_train)


# Function to build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Function to train the model on the given stock data
def train_model(stock_name):
    stock_data, scaled_data, scaler = load_stock_data(stock_name)

    # Split into train and test sets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - 60:]

    # Prepare the training data
    x_train, y_train = prepare_data(train_data)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build and train the LSTM model
    model = build_lstm_model(x_train.shape)
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Save the model
    model.save(f'models/{stock_name}_lstm_model.h5')

    return model, scaler, stock_data


# Function to predict future stock prices
def predict_future(stock_name, num_days=60):
    # Check if the model already exists
    model_path = f'models/{stock_name}_lstm_model.h5'
    stock_data, scaled_data, scaler = load_stock_data(stock_name)

    if not os.path.exists(model_path):
        # If the model does not exist, train and save it
        model, scaler, _ = train_model(stock_name)
    else:
        # Load the pre-trained model if it exists
        model = build_lstm_model((scaled_data.shape[1], 60))
        model.load_weights(model_path)

    # Get the last 60 data points for prediction
    input_data = scaled_data[-60:]
    x_input = np.reshape(input_data, (1, 60, 1))
    predictions = []

    for _ in range(num_days):
        pred = model.predict(x_input)
        predictions.append(pred[0][0])
        # Append prediction correctly by reshaping `pred` to match the dimensions of `x_input`
        x_input = np.append(x_input[:, 1:, :], np.reshape(pred, (1, 1, 1)), axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions


# Function to get technical details using yfinance
def get_technical_details(stock_name):
    stock = yf.Ticker(stock_name)
    info = stock.info

    return {
        'Market Cap': info.get('marketCap', 'N/A'),
        'ROE': info.get('returnOnEquity', 'N/A'),
        'P/E Ratio': info.get('trailingPE', 'N/A'),
        'P/B Ratio': info.get('priceToBook', 'N/A'),
        'Industry P/E': info.get('industryPE', 'N/A'),
        'Debt to Equity': info.get('debtToEquity', 'N/A'),
        'EPS': info.get('trailingEps', 'N/A'),
        'Div Yield': info.get('dividendYield', 'N/A'),
        'Book Value': info.get('bookValue', 'N/A'),
        'Face Value': info.get('faceValue', 'N/A')
    }


# Function to perform sentiment analysis on stock-related news
def perform_sentiment_analysis(stock_name):
    # Dummy news headlines (replace with actual headlines if available)
    news_headlines = [
        f"{stock_name} sees strong uptrend in recent market activities.",
        f"Concerns grow over {stock_name}'s recent earnings performance.",
        f"Experts believe {stock_name} has strong growth potential."
    ]

    analyzer = SentimentIntensityAnalyzer()

    sentiments = []
    for headline in news_headlines:
        sentiment = analyzer.polarity_scores(headline)
        sentiments.append({
            'headline': headline,
            'sentiment': sentiment
        })

    return sentiments


# Function to get competitor stocks (based on industry)
def get_competitor_stocks(stock_name):
    stock = yf.Ticker(stock_name)
    info = stock.info

    # For now, this is a dummy competitor list based on a predefined industry (this should be dynamic in production)
    industry = info.get('industry', 'N/A')
    competitor_stocks = {
        'Technology': ['MSFT', 'GOOGL', 'AMZN'],
        'Financial': ['JPM', 'BAC', 'WFC'],
        'Healthcare': ['PFE', 'MRK', 'JNJ'],
        'Energy': ['XOM', 'CVX', 'COP']
    }

    # Get the competitors from the same industry
    return competitor_stocks.get(industry, [])

