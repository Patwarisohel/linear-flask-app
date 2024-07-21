import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# Fetch historical stock data
def fetch_data(ticker, period='1y'):
    stock_data = yf.download(ticker, period=period)
    stock_data = stock_data[['Close']].reset_index()
    return stock_data

# Create features for the model
def create_features(data, window=30):
    data['Target'] = data['Close'].shift(-window)
    for i in range(1, window+1):
        data[f'Close_lag_{i}'] = data['Close'].shift(i)
    data = data.dropna()
    return data

# Fetch data and create features
ticker = 'AAPL'
data = fetch_data(ticker)
data = create_features(data)

# Split data into training and testing sets
X = data.drop(columns=['Date', 'Target', 'Close'])
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
print(predictions)

from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([data[f'Close_lag_{i}'] for i in range(1, 31)]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify(prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/automate', methods=['GET'])
def automate():
    # Fetch the latest data
    data = fetch_data(ticker, period='1y')
    data = create_features(data)

    # Use the last row of data for prediction
    features = data.drop(columns=['Date', 'Target', 'Close']).iloc[-1].values.reshape(1, -1)
    prediction = model.predict(features)

    return jsonify(prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

