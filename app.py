import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load your trained model
model = load_model('your_lstm_model.h5')  # Replace with your actual model file name

st.title("📊 Stock Price Prediction with LSTM")

# Input: Stock symbol
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")

if st.button("Predict"):
    df = yf.download(stock_symbol, start="2015-01-01", end="2024-12-31")
    df = df[['Close']].dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    seq_len = 60
    X = []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    pred = model.predict(X)
    pred = scaler.inverse_transform(pred)

    st.subheader("📈 Actual vs Predicted Prices")
    fig, ax = plt.subplots()
    ax.plot(df.index[seq_len:], df['Close'][seq_len:], label='Actual')
    ax.plot(df.index[seq_len:], pred, label='Predicted')
    ax.legend()
    st.pyplot(fig)
