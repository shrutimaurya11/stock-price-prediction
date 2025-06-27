# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import gradio as gr
import yfinance as yf
import requests
from datetime import datetime
import time
import cachetools

# Configure caching
cache = cachetools.TTLCache(maxsize=100, ttl=3600)  # 1 hour cache

# Alternative data source when yfinance fails
def get_alternative_data(ticker):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"interval": "1d", "range": "5y"}
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        # Handle non-JSON response
        if response.text.strip().startswith('<'):
            raise ValueError("Non-JSON response (HTML or error)")
        
        try:
            data = response.json()
        except Exception as e:
            raise ValueError(f"JSON decode failed: {e}")
        
        if not data or 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
            raise ValueError("Invalid data structure in API response")
        
        result = data['chart']['result'][0]
        timestamps = result.get('timestamp')
        closes = result['indicators']['quote'][0]['close']
        
        if not timestamps or not closes or len(timestamps) != len(closes):
            raise ValueError("Mismatched or missing timestamps and closes")
        
        dates = [datetime.fromtimestamp(ts) for ts in timestamps]
        return pd.DataFrame({'Close': closes}, index=dates)
    
    except Exception as e:
        print(f"Alternative data fetch failed: {str(e)}")
        return pd.DataFrame()

# Robust data fetch with fallbacks
def robust_data_fetch(ticker):
    cache_key = f"data_{ticker}"
    if cache_key in cache:
        return cache[cache_key]
    
    # Try yfinance first
    try:
        data = yf.download(ticker, period="5y", progress=False)
        if not data.empty:
            cache[cache_key] = data[['Close']]
            return data[['Close']]
    except Exception as e:
        print(f"yfinance download failed: {str(e)}")
    
    # Try alternative method
    try:
        data = get_alternative_data(ticker)
        if not data.empty:
            cache[cache_key] = data
            return data
    except Exception as e:
        print(f"Alternative fetch failed: {str(e)}")
    
    # Final fallback
    try:
        time.sleep(2)
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="5y")
        if not data.empty:
            print("✅ Final fallback succeeded with yf.Ticker().history()")
            cache[cache_key] = data[['Close']]
            return data[['Close']]
    except Exception as e:
        print(f"Final fallback failed: {str(e)}")
    
    return pd.DataFrame()

# Ticker validation
def validate_ticker(ticker):
    if not ticker or not isinstance(ticker, str):
        return False
    try:
        data = robust_data_fetch(ticker)
        return not data.empty
    except:
        return False

# Prediction function
def predict_stock(ticker, lookback_days=60, epochs=50):
    try:
        if not ticker.strip():
            return "Error: Please enter a valid ticker symbol", None
        
        ticker = ticker.strip().upper()
        
        if not validate_ticker(ticker):
            return f"Error: Could not fetch data for '{ticker}'. Please check the symbol.", None
        
        data = robust_data_fetch(ticker)
        if data.empty:
            return f"Error: No historical data found for {ticker}", None
        if len(data) < lookback_days:
            return f"Error: Need at least {lookback_days} days of data (only have {len(data)})", None
        
        cache_key = f"model_{ticker}_{lookback_days}_{epochs}"
        if cache_key not in cache:
            # Preprocessing
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            X, y = [], []
            for i in range(lookback_days, len(scaled_data)):
                X.append(scaled_data[i-lookback_days:i])
                y.append(scaled_data[i])
            X, y = np.array(X), np.array(y)
            
            split = int(0.8 * len(X))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(100, return_sequences=True, input_shape=(lookback_days, 1)),
                Dropout(0.3),
                LSTM(50),
                Dropout(0.3),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=32, verbose=0)
            cache[cache_key] = (model, scaler)
        else:
            model, scaler = cache[cache_key]
        
        # Predict
        scaled_data = scaler.transform(data)
        last_sequence = scaled_data[-lookback_days:]
        predicted_price = scaler.inverse_transform(model.predict(last_sequence.reshape(1, lookback_days, 1)))
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data['Close'], label='Historical Price')
        ax.axvline(x=data.index[-1], color='r', linestyle='--', label='Prediction Point')
        ax.scatter(data.index[-1], predicted_price[0][0], color='g', s=100, label='Predicted Price')
        ax.set_title(f'{ticker} Price Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True)
        plt.close('all')
        
        last_price = data['Close'].iloc[-1]
        percent_change = ((predicted_price[0][0] - last_price) / last_price) * 100
        
        result_text = (
            f"🔍 Prediction for {ticker}\n"
            f"📅 Last closing price: ${last_price:.2f}\n"
            f"🔮 Next predicted price: ${predicted_price[0][0]:.2f}\n"
            f"📈 Predicted change: {percent_change:.2f}%\n\n"
            f"Model trained on {len(data)} days of historical data"
        )
        
        return result_text, fig
        
    except Exception as e:
        return f"⚠️ Error processing {ticker}: {str(e)}", None

# Gradio app
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 📈 AI Stock & Crypto Price Predictor
    *Predict next-day closing prices using LSTM neural networks*
    """)
    
    with gr.Row():
        with gr.Column():
            ticker_input = gr.Textbox(label="Enter Stock/Crypto Symbol", placeholder="e.g. AAPL, TSLA, BTC-USD", value="AAPL")
            with gr.Accordion("Advanced Options", open=False):
                lookback = gr.Slider(30, 120, value=60, label="Lookback Period (days)")
                epochs = gr.Slider(10, 100, value=50, step=10, label="Training Epochs")
            submit_btn = gr.Button("Generate Prediction", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="Prediction Results")
            output_plot = gr.Plot(label="Price History & Prediction")
    
    examples = gr.Examples(
        examples=[["AAPL"], ["MSFT"], ["BTC-USD"], ["NVDA"]],
        inputs=[ticker_input],
        label="Try these verified examples"
    )
    
    submit_btn.click(fn=predict_stock, inputs=[ticker_input, lookback, epochs], outputs=[output_text, output_plot])

if __name__ == "__main__":
    app.launch(share=True)



