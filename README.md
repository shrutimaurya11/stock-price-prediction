# 📈 Stock Price Prediction using LSTM

This project predicts the next-day closing price of Tesla (TSLA) stock using an LSTM deep learning model. It uses historical stock data, technical indicators, and a Streamlit frontend for interaction.

## 🚀 Features

- Predicts TSLA stock closing price
- Uses technical indicators: MA10, MA50, EMA10, Bollinger Bands
- Built using LSTM model (TensorFlow)
- Interactive UI using Streamlit
- Real-time data from Yahoo Finance (yfinance)

## 🗂️ Project Structure

📦 stock-price-prediction
├── app.py # Main Streamlit app
├── model.h5 # Trained LSTM model
├── TSLA.csv # Raw TSLA stock data
├── TSLA_complete.csv # Processed data with indicators
├── yfinance_cache.sqlite # Cached data for faster load
├── requirements.txt # All Python dependencies
└── README.md # Project documentation

## 🛠️ Setup Instructions

1. **Clone the repo**  
   ```bash
   git clone https://github.com/shrutimaurya11/stock-price-prediction.git
   cd stock-price-prediction
pip install -r requirements.txt
streamlit run app.py
🌐 Live Demo
Check it out here 👉 Hugging Face Space

🙋‍♀️ Author
Shruti Maurya
🔗 GitHub

Made with ❤️ using Python, TensorFlow

---

Let me know if you also want to include Hugging Face deployment steps, model training steps, or add badges like `Made with Python`, `Live on Hugging Face`, etc.









