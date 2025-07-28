import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Try to import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("⚠️ TensorFlow not installed. LSTM model will not be available. Install with: pip install tensorflow")

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.warning("⚠️ Statsmodels not installed. ARIMA model will not be available. Install with: pip install statsmodels")

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Crypto Analysis & Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4037 0%, #99f2c8 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CryptoAnalyzer:
    def __init__(self):
        self.data = None
        self.predictions = None
        
    def fetch_crypto_data(self, symbol, timeframe='1d', days=365):
        """Fetch cryptocurrency data from CoinGecko API (free)"""
        try:
            # Convert timeframe to CoinGecko format
            timeframe_map = {'1d': 'daily', '4h': 'hourly', '1w': 'daily'}
            
            # CoinGecko API endpoint
            url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': timeframe_map.get(timeframe, 'daily')
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'prices' in data:
                # Convert to DataFrame
                prices = data['prices']
                volumes = data['total_volumes']
                
                df = pd.DataFrame(prices, columns=['timestamp', 'close'])
                df['volume'] = [v[1] for v in volumes]
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Generate OHLC from close prices (approximation for free API)
                df['open'] = df['close'].shift(1)
                df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.02, len(df)))
                df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.02, len(df)))
                
                # Fill NaN values
                df.fillna(method='ffill', inplace=True)
                df.dropna(inplace=True)
                
                return df
            else:
                st.error("Error fetching data from CoinGecko API")
                return None
                
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # Moving Averages
        df['MA7'] = ta.trend.sma_indicator(df['close'], window=7)
        df['MA25'] = ta.trend.sma_indicator(df['close'], window=25)
        df['MA99'] = ta.trend.sma_indicator(df['close'], window=99)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_lower'] = bollinger.bollinger_lband()
        df['BB_middle'] = bollinger.bollinger_mavg()
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['close'])
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_histogram'] = macd.macd_diff()
        
        return df
    
    def generate_signals(self, df):
        """Generate buy/sell signals"""
        signals = []
        
        for i in range(1, len(df)):
            signal_type = None
            signal_reason = ""
            
            # MA Crossover signals
            if df['MA7'].iloc[i] > df['MA25'].iloc[i] and df['MA7'].iloc[i-1] <= df['MA25'].iloc[i-1]:
                signal_type = 'BUY'
                signal_reason = 'MA7 crosses above MA25'
            elif df['MA7'].iloc[i] < df['MA25'].iloc[i] and df['MA7'].iloc[i-1] >= df['MA25'].iloc[i-1]:
                signal_type = 'SELL'
                signal_reason = 'MA7 crosses below MA25'
            
            # RSI signals
            elif df['RSI'].iloc[i] < 30 and df['RSI'].iloc[i-1] >= 30:
                signal_type = 'BUY'
                signal_reason = 'RSI oversold (<30)'
            elif df['RSI'].iloc[i] > 70 and df['RSI'].iloc[i-1] <= 70:
                signal_type = 'SELL'
                signal_reason = 'RSI overbought (>70)'
            
            # MACD signals
            elif df['MACD'].iloc[i] > df['MACD_signal'].iloc[i] and df['MACD'].iloc[i-1] <= df['MACD_signal'].iloc[i-1]:
                signal_type = 'BUY'
                signal_reason = 'MACD bullish crossover'
            elif df['MACD'].iloc[i] < df['MACD_signal'].iloc[i] and df['MACD'].iloc[i-1] >= df['MACD_signal'].iloc[i-1]:
                signal_type = 'SELL'
                signal_reason = 'MACD bearish crossover'
            
            # Bollinger Band signals
            elif df['close'].iloc[i] < df['BB_lower'].iloc[i]:
                signal_type = 'BUY'
                signal_reason = 'Price touches lower Bollinger Band'
            elif df['close'].iloc[i] > df['BB_upper'].iloc[i]:
                signal_type = 'SELL'
                signal_reason = 'Price touches upper Bollinger Band'
            
            if signal_type:
                signals.append({
                    'timestamp': df.index[i],
                    'price': df['close'].iloc[i],
                    'signal': signal_type,
                    'reason': signal_reason
                })
        
        return pd.DataFrame(signals)
    
    def prepare_ml_data(self, df, target_col='close', lookback=60):
        """Prepare data for machine learning models"""
        data = df[target_col].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y), scaler
    
    def train_lstm_model(self, X, y):
        """Train LSTM model for price prediction"""
        if not TENSORFLOW_AVAILABLE:
            st.error("TensorFlow is required for LSTM model. Please install it with: pip install tensorflow")
            return None, None, None
            
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Reshape for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)
        
        return model, X_test, y_test
    
    def predict_prices(self, df, model_type='LSTM', target='close', forecast_days=30):
        """Generate price predictions"""
        try:
            if model_type == 'Linear Regression':
                # Clean data for Linear Regression
                # Drop rows where any of the feature columns or target column have NaN values
                # This is crucial because TA indicators introduce NaNs at the beginning of the series
                required_cols = ['MA7', 'MA25', 'RSI', 'MACD', 'volume', target]
                df_clean = df.dropna(subset=required_cols)
                
                if df_clean.empty:
                    st.error("Not enough data after cleaning for Linear Regression. Try selecting a longer timeframe.")
                    return None
                    
                X = df_clean[['MA7', 'MA25', 'RSI', 'MACD', 'volume']].values
                y = df_clean[target].values
                
                # Train model
                model = LinearRegression()
                train_size = int(0.8 * len(X))
                
                if train_size == 0 or train_size >= len(X):
                    st.error("Not enough data to train Linear Regression model after splitting. Increase data duration or change model.")
                    return None
                    
                X_train, y_train = X[:train_size], y[:train_size]
                model.fit(X_train, y_train)
                
                # Predict future
                # Ensure last_features also doesn't contain NaN by taking from df_clean
                last_features_df = df_clean[['MA7', 'MA25', 'RSI', 'MACD', 'volume']].iloc[-1:].copy()
                last_features = last_features_df.values.repeat(forecast_days, axis=0)
                
                predictions = model.predict(last_features)
                
            elif model_type == 'LSTM':
                if not TENSORFLOW_AVAILABLE:
                    st.error("TensorFlow is required for LSTM model. Please install it with: pip install tensorflow")
                    return None
                    
                X, y, scaler = self.prepare_ml_data(df, target)
                model, X_test, y_test = self.train_lstm_model(X, y)
                
                if model is None:
                    return None
                
                # Predict future
                last_sequence = X[-1].reshape(1, -1, 1)
                predictions = []
                
                for _ in range(forecast_days):
                    pred = model.predict(last_sequence, verbose=0)
                    predictions.append(pred[0, 0])
                    # Update sequence for next prediction
                    last_sequence = np.roll(last_sequence, -1, axis=1)
                    last_sequence[0, -1, 0] = pred[0, 0]
                
                predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
                
            elif model_type == 'ARIMA':
                if not STATSMODELS_AVAILABLE:
                    st.error("Statsmodels is required for ARIMA model. Please install it with: pip install statsmodels")
                    return None
                    
                # ARIMA model needs a series without NaNs
                series_for_arima = df[target].dropna()
                if series_for_arima.empty:
                    st.error("Not enough data for ARIMA model after cleaning.")
                    return None

                model = ARIMA(series_for_arima, order=(5,1,0))
                fitted_model = model.fit()
                predictions = fitted_model.forecast(steps=forecast_days)
                
            # Create future dates
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
            
            # Calculate confidence intervals (simple approach)
            volatility = df[target].pct_change().std()
            upper_bound = predictions * (1 + 2 * volatility)
            lower_bound = predictions * (1 - 2 * volatility)
            
            prediction_df = pd.DataFrame({
                'date': future_dates,
                'predicted_price': predictions,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound,
                'confidence': 0.95  # 95% confidence interval
            })
            
            return prediction_df
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None

def create_chart(df, signals_df, prediction_df=None):
    """Create comprehensive trading chart"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price & Indicators', 'Volume', 'RSI', 'MACD'),
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ), row=1, col=1)
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['MA
