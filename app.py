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
                # Prepare features
                df_features = df[['MA7', 'MA25', 'RSI', 'MACD', 'volume']].fillna(method='ffill')
                X = df_features.values
                y = df[target].values
                
                # Train model
                model = LinearRegression()
                train_size = int(0.8 * len(X))
                X_train, y_train = X[:train_size], y[:train_size]
                model.fit(X_train, y_train)
                
                # Predict future
                last_features = X[-1:].repeat(forecast_days, axis=0)
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
                    
                # ARIMA model
                model = ARIMA(df[target].fillna(method='ffill'), order=(5,1,0))
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
    fig.add_trace(go.Scatter(x=df.index, y=df['MA7'], name='MA7', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA25'], name='MA25', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA99'], name='MA99', line=dict(color='purple')), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # Buy/Sell Signals
    if not signals_df.empty:
        buy_signals = signals_df[signals_df['signal'] == 'BUY']
        sell_signals = signals_df[signals_df['signal'] == 'SELL']
        
        fig.add_trace(go.Scatter(
            x=buy_signals['timestamp'],
            y=buy_signals['price'],
            mode='markers',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            name='Buy Signal'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=sell_signals['timestamp'],
            y=sell_signals['price'],
            mode='markers',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            name='Sell Signal'
        ), row=1, col=1)
    
    # Predictions
    if prediction_df is not None:
        fig.add_trace(go.Scatter(
            x=prediction_df['date'],
            y=prediction_df['predicted_price'],
            name='Prediction',
            line=dict(color='yellow', width=3)
        ), row=1, col=1)
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=prediction_df['date'],
            y=prediction_df['upper_bound'],
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0)',
            showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=prediction_df['date'],
            y=prediction_df['lower_bound'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0)',
            name='95% Confidence',
            fillcolor='rgba(255,255,0,0.2)'
        ), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='blue'), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal', line=dict(color='red')), row=4, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_histogram'], name='Histogram', marker_color='gray'), row=4, col=1)
    
    fig.update_layout(
        title="Advanced Crypto Analysis & Forecasting",
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def main():
    # Header
    st.markdown('<div class="main-header"><h1 style="color: white; text-align: center;">🚀 Advanced Crypto Analysis & 30-Day Forecasting</h1></div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
# Crypto selection
    crypto_symbols = {
    'Bitcoin': 'bitcoin',
    'Ethereum': 'ethereum',
    'Tether': 'tether', # USDT
    'BNB': 'binancecoin',
    'Solana': 'solana',
    'XRP': 'ripple',
    'Dogecoin': 'dogecoin',
    'Cardano': 'cardano',
    'Shiba Inu': 'shiba-inu',
    'Avalanche': 'avalanche-2',
    'Polkadot': 'polkadot',
    'Tron': 'tron',
    'Polygon': 'matic-network', # MATIC
    'Chainlink': 'chainlink', # LINK
    'Litecoin': 'litecoin', # LTC
    'Cosmos': 'cosmos', # ATOM
    'Ethereum Classic': 'ethereum-classic', # ETC
    'Monero': 'monero', # XMR
    'NEAR Protocol': 'near', # NEAR
    'Algorand': 'algorand', # ALGO
    'Decentraland': 'decentraland', # MANA
    'The Sandbox': 'the-sandbox', # SAND
    'Axie Infinity': 'axie-infinity', # AXS
    'ImmutableX': 'immutable-x', # IMX
    'Render Token': 'render-token', # RNDR
    'Pepe': 'pepe', # PEPE
    'Floki': 'floki', # FLOKI
    'Injective Protocol': 'injective-protocol', # INJ
    'Sui': 'sui', # SUI
    'Arbitrum': 'arbitrum', # ARB
    'Optimism': 'optimism', # OP
    'Aptos': 'aptos', # APT
    'Hedera': 'hedera-hashgraph', # HBAR
    'Filecoin': 'filecoin', # FIL
    'The Graph': 'the-graph', # GRT
    'Aave': 'aave', # AAVE
    'Uniswap': 'uniswap', # UNI
    'Maker': 'maker', # MKR
    'Compound': 'compound-governance-token', # COMP
    'Fantom': 'fantom', # FTM
    'Cronos': 'cronos', # CRO
    'VeChain': 'vechain', # VET
    'EOS': 'eos', # EOS
    'IOTA': 'iota', # IOTA
    'Kusama': 'kusama', # KSM
    'Conflux': 'conflux-token', # CFX
    'Gala': 'gala' # GALA
    }
    
    selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", list(crypto_symbols.keys()))
    symbol = crypto_symbols[selected_crypto]
    
    # Timeframe
    timeframe = st.sidebar.selectbox("Timeframe", ['1d', '4h', '1w'])
    
    # Prediction settings
    st.sidebar.subheader("🤖 Prediction Settings")
    
    # Filter available models based on installed packages
    available_models = ['Linear Regression']
    if TENSORFLOW_AVAILABLE:
        available_models.append('LSTM')
    if STATSMODELS_AVAILABLE:
        available_models.append('ARIMA')
    
    model_type = st.sidebar.selectbox("AI Model", available_models)
    prediction_target = st.sidebar.selectbox("Prediction Target", ['close', 'high', 'low'])
    forecast_days = st.sidebar.slider("Forecast Days", 7, 30, 30)
    
    # Initialize analyzer
    analyzer = CryptoAnalyzer()
    
    if st.sidebar.button("🔄 Analyze & Predict", type="primary"):
        with st.spinner("Fetching data and generating analysis..."):
            # Fetch data
            df = analyzer.fetch_crypto_data(symbol, timeframe)
            
            if df is not None:
                # Calculate indicators
                df = analyzer.calculate_indicators(df)
                
                # Generate signals
                signals_df = analyzer.generate_signals(df)
                
                # Generate predictions
                prediction_df = analyzer.predict_prices(df, model_type, prediction_target, forecast_days)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = df['close'].iloc[-1]
                price_change = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
                
                with col1:
                    st.metric("Current Price", f"${current_price:.4f}", f"{price_change:.2f}%")
                
                with col2:
                    st.metric("24h Volume", f"${df['volume'].iloc[-1]:,.0f}")
                
                with col3:
                    rsi_value = df['RSI'].iloc[-1]
                    rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                    st.metric("RSI", f"{rsi_value:.1f}", rsi_status)
                
                with col4:
                    if prediction_df is not None:
                        future_price = prediction_df['predicted_price'].iloc[-1]
                        price_change_pred = ((future_price - current_price) / current_price) * 100
                        st.metric(f"{forecast_days}d Prediction", f"${future_price:.4f}", f"{price_change_pred:.2f}%")
                
                # Display chart
                fig = create_chart(df, signals_df, prediction_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction summary
                if prediction_df is not None:
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.subheader("🔮 30-Day Forecast Summary")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Model Used:** {model_type}")
                        st.write(f"**Target:** {prediction_target.title()} Price")
                        st.write(f"**Current Price:** ${current_price:.4f}")
                        st.write(f"**Predicted Price ({forecast_days} days):** ${prediction_df['predicted_price'].iloc[-1]:.4f}")
                    
                    with col2:
                        trend = "📈 Bullish" if prediction_df['predicted_price'].iloc[-1] > current_price else "📉 Bearish"
                        st.write(f"**Trend:** {trend}")
                        st.write(f"**Price Range:** ${prediction_df['lower_bound'].iloc[-1]:.4f} - ${prediction_df['upper_bound'].iloc[-1]:.4f}")
                        st.write(f"**Confidence:** 95%")
                        
                        # Calculate max gain/loss potential
                        max_gain = ((prediction_df['upper_bound'].max() - current_price) / current_price) * 100
                        max_loss = ((prediction_df['lower_bound'].min() - current_price) / current_price) * 100
                        st.write(f"**Max Potential:** +{max_gain:.1f}% / {max_loss:.1f}%")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Recent signals
                if not signals_df.empty:
                    st.subheader("📊 Recent Trading Signals")
                    recent_signals = signals_df.tail(5)
                    
                    for _, signal in recent_signals.iterrows():
                        signal_color = "🟢" if signal['signal'] == 'BUY' else "🔴"
                        st.write(f"{signal_color} **{signal['signal']}** at ${signal['price']:.4f} - {signal['reason']} ({signal['timestamp'].strftime('%Y-%m-%d %H:%M')})")
                
                # Export options
                st.subheader("📥 Export Data")
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction_df is not None:
                        csv_data = prediction_df.to_csv(index=False)
                        st.download_button(
                            "Download Predictions CSV",
                            csv_data,
                            f"{selected_crypto}_predictions.csv",
                            "text/csv"
                        )
                
                with col2:
                    if not signals_df.empty:
                        signals_csv = signals_df.to_csv(index=False)
                        st.download_button(
                            "Download Signals CSV",
                            signals_csv,
                            f"{selected_crypto}_signals.csv",
                            "text/csv"
                        )
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Disclaimer:** This is for educational purposes only. Not financial advice. Always do your own research before making investment decisions.")

if __name__ == "__main__":
    main()
