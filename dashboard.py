"""
Enhanced Trading Dashboard v2.0
================================
Professional Streamlit dashboard with:
- Real-time price updates
- Interactive charts
- Trade execution interface
- Performance metrics
- Risk monitoring
- Multi-asset support
- AI insights
- Trade history
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
from stable_baselines3 import PPO
from pathlib import Path

# Import our enhanced modules
try:
    from market_engine  import get_crypto_price, get_ohlcv, get_stock_price, get_stock_ohlcv, get_market_summary
    from strategy_engine import analyze_market
    from ai_reasoning_gemini import explain_decision, get_gemini_stats
    from trade_executor import TradeExecutor, ExecutionMode
    from notifications import NotificationManager, NotificationType
except ImportError as e:
    st.error(f"⚠️ Import error: {e}. Make sure all modules are in the same directory!")

logger = logging.getLogger(__name__)

import sys
import io
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Check if trained model exists
model_path = Path('rl_models/final_model.zip')

if model_path.exists():
    trained_ai = PPO.load(str(model_path))
    st.sidebar.success("✅ Using Trained AI Model!")
else:
    trained_ai = None
    st.sidebar.info("ℹ️ Train AI with: python Train_ai_simple.py")


# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title=" Trading Agent",
    page_icon="📈",
    layout="wide",  # Use full width
    initial_sidebar_state="expanded"
)


# ==========================================
# CUSTOM CSS (NEW!)
# ==========================================
st.markdown("""
<style>
    /* Main theme colors */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    
    /* Signal badges */
    .signal-buy {
        background-color: #00ff00;
        color: #000;
        padding: 5px 15px;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .signal-sell {
        background-color: #ff0000;
        color: #fff;
        padding: 5px 15px;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .signal-hold {
        background-color: #888;
        color: #fff;
        padding: 5px 15px;
        border-radius: 5px;
        font-weight: bold;
    }
    
    /* Risk indicators */
    .risk-low { color: #00ff00; }
    .risk-moderate { color: #ffaa00; }
    .risk-high { color: #ff6600; }
    .risk-very-high { color: #ff0000; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE INITIALIZATION (NEW!)
# ==========================================
if 'executor' not in st.session_state:
    st.session_state.executor = None

if 'notifier' not in st.session_state:
    class NotificationManager:
        def __init__(self):
            # initialize your notifier here
            pass

        def notify(self, message):
            print(f"Notification: {message}")

    # create an instance and store it in session_state
    st.session_state.notifier = NotificationManager()

if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None

if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []


# ==========================================
# HEADER
# ==========================================
st.title("📈 Whitney Personal Trading Agent")
st.markdown("### AI-Powered Multi-Asset Trading System")
st.markdown("---")


# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Asset Selection
    st.subheader("Asset Selection")
    
    asset_type = st.radio(
        "Asset Type",
        ["Crypto", "Stock"],
        help="Choose between cryptocurrency or stock trading"
    )
    
    if asset_type == "Crypto":
        crypto_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"]
        asset = st.selectbox("Crypto Pair", crypto_symbols)
        timeframe = st.selectbox(
            "Timeframe",
            ["15m", "1h", "4h", "1d"],
            index=1,
            help="Chart timeframe for analysis"
        )
    else:
        stock_tickers = ["PLTR", "T", "IBM", "CSCO", "AAPL", "MSFT", "TSLA", "SPY"]
        asset = st.selectbox("Stock Ticker", stock_tickers)
        timeframe = st.selectbox(
            "Timeframe",
            ["1h", "1d", "1wk"],
            index=1
        )
    
    st.markdown("---")
    
    # Trading Mode
    st.subheader("Trading Mode")
    mode = st.radio(
        "Mode",
        ["Paper Trading", "Live Trading"],
        help="Paper = Simulated, Live = Real money"
    )
    
    # Capital and Risk
    st.subheader("Capital & Risk")
    capital = st.number_input(
        "Trading Capital (KES)",
        min_value=10000,
        max_value=10000000,
        value=500000,
        step=10000,
        help="Your total trading capital in Kenyan Shillings"
    )
    
    risk_profile = st.select_slider(
        "Risk Profile",
        options=["Conservative", "Moderate", "Aggressive"],
        value="Moderate"
    )
    
    max_risk_pct = st.slider(
        "Max Risk per Trade (%)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Maximum percentage of capital to risk per trade"
    )
    
    st.markdown("---")
    
    # Actions
    st.subheader("Actions")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    
    if st.button("⚙️ View System Stats", use_container_width=True):
        st.session_state.show_stats = not st.session_state.get('show_stats', False)
    
    if st.button("📧 Test Notifications", use_container_width=True):
        test_result = st.session_state.notifier.send_notification(
            NotificationType.SYSTEM_STATUS,
            {"message": "Test notification from Whitney Trading Agent"}
        )
        if test_result["success"]:
            st.success("✅ Notification sent!")
        else:
            st.error("❌ Notification failed")


# ==========================================
# MAIN CONTENT
# ==========================================

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header(f"📊 {asset} Analysis")
    
    # Fetch data with loading spinner
    with st.spinner(f"Fetching {asset} data..."):
        try:
            if asset_type == "Crypto":
                df = get_ohlcv(asset, timeframe=timeframe, limit=200, add_indicators=True)
                current_price = get_crypto_price(asset)
            else:
                period_map = {"1h": "5d", "1d": "3mo", "1wk": "1y"}
                period = period_map.get(timeframe, "1mo")
                df = get_stock_ohlcv(asset, period=period, interval=timeframe)
                current_price = get_stock_price(asset)
            
            if df.empty:
                st.error(f"❌ No data available for {asset}")
                st.stop()
            
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
            st.stop()
    
    # Display current price prominently
    if current_price:
        price_col1, price_col2, price_col3 = st.columns(3)
        
        with price_col1:
            st.metric(
                "Current Price",
                f"${current_price:,.2f}" if asset_type == "Crypto" else f"${current_price:.2f}",
                delta=None
            )
        
        with price_col2:
            if len(df) > 1:
                price_change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
                st.metric(
                    "24h Change",
                    f"{price_change:+.2f}%",
                    delta=f"{price_change:.2f}%"
                )
        
        with price_col3:
            if 'volume' in df.columns:
                st.metric(
                    "24h Volume",
                    f"{df['volume'].iloc[-1]:,.0f}"
                )
    
    # Create interactive chart with Plotly
    st.subheader("Price Chart with Indicators")
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('Price & MAs', 'RSI', 'MACD')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving averages
    if 'sma_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['sma_20'], name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
    if 'ema_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['ema_50'], name='EMA 50', line=dict(color='blue')),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'bb_high' in df.columns and 'bb_low' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['bb_high'], name='BB Upper', line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['bb_low'], name='BB Lower', line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['rsi'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['macd'], name='MACD', line=dict(color='blue')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['macd_signal'], name='Signal', line=dict(color='red')),
            row=3, col=1
        )
        if 'macd_diff' in df.columns:
            fig.add_trace(
                go.Bar(x=df['time'], y=df['macd_diff'], name='Histogram'),
                row=3, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)


with col2:
    st.header("🤖 AI Analysis")
    
    # Analyze button
    if st.button("🔍 Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing market..."):
            try:
                # Run strategy analysis
                analysis = analyze_market(
                    df=df,
                    asset=asset,
                    timeframe=timeframe,
                    risk_profile=risk_profile.lower()
                )
                
                st.session_state.last_analysis = analysis
                
            except Exception as e:
                st.error(f"Analysis error: {e}")
                st.stop()
    
    # Display analysis if available
    if st.session_state.last_analysis:
        analysis = st.session_state.last_analysis
        
        # Signal display
        signal = analysis.get('signal', 'ERROR')
        confidence = analysis.get('confidence', 0)
        
        if signal == "BUY":
            st.markdown(f'<div class="signal-buy">🟢 BUY SIGNAL</div>', unsafe_allow_html=True)
        elif signal == "SELL":
            st.markdown(f'<div class="signal-sell">🔴 SELL SIGNAL</div>', unsafe_allow_html=True)
        elif signal == "HOLD":
            st.markdown(f'<div class="signal-hold">⚪ HOLD</div>', unsafe_allow_html=True)
        else:
            st.error(f"⚠️ {signal}")
        
        st.markdown("---")
        
        # Metrics
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Confidence", f"{confidence:.0f}%")
        with metric_col2:
            st.metric("Score", f"{analysis.get('score', 0):.1f}/10")
        
        # Trend
        trend = analysis.get('trend', {})
        st.subheader("📈 Trend Analysis")
        st.write(f"**Direction:** {trend.get('direction', 'UNKNOWN')}")
        st.write(f"**Strength:** {trend.get('strength', 0)}/10")
        st.progress(trend.get('strength', 0) / 10)
        
        # Risk
        risk = analysis.get('risk_assessment', {})
        st.subheader("⚠️ Risk Assessment")
        risk_level = risk.get('level', 'UNKNOWN')
        risk_class = f"risk-{risk_level.lower().replace('_', '-')}"
        st.markdown(f'<p class="{risk_class}">Risk Level: {risk_level}</p>', unsafe_allow_html=True)
        
        # Position sizing
        position = analysis.get('position_sizing', {})
        st.subheader("💰 Position Sizing")
        recommended_pct = position.get('recommended_pct', 0)
        st.write(f"**Recommended:** {recommended_pct}% of capital")
        
        recommended_kes = capital * (recommended_pct / 100)
        st.write(f"**Amount:** {recommended_kes:,.0f} KES")
        
        # Reasoning
        with st.expander("📋 Detailed Reasoning", expanded=False):
            for reason in analysis.get('reasoning', []):
                st.write(f"• {reason}")
        
        # Warnings
        warnings = analysis.get('warnings', [])
        if warnings:
            with st.expander("⚠️ Warnings", expanded=True):
                for warning in warnings:
                    st.warning(warning)
        
        # Get AI explanation
        with st.expander("🧠 AI Expert Insights", expanded=True):
            if st.button("Get AI Explanation", use_container_width=True):
                with st.spinner("Consulting AI..."):
                    try:
                        ai_result = explain_decision(
                            signal=signal,
                            asset=asset,
                            analysis_data=analysis,
                            risk_profile=risk_profile.lower()
                        )
                        
                        st.write(ai_result.get('explanation', 'No explanation available'))
                        
                        # Show structured data if available
                        if ai_result.get('structured'):
                            with st.expander("📊 Structured Analysis"):
                                st.json(ai_result['structured'])
                        
                    except Exception as e:
                        st.error(f"AI analysis failed: {e}")
        
        # Trade execution
        st.markdown("---")
        st.subheader("⚡ Execute Trade")
        
        if signal in ["BUY", "SELL"]:
            if st.button(f"Execute {signal} Order", type="primary", use_container_width=True):
                # Initialize executor if not exists
                if st.session_state.executor is None:
                    st.session_state.executor = TradeExecutor(
                        capital_kes=capital,
                        max_risk_per_trade_pct=max_risk_pct,
                        mode=ExecutionMode.PAPER if mode == "Paper Trading" else ExecutionMode.LIVE
                    )
                
                # Execute trade
                with st.spinner(f"Executing {signal} order..."):
                    try:
                        result = st.session_state.executor.execute_trade(
                            signal=signal,
                            asset=asset,
                            analysis_data=analysis
                        )
                        
                        if result.get('success'):
                            st.success(f"✅ Trade executed successfully!")
                            st.json(result)
                            
                            # Send notification
                            st.session_state.notifier.send_notification(
                                NotificationType.TRADE_EXECUTED,
                                result
                            )
                            
                            # Add to history
                            st.session_state.trade_history.append(result)
                        else:
                            st.error(f"❌ Trade failed: {result.get('message', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"Execution error: {e}")
        else:
            st.info("⏸️ No trade signal - HOLD recommended")


# ==========================================
# BOTTOM SECTIONS
# ==========================================

st.markdown("---")

# Tabs for additional information
tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Summary", "📝 Trade History", "📈 Performance", "⚙️ System Stats"])

with tab1:
    st.subheader("Market Summary")
    
    if st.button("Refresh Market Summary"):
        with st.spinner("Fetching market data..."):
            try:
                summary = get_market_summary()
                
                col_crypto, col_stocks = st.columns(2)
                
                with col_crypto:
                    st.write("**Crypto Markets:**")
                    for pair, price in summary.get('crypto', {}).items():
                        st.write(f"{pair}: ${price:,.2f}")
                
                with col_stocks:
                    st.write("**Stock Markets:**")
                    for ticker, price in summary.get('stocks', {}).items():
                        st.write(f"{ticker}: ${price:.2f}")
            
            except Exception as e:
                st.error(f"Error fetching market summary: {e}")

with tab2:
    st.subheader("Trade History")
    
    if st.session_state.trade_history:
        history_df = pd.DataFrame(st.session_state.trade_history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No trades executed yet")

with tab3:
    st.subheader("Performance Metrics")
    
    if st.session_state.executor:
        stats = st.session_state.executor.get_statistics()
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("Total Trades", stats.get('total_trades', 0))
        with perf_col2:
            st.metric("Winning Trades", stats.get('winning_trades', 0))
        with perf_col3:
            st.metric("Losing Trades", stats.get('losing_trades', 0))
        with perf_col4:
            win_rate = 0
            if stats.get('total_trades', 0) > 0:
                win_rate = (stats.get('winning_trades', 0) / stats['total_trades']) * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
    else:
        st.info("Start trading to see performance metrics")

with tab4:
    st.subheader("System Statistics")
    
    # AI stats
    ai_stats = get_gemini_stats()
    st.write(ai_stats)
    
    st.write("**AI Usage:**")
    st.json(ai_stats)
    
    # Notification stats
    notif_stats = st.session_state.notifier.get_stats()
    st.write("Notifications stats:", notif_stats)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Whitney Trading Agent v2.0 | Powered by Claude AI</p>
        <p>⚠️ Trading involves risk. Never invest more than you can afford to lose.</p>
    </div>
    """,
    unsafe_allow_html=True
)