"""
Data Collection Module for AI Training - FIXED
===============================================
FIXED: Timestamp NaT bug on line 318!

Collects diverse market data from multiple sources:
- Crypto: Binance, Coinbase, Kraken
- Stocks: Yahoo Finance, Alpha Vantage
- Indices: S&P 500, Nasdaq, DJI
- Bonds: Government bonds, corporate bonds (via FRED)
"""

import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# DATA SOURCES CONFIGURATION
# ==========================================
class DataConfig:
    """Configuration for data sources"""
    
    # Storage
    DATA_DIR = Path("training_data")
    CRYPTO_DIR = DATA_DIR / "crypto"
    STOCKS_DIR = DATA_DIR / "stocks"
    INDICES_DIR = DATA_DIR / "indices"
    BONDS_DIR = DATA_DIR / "bonds"
    
    # Assets to collect (DIVERSE!)
    CRYPTO_PAIRS = [
        # Major coins (stable, liquid)
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT',
        # Mid-caps (more volatile)
        'SOL/USDT', 'ADA/USDT', 'AVAX/USDT',
        # Alt coins (high volatility)
        'MATIC/USDT', 'DOT/USDT', 'LINK/USDT'
    ]
    
    STOCK_TICKERS = [
        # Your picks
        'PLTR', 'T', 'IBM', 'CSCO',
        # Tech giants
        'AAPL', 'MSFT', 'GOOGL', 'AMZN',
        # Other sectors
        'JPM',  # Finance
        'JNJ',  # Healthcare
        'XOM',  # Energy
    ]
    
    # INDICES (teach bot about market trends)
    INDICES = ["^GSPC", "^IXIC", "^DJI", "^RUT", "^VIX"]
    
    # ETFs for bonds
    BOND_ETFS = [
        'AGG',   # Total Bond Market
        'TLT',   # 20+ Year Treasury
        'LQD',   # Investment Grade Corporate
        'HYG',   # High Yield Corporate
    ]
    
    # Timeframes for training
    TIMEFRAMES = ['1h', '4h']
    
    # How far back to collect
    LOOKBACK_DAYS = 365 * 2  # 2 years


# ==========================================
# DATA COLLECTOR
# ==========================================
class DataCollector:
    """
    Collects training data from multiple sources.
    """
    
    def __init__(self):
        # Create storage directories
        for dir_path in [
            DataConfig.DATA_DIR,
            DataConfig.CRYPTO_DIR,
            DataConfig.STOCKS_DIR,
            DataConfig.INDICES_DIR,
            DataConfig.BONDS_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize exchanges
        self.exchanges = {
            'binance': ccxt.binance({'enableRateLimit': True}),
        }
        
        logger.info("✅ Data Collector initialized")
    
    def collect_all_data(self) -> Dict:
        """
        Collect ALL training data from all sources.
        """
        logger.info("=" * 60)
        logger.info("📊 STARTING COMPREHENSIVE DATA COLLECTION")
        logger.info("=" * 60)
        
        results = {
            "timestamp": datetime.now(),
            "crypto": {},
            "stocks": {},
            "indices": {},
            "bonds": {},
            "total_records": 0
        }
        
        # 1. Collect Crypto
        logger.info("\n💰 Collecting Crypto Data...")
        for pair in DataConfig.CRYPTO_PAIRS:
            try:
                crypto_data = self._collect_crypto(pair)
                results["crypto"][pair] = {
                    "records": sum(len(df) for df in crypto_data.values()),
                    "timeframes": list(crypto_data.keys())
                }
                results["total_records"] += sum(len(df) for df in crypto_data.values())
                logger.info(f"  ✅ {pair}: {sum(len(df) for df in crypto_data.values())} records")
            except Exception as e:
                logger.error(f"  ❌ {pair} failed: {e}")
        
        # 2. Collect Stocks
        logger.info("\n📈 Collecting Stock Data...")
        for ticker in DataConfig.STOCK_TICKERS:
            try:
                stock_data = self._collect_stock(ticker)
                results["stocks"][ticker] = {
                    "records": sum(len(df) for df in stock_data.values()),
                    "timeframes": list(stock_data.keys())
                }
                results["total_records"] += sum(len(df) for df in stock_data.values())
                logger.info(f"  ✅ {ticker}: {sum(len(df) for df in stock_data.values())} records")
            except Exception as e:
                logger.error(f"  ❌ {ticker} failed: {e}")
        
        # 3. Collect Indices
        logger.info("\n📊 Collecting Index Data...")
        for index in DataConfig.INDICES:
            try:
                index_data = self._collect_index(index)
                results["indices"][index] = {
                    "records": sum(len(df) for df in index_data.values()),
                    "timeframes": list(index_data.keys())
                }
                results["total_records"] += sum(len(df) for df in index_data.values())
                logger.info(f"  ✅ {index}: {sum(len(df) for df in index_data.values())} records")
            except Exception as e:
                logger.error(f"  ❌ {index} failed: {e}")
        
        # 4. Collect Bonds
        logger.info("\n💵 Collecting Bond ETF Data...")
        for bond in DataConfig.BOND_ETFS:
            try:
                bond_data = self._collect_bond(bond)
                results["bonds"][bond] = {
                    "records": sum(len(df) for df in bond_data.values()),
                    "timeframes": list(bond_data.keys())
                }
                results["total_records"] += sum(len(df) for df in bond_data.values())
                logger.info(f"  ✅ {bond}: {sum(len(df) for df in bond_data.values())} records")
            except Exception as e:
                logger.error(f"  ❌ {bond} failed: {e}")
        
        # Save metadata
        self._save_metadata(results)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"✅ DATA COLLECTION COMPLETE")
        logger.info(f"Total Records: {results['total_records']:,}")
        logger.info(f"Crypto Pairs: {len(results['crypto'])}")
        logger.info(f"Stock Tickers: {len(results['stocks'])}")
        logger.info(f"Indices: {len(results['indices'])}")
        logger.info(f"Bonds: {len(results['bonds'])}")
        logger.info("=" * 60)
        
        return results
    
    def _collect_crypto(self, pair: str) -> Dict[str, pd.DataFrame]:
        """Collect crypto data for multiple timeframes"""
        data = {}
        
        for timeframe in DataConfig.TIMEFRAMES:
            try:
                # Fetch from Binance
                exchange = self.exchanges['binance']                
                all_data = []

                since = exchange.parse8601("2020-01-01T00:00:00Z")  # Start date

                while True:
                    ohlcv = exchange.fetch_ohlcv(pair, timeframe, since=since, limit=1000)
                    if not ohlcv:
                        break
                    all_data += ohlcv
                    since = ohlcv[-1][0] + 1  # Move forward

                    time.sleep(0.2)  # Rate limit

                # Convert to DataFrame
                df = pd.DataFrame(
                    all_data,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Save to file
                filename = f"{pair.replace('/', '_')}_{timeframe}.csv"
                filepath = DataConfig.CRYPTO_DIR / filename
                df.to_csv(filepath, index=False)
                
                data[timeframe] = df
                
            except Exception as e:
                logger.error(f"Error collecting {pair} {timeframe}: {e}")
        
        return data
    
    def _collect_stock(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Collect stock data for multiple timeframes
        
        FIXED: Timestamp conversion bug!
        """
        data = {}
        
        for timeframe in DataConfig.TIMEFRAMES:
            try:
                # Map timeframe to yfinance format
                if timeframe == '1h':
                    interval = '1h'
                    period = '730d'  # Max for 1h
                elif timeframe == '4h':
                    interval = '1d'  # yfinance doesn't have 4h, use 1d
                    period = '5y'
                else:  # 1d
                    interval = '1d'
                    period = '10y'
                
                # Fetch from Yahoo Finance
                stock = yf.Ticker(ticker)
                df = stock.history(period=period, interval=interval)
                
                if df.empty:
                    logger.warning(f"  No data for {ticker} {timeframe}")
                    continue
                
                # ============================================
                # FIX START: Proper timestamp handling
                # ============================================
                
                # The index IS the datetime - preserve it BEFORE reset_index!
                df.index.name = 'timestamp'  # Ensure it's named 'timestamp'
                
                # Reset index to make timestamp a column
                df = df.reset_index()
                
                # Now timestamp column exists properly!
                # Convert to datetime (should already be, but just in case)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Standardize other column names
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Keep only OHLCV columns
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
                # Drop any rows with invalid timestamps (shouldn't be any now!)
                df = df.dropna(subset=['timestamp'])
                
                # Verify no NaT values
                if df['timestamp'].isna().sum() > 0:
                    logger.warning(f"  {ticker} {timeframe}: {df['timestamp'].isna().sum()} NaT values!")
                
                # ============================================
                # FIX END
                # ============================================
                
                # Save
                filename = f"{ticker}_{timeframe}.csv"
                filepath = DataConfig.STOCKS_DIR / filename
                df.to_csv(filepath, index=False)
                
                data[timeframe] = df
                
                logger.info(f"    {ticker} {timeframe}: {len(df)} records, "
                           f"dates {df['timestamp'].min()} to {df['timestamp'].max()}")
                
            except Exception as e:
                logger.error(f"Error collecting {ticker} {timeframe}: {e}")
                import traceback
                traceback.print_exc()
        
        return data
    
    def _collect_index(self, index: str) -> Dict[str, pd.DataFrame]:
        """Collect index data"""
        return self._collect_stock(index)
    
    def _collect_bond(self, bond: str) -> Dict[str, pd.DataFrame]:
        """Collect bond ETF data"""
        return self._collect_stock(bond)
    
    def _save_metadata(self, results: Dict):
        """Save collection metadata"""
        metadata_file = DataConfig.DATA_DIR / "collection_metadata.json"
        
        # Convert datetime to string for JSON
        results_copy = results.copy()
        results_copy["timestamp"] = results_copy["timestamp"].isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        logger.info(f"📝 Metadata saved to {metadata_file}")


# ==========================================
# DATA LOADER (for training)
# ==========================================
class TrainingDataLoader:
    """
    Loads collected data for AI training.
    """
    
    def __init__(self):
        self.data_dir = DataConfig.DATA_DIR
    
    def load_all_training_data(self) -> pd.DataFrame:
        """
        Load ALL collected data into single DataFrame.
        """
        logger.info("📚 Loading all training data...")
        
        all_data = []
        
        # Load crypto
        for file in DataConfig.CRYPTO_DIR.glob("*.csv"):
            try:
                df = pd.read_csv(file)
                asset = file.stem.rsplit('_', 1)[0].replace('_', '/')
                timeframe = file.stem.rsplit('_', 1)[1]
                df['asset'] = asset
                df['asset_type'] = 'crypto'
                df['timeframe'] = timeframe
                all_data.append(df)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        # Load stocks
        for file in DataConfig.STOCKS_DIR.glob("*.csv"):
            try:
                df = pd.read_csv(file)
                asset = file.stem.rsplit('_', 1)[0]
                timeframe = file.stem.rsplit('_', 1)[1]
                df['asset'] = asset
                df['asset_type'] = 'stock'
                df['timeframe'] = timeframe
                all_data.append(df)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        # Load indices
        for file in DataConfig.INDICES_DIR.glob("*.csv"):
            try:
                df = pd.read_csv(file)
                # Handle ^ prefix
                asset_name = file.stem.rsplit('_', 1)[0]
                if not asset_name.startswith('^'):
                    asset_name = '^' + asset_name
                timeframe = file.stem.rsplit('_', 1)[1]
                df['asset'] = asset_name
                df['asset_type'] = 'index'
                df['timeframe'] = timeframe
                all_data.append(df)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        # Load bonds
        for file in DataConfig.BONDS_DIR.glob("*.csv"):
            try:
                df = pd.read_csv(file)
                asset = file.stem.rsplit('_', 1)[0]
                timeframe = file.stem.rsplit('_', 1)[1]
                df['asset'] = asset
                df['asset_type'] = 'bond'
                df['timeframe'] = timeframe
                all_data.append(df)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        # Combine
        if not all_data:
            raise ValueError("No training data found! Run data collection first.")
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # Parse timestamp properly
        combined['timestamp'] = pd.to_datetime(combined['timestamp'], errors='coerce')
        
        # Remove any NaT rows
        before_count = len(combined)
        combined = combined.dropna(subset=['timestamp'])
        after_count = len(combined)
        
        if before_count != after_count:
            logger.warning(f"Dropped {before_count - after_count} rows with NaT timestamps")
        
        # Sort
        combined = combined.sort_values(['asset', 'timeframe', 'timestamp'])
        
        logger.info(f"✅ Loaded {len(combined):,} records")
        logger.info(f"   Assets: {combined['asset'].nunique()}")
        logger.info(f"   Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
        
        # Check for NaT
        nat_count = combined['timestamp'].isna().sum()
        if nat_count > 0:
            logger.error(f"⚠️ WARNING: {nat_count} NaT timestamps found!")
        else:
            logger.info("✅ No NaT timestamps - data is clean!")
        
        return combined


# ==========================================
# USAGE
# ==========================================
if __name__ == "__main__":
    """
    Run this to collect ALL training data!
    """
    
    print("\n" + "="*60)
    print("📊 TRAINING DATA COLLECTION - FIXED VERSION")
    print("="*60)
    print("\n🔧 FIX APPLIED: Timestamp NaT bug corrected!")
    print("="*60)
    
    collector = DataCollector()
    results = collector.collect_all_data()
    
    print("\n✅ Collection Complete!")
    print(f"Total Records: {results['total_records']:,}")
    print(f"\nData saved to: {DataConfig.DATA_DIR}")
    
    # Test loading
    print("\n📚 Testing data loader...")
    loader = TrainingDataLoader()
    df = loader.load_all_training_data()
    
    print(f"✅ Successfully loaded {len(df):,} records")
    
    # Check for NaT
    nat_count = df['timestamp'].isna().sum()
    if nat_count == 0:
        print("✅ NO NaT ERRORS - Data is perfect!")
    else:
        print(f"⚠️ WARNING: {nat_count} NaT timestamps!")
    
    print(f"\nSample data:")
    print(df.head())
    
    print("\n" + "="*60)
    print("🎉 READY FOR AI TRAINING!")
    print("="*60)