"""
Quick Data Expansion
=====================
Your dataset is too small (365 rows).
This script gets MORE data quickly!
"""

import pandas as pd
import ccxt
from pathlib import Path
import time

print("\n" + "="*60)
print("📊 QUICK DATA EXPANSION")
print("="*60)
print("\nYour BTC_USDT_4h.csv only has 365 rows!")
print("Let's get more data...\n")

# Setup
data_dir = Path('training_data/crypto')
data_dir.mkdir(parents=True, exist_ok=True)

exchange = ccxt.binance({'enableRateLimit': True})

# Get MORE data for multiple timeframes
configs = [
    ('BTC/USDT', '1h', 'BTC_USDT_1h.csv'),
    ('BTC/USDT', '4h', 'BTC_USDT_4h.csv'),
    ('ETH/USDT', '1h', 'ETH_USDT_1h.csv'),
]

for pair, timeframe, filename in configs:
    print(f"\n📥 Fetching {pair} {timeframe}...")
    
    try:
        all_data = []
        since = exchange.parse8601("2023-01-01T00:00:00Z")  # 2 years
        
        while True:
            ohlcv = exchange.fetch_ohlcv(pair, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            
            all_data += ohlcv
            since = ohlcv[-1][0] + 1
            
            print(f"   Got {len(all_data)} candles...", end='\r')
            time.sleep(0.3)  # Rate limit
        
        # Save
        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        filepath = data_dir / filename
        df.to_csv(filepath, index=False)
        
        print(f"   ✅ Saved {len(df):,} candles to {filename}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("✅ DONE!")
print("="*60)
print("\nNow run: python Train_ai_simple.py")
