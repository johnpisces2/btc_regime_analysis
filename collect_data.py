import ccxt
import pandas as pd
from datetime import datetime, timezone, timedelta
import time
import os
import argparse

UTC8 = timezone(timedelta(hours=8))


def fetch_btc_ohlcv(symbol="BTC/USDT", exchange_id="binance", 
                     start_date="2018-01-01", end_date=None,
                     timeframe="4h", limit=1000):
    """Fetch historical BTC OHLCV candles with ccxt."""
    exchange = getattr(ccxt, exchange_id)({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
    })
    
    if end_date is None:
        end_date = datetime.now(UTC8).strftime("%Y-%m-%d")
    
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC8).timestamp() * 1000)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC8)
    today = datetime.now(UTC8).date()
    if end_dt.date() == today:
        end_ts = int(datetime.now(UTC8).timestamp() * 1000)
    else:
        end_ts = int(end_dt.timestamp() * 1000)
    
    all_ohlcv = []
    current_ts = start_ts
    
    print(f"Fetching {symbol} {timeframe} data from {exchange_id}...")
    print(f"Date range: {start_date} -> {end_date}")
    
    while current_ts < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, current_ts, limit)
            
            if len(ohlcv) == 0:
                break
            
            all_ohlcv.extend(ohlcv)
            
            current_ts = ohlcv[-1][0] + 1
            
            if end_ts > start_ts:
                progress = min(100, (current_ts - start_ts) / (end_ts - start_ts) * 100)
            else:
                progress = 100.0
            print(f"\rProgress: {progress:.1f}% ({len(all_ohlcv)} candles)", end="")
            
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(5)
    
    print(f"\nDone. Fetched {len(all_ohlcv)} candles.")

    if len(all_ohlcv) == 0:
        df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'datetime'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[df['timestamp'] <= end_ts].drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def save_data(df, filename="btc_ohlcv.csv"):
    """Save data to CSV."""
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)
    df.to_csv(filepath, index=False)
    print(f"Saved data to {filepath}")
    return filepath


def load_data(filename="btc_ohlcv.csv"):
    """Load data from CSV."""
    filepath = os.path.join("data", filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"Loaded {len(df)} rows from {filepath}")
        return df
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical BTC OHLCV data with ccxt")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--start-date", type=str, default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--timeframe", type=str, default="4h", help="Candle timeframe")
    parser.add_argument("--output", type=str, default="btc_ohlcv.csv", help="Output filename")
    args = parser.parse_args()

    df = fetch_btc_ohlcv(
        symbol=args.symbol,
        exchange_id=args.exchange,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe
    )
    save_data(df, filename=args.output)
