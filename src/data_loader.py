# src/data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np
import os

# Define tickers
TICKERS = {
    'Oil': 'CL=F',
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'DXY': 'DX-Y.NYB',
    'SP500': '^GSPC',
    'US10Y': '^TNX'
}

def fetch_market_data(start_date="2018-01-01", end_date=None):
    """Fetches adjusted close prices for commodities and macro indices."""
    data = {}
    for name, ticker in TICKERS.items():
        print(f"Downloading data for {name} ({ticker})...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        # Handle multi-index columns if returned by recent yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close']
        else:
            df = df[['Close']]
        data[name] = df
        
    # Combine into a single DataFrame
    combined_df = pd.DataFrame(index=data['Gold'].index)
    for name, df in data.items():
        combined_df[name] = df[df.columns[0]]
        
    # Forward fill missing values (weekends/holidays) and drop remaining NaNs
    combined_df = combined_df.ffill().dropna()
    return combined_df

if __name__ == "__main__":
    df = fetch_market_data()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_market_data.csv")
    print("\nData fetched successfully and saved to data/raw_market_data.csv!")
    print(df.tail())