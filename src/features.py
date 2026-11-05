# src/features.py
import pandas as pd
import numpy as np
import os

def engineer_features(df, sentiment_csv="data/daily_sentiment.csv", target_commodity='Gold'):
    """
    Adds lag features, moving averages, percentage returns, 
    and news sentiment polarity to the market dataset.
    """
    df = df.copy()
    
    # 1. Merge Sentiment Data if available
    if sentiment_csv and os.path.exists(sentiment_csv):
        sent_df = pd.read_csv(sentiment_csv, index_col=0, parse_dates=True)
        df = df.join(sent_df, how='left')
        # Fill missing news days with neutral sentiment (0.0)
        df['sentiment_polarity'] = df['sentiment_polarity'].fillna(0.0)
    else:
        df['sentiment_polarity'] = 0.0
        
    # 2. Compute Daily Returns (%) for all market columns
    for col in ['Oil', 'Gold', 'Silver', 'DXY', 'SP500', 'US10Y']:
        if col in df.columns:
            df[f'{col}_ret'] = df[col].pct_change()
            
    # 3. Moving Averages for the target commodity
    if target_commodity in df.columns:
        df[f'{target_commodity}_SMA_7'] = df[target_commodity].rolling(window=7).mean()
        df[f'{target_commodity}_SMA_30'] = df[target_commodity].rolling(window=30).mean()
        
        # 4. Lag Features (Past 1, 2, and 5 days prices for target commodity)
        for lag in [1, 2, 5]:
            df[f'{target_commodity}_lag_{lag}'] = df[target_commodity].shift(lag)
            
        # 5. Target Variable: Next day's close price
        df['Target'] = df[target_commodity].shift(-1)
        
    # Drop rows with NaNs resulting from shifts, rolling windows, or joins
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    market_data_path = "data/raw_market_data.csv"
    sentiment_data_path = "data/daily_sentiment.csv"
    
    if not os.path.exists(market_data_path):
        raise FileNotFoundError(f"Raw market data not found at {market_data_path}. Run data_loader.py first.")
        
    raw_df = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
    processed_df = engineer_features(raw_df, sentiment_csv=sentiment_data_path, target_commodity='Gold')
    
    os.makedirs("data", exist_ok=True)
    processed_df.to_csv("data/processed_gold_data.csv")
    print("Feature engineering with sentiment complete! Saved to data/processed_gold_data.csv.")
    print(processed_df.tail())