# src/sentiment_loader.py
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

NEWS_API_KEY = '96d1c8b648d04ccc960b4b6b052580df'
analyzer = SentimentIntensityAnalyzer()

def fetch_and_compute_sentiment(query="gold OR oil OR inflation", days_back=30):
    """
    Fetches past headlines using NewsAPI via direct HTTP request and computes daily compound sentiment scores.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days_back)
    from_date_str = start_date.strftime('%Y-%m-%d')
    
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': query,
        'from': from_date_str,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 100,
        'apiKey': NEWS_API_KEY
    }
    
    print(f"Fetching news articles for query: '{query}' from {from_date_str}...")
    
    all_articles = []
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get('status') != 'ok':
            print(f"NewsAPI Error: {data.get('message', 'Unknown error')}")
            return pd.DataFrame(columns=['sentiment_polarity'])
            
        articles = data.get('articles', [])
        print(f"Successfully retrieved {len(articles)} articles.")
        
        for article in articles:
            pub_date = article.get('publishedAt')
            if not pub_date:
                continue
            
            pub_date = pub_date[:10] # Extract YYYY-MM-DD
            title = article.get('title') or ""
            description = article.get('description') or ""
            text = f"{title} {description}"
            
            # Compute Sentiment Polarity using VADER (-1.0 to 1.0)
            vs = analyzer.polarity_scores(text)
            compound_score = vs['compound']
            
            all_articles.append({
                'Date': pub_date,
                'sentiment_polarity': compound_score
            })
            
    except Exception as e:
        print(f"Error fetching news via request: {e}")
        return pd.DataFrame(columns=['sentiment_polarity'])

    # Convert to DataFrame
    df_sent = pd.DataFrame(all_articles)
    if df_sent.empty:
        print("No articles returned or empty dataset. Creating a neutral fallback sentiment frame.")
        # Create a dummy row for today with 0.0 sentiment so joins don't fail
        dummy_date = datetime.today().strftime('%Y-%m-%d')
        df_daily = pd.DataFrame({'sentiment_polarity': [0.0]}, index=pd.to_datetime([dummy_date]))
        df_daily.index.name = 'Date'
        return df_daily
    
    # Group by Date to get the daily average sentiment score
    df_daily = df_sent.groupby('Date')['sentiment_polarity'].mean().reset_index()
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    df_daily.set_index('Date', inplace=True)
    
    return df_daily

if __name__ == "__main__":
    sentiment_df = fetch_and_compute_sentiment(query="oil OR gold price OR inflation", days_back=30)
    
    os.makedirs("data", exist_ok=True)
    sentiment_df.to_csv("data/daily_sentiment.csv")
    print("Daily sentiment scores saved successfully to data/daily_sentiment.csv!")
    print(sentiment_df.tail())