# src/evaluate_plot.py
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import os

def plot_predictions():
    df = pd.read_csv("data/processed_gold_data.csv", index_col=0, parse_dates=True)
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Plotting
    os.makedirs("notebooks", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual Price', color='black', alpha=0.8)
    plt.plot(y_test.index, preds, label='XGBoost Predicted Price', color='blue', linestyle='--')
    plt.title("Gold Price Prediction vs Actual Market Data (Baseline)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.savefig("notebooks/prediction_plot.png")
    print("Plot saved successfully to notebooks/prediction_plot.png!")

if __name__ == "__main__":
    plot_predictions()