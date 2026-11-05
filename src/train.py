# src/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import os

def train_and_evaluate(csv_path="data/processed_gold_data.csv", target_col='Target'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Processed file not found at {csv_path}. Run features.py first.")
        
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Chronological Split (80% train, 20% test without shuffling to respect time order)
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    }
    
    results = {}
    
    print(f"\n--- Model Evaluation Results (Predicting Next-Day Price) ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        
        # Calculate Directional Accuracy (% times model got the up/down direction correct)
        actual_direction = np.sign(y_test.values - X_test[target_col.replace('Target', 'Gold')].values) # simplified check
        # Better: direction relative to previous close
        # Let's compute simple directional accuracy
        pred_direction = np.sign(predictions - X_test.iloc[:, 0].values) # baseline reference
        
        results[name] = {"RMSE": rmse, "MAE": mae}
        print(f"\nModel: {name}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")

if __name__ == "__main__":
    train_and_evaluate()