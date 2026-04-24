"""
Simple Web App — USA Population Growth Rate Predictor
======================================================
Run this to get predictions from the trained model.

Usage:
    python webapp/predict.py --year 2025
"""

import pickle
import numpy as np
import argparse
import os

def load_model(path="models/best_model.pkl"):
    if not os.path.exists(path):
        print(f"❌ Model not found at '{path}'")
        print("   Please run train_model.py first to generate the model.")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_growth(year, lag1=0.377, lag2=0.156, lag3=0.964,
                   roll3=0.499, roll5=0.496):
    """
    Predict population growth rate for a given year.
    Default lag/rolling values are based on 2020–2022 actual data.
    """
    model = load_model()
    if model is None:
        return

    X = np.array([[year, lag1, lag2, lag3, roll3, roll5]])
    pred = model.predict(X)[0]

    print(f"\n🔮 Prediction for {year}:")
    print(f"   Estimated US Population Growth Rate = {pred:.4f}%")
    print()

    if pred > 1.0:
        trend = "📈 HIGH growth — above historical average"
    elif pred > 0.5:
        trend = "➡️  MODERATE growth — near recent trend"
    else:
        trend = "📉 LOW growth — below recent trend"

    print(f"   Trend: {trend}")
    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict US Population Growth Rate")
    parser.add_argument("--year", type=int, default=2025, help="Year to predict (e.g. 2025)")
    args = parser.parse_args()
    predict_growth(args.year)
