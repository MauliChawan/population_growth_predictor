"""
USA Population Growth Rate Predictor
=====================================
This script trains a machine learning model to predict
future US population growth rates based on historical data (1961-2022).

Dataset: FRED (Federal Reserve Economic Data)
Target: SPPOPGROWUSA — Annual population growth rate (%)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pickle
import os

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  USA Population Growth Rate Predictor")
print("=" * 60)

df = pd.read_csv("data/Population-Growth.csv")
print(f"\n✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

# ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────
# Extract the year from the DATE column
df['Year'] = pd.to_datetime(df['DATE']).dt.year

# Create lag features (previous years' growth rates help predict the next one)
df['Lag_1'] = df['SPPOPGROWUSA'].shift(1)   # 1 year ago
df['Lag_2'] = df['SPPOPGROWUSA'].shift(2)   # 2 years ago
df['Lag_3'] = df['SPPOPGROWUSA'].shift(3)   # 3 years ago

# Rolling average (trend over last 3 and 5 years)
df['Rolling_3yr_avg'] = df['SPPOPGROWUSA'].rolling(window=3).mean()
df['Rolling_5yr_avg'] = df['SPPOPGROWUSA'].rolling(window=5).mean()

# Drop rows with NaN values (caused by lag/rolling features)
df = df.dropna()

print(f"\n✅ Features created. Working dataset: {df.shape[0]} rows")

# ── 3. DEFINE FEATURES & TARGET ──────────────────────────────────────────────
features = ['Year', 'Lag_1', 'Lag_2', 'Lag_3', 'Rolling_3yr_avg', 'Rolling_5yr_avg']
target   = 'SPPOPGROWUSA'

X = df[features]
y = df[target]

print(f"\n📊 Features used: {features}")
print(f"🎯 Target: {target}")

# ── 4. TRAIN / TEST SPLIT ────────────────────────────────────────────────────
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False  # shuffle=False keeps time order
)
print(f"\n📦 Training samples : {len(X_train)}")
print(f"📦 Testing  samples : {len(X_test)}")

# ── 5. TRAIN MODELS ───────────────────────────────────────────────────────────
models = {
    "Linear Regression"      : LinearRegression(),
    "Random Forest"          : RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting"      : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
print("\n" + "─" * 60)
print("  MODEL TRAINING & EVALUATION")
print("─" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    results[name] = {"model": model, "MAE": mae, "RMSE": rmse, "R2": r2, "predictions": y_pred}

    print(f"\n🤖 {name}")
    print(f"   MAE  = {mae:.4f}  (lower is better)")
    print(f"   RMSE = {rmse:.4f}  (lower is better)")
    print(f"   R²   = {r2:.4f}  (closer to 1 is better)")

# ── 6. PICK THE BEST MODEL ───────────────────────────────────────────────────
best_name  = min(results, key=lambda k: results[k]["RMSE"])
best_model = results[best_name]["model"]
print(f"\n🏆 Best model: {best_name} (lowest RMSE)")

# ── 7. SAVE BEST MODEL ───────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
model_path = "models/best_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)
print(f"\n💾 Model saved to: {model_path}")

# ── 8. VISUALISATIONS ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("USA Population Growth Rate — ML Analysis", fontsize=15, fontweight="bold")

# Plot 1: Historical data
ax1 = axes[0, 0]
ax1.plot(df['Year'], df['SPPOPGROWUSA'], color='steelblue', linewidth=2, marker='o', markersize=4)
ax1.set_title("Historical Population Growth Rate (1961–2022)")
ax1.set_xlabel("Year")
ax1.set_ylabel("Growth Rate (%)")
ax1.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted (best model)
ax2 = axes[0, 1]
y_pred_best = results[best_name]["predictions"]
ax2.plot(y_test.values, label="Actual",    color='steelblue',  linewidth=2)
ax2.plot(y_pred_best,   label="Predicted", color='orangered', linewidth=2, linestyle='--')
ax2.set_title(f"Actual vs Predicted ({best_name})")
ax2.set_xlabel("Test Sample Index")
ax2.set_ylabel("Growth Rate (%)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Model Comparison (R² scores)
ax3 = axes[1, 0]
model_names = list(results.keys())
r2_scores   = [results[k]["R2"] for k in model_names]
colors = ['#4CAF50' if k == best_name else '#90CAF9' for k in model_names]
bars = ax3.bar(model_names, r2_scores, color=colors, edgecolor='white', linewidth=1.2)
ax3.set_title("R² Score Comparison (higher = better)")
ax3.set_ylabel("R² Score")
ax3.set_ylim(0, 1)
for bar, val in zip(bars, r2_scores):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.3f}", ha='center', va='bottom', fontweight='bold')
ax3.tick_params(axis='x', rotation=10)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Future Forecast (next 10 years)
ax4 = axes[1, 1]
future_years = list(range(2023, 2033))
last_known   = df['SPPOPGROWUSA'].values
forecasts    = []
lag1, lag2, lag3 = last_known[-1], last_known[-2], last_known[-3]
roll3 = np.mean(last_known[-3:])
roll5 = np.mean(last_known[-5:])

for yr in future_years:
    inp     = np.array([[yr, lag1, lag2, lag3, roll3, roll5]])
    pred    = best_model.predict(inp)[0]
    forecasts.append(pred)
    lag3, lag2, lag1 = lag2, lag1, pred
    roll3 = np.mean(forecasts[-3:]) if len(forecasts) >= 3 else np.mean(forecasts)
    roll5 = np.mean(forecasts[-5:]) if len(forecasts) >= 5 else np.mean(forecasts)

ax4.plot(df['Year'].values[-15:], last_known[-15:], color='steelblue',  linewidth=2, label="Historical")
ax4.plot(future_years,            forecasts,         color='orangered', linewidth=2, linestyle='--', marker='o', markersize=5, label="Forecast (2023–2032)")
ax4.axvline(x=2022, color='gray', linestyle=':', linewidth=1.5, label="Forecast Start")
ax4.set_title("10-Year Population Growth Forecast")
ax4.set_xlabel("Year")
ax4.set_ylabel("Growth Rate (%)")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("models/analysis_plots.png", dpi=150, bbox_inches='tight')
print("📊 Plots saved to: models/analysis_plots.png")
plt.show()

# ── 9. FUTURE FORECAST TABLE ─────────────────────────────────────────────────
print("\n" + "─" * 40)
print("  📅 FUTURE FORECAST (2023–2032)")
print("─" * 40)
for yr, fc in zip(future_years, forecasts):
    print(f"  {yr}  →  {fc:.4f} %")

print("\n✅ Training complete!")
