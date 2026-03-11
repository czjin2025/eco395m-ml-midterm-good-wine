# code/jin_xgboost_model.py
import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import from main.py
from main import load_and_preprocess_data, evaluate_model
from xgboost import XGBRegressor

# Create models folder if it doesn't exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Store all results
all_results = []

# 1. RED WINE
print("=" * 60)
print("RED WINE - XGBOOST REGRESSION")
print("=" * 60)

# Load red wine data
X_train_r, X_test_r, y_train_r, y_test_r, features_r = load_and_preprocess_data('red')

# Create XGBoost regressor
xgb_red = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbosity=0
)

# Train the model
print("\nTraining XGBoost model...")
xgb_red.fit(X_train_r, y_train_r)

# Make predictions
y_pred_r = xgb_red.predict(X_test_r)

# Evaluate using our function
result_r = evaluate_model(y_test_r, y_pred_r, 'XGBoost', 'Red')
all_results.append(result_r)

# Feature importance
importance_r = pd.DataFrame({
    'feature': features_r,
    'importance': xgb_red.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 most important features:")
print(importance_r.head(5).to_string(index=False))

# Save feature importance
importance_r.to_csv('results/xgb_importance_red.csv', index=False)
print(f"\nFeature importance saved to results/xgb_importance_red.csv")

# Save the model
joblib.dump(xgb_red, 'models/xgboost_red.pkl')
print(f"Model saved to models/xgboost_red.pkl")


# 2. WHITE WINE
print("\n" + "=" * 60)
print("WHITE WINE - XGBOOST REGRESSION")
print("=" * 60)

# Load white wine data
X_train_w, X_test_w, y_train_w, y_test_w, features_w = load_and_preprocess_data('white')

# Create XGBoost regressor
xgb_white = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbosity=0
)

# Train the model
print("\nTraining XGBoost model...")
xgb_white.fit(X_train_w, y_train_w)

# Make predictions
y_pred_w = xgb_white.predict(X_test_w)

# Evaluate using our function
result_w = evaluate_model(y_test_w, y_pred_w, 'XGBoost', 'White')
all_results.append(result_w)

# Feature importance
importance_w = pd.DataFrame({
    'feature': features_w,
    'importance': xgb_white.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 most important features:")
print(importance_w.head(5).to_string(index=False))

# Save feature importance
importance_w.to_csv('results/xgb_importance_white.csv', index=False)
print(f"\nFeature importance saved to results/xgb_importance_white.csv")

# Save the model
joblib.dump(xgb_white, 'models/xgboost_white.pkl')
print(f"Model saved to models/xgboost_white.pkl")

# 3. SUMMARY
print("\n" + "=" * 60)
print("SUMMARY - ALL RESULTS")
print("=" * 60)

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv('results/xgboost_results.csv', index=False)
print(f"\nResults saved to results/xgboost_results.csv")