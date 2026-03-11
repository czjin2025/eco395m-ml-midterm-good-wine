import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import cross_validate_model from main.py instead of load_and_preprocess_data and evaluate_model
from main import cross_validate_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# Create folders if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Store all results
all_results = []

# 1. RED WINE - XGBOOST WITH CROSS VALIDATION
print("=" * 60)
print("RED WINE - XGBOOST WITH CROSS VALIDATION")
print("=" * 60)

# Read data directly, don't use load_and_preprocess_data
df_red = pd.read_csv('data/winequality-red.csv', sep=';')
X_red = df_red.drop('quality', axis=1)
y_red = df_red['quality']
features_red = X_red.columns.tolist()

print(f"\n=== RED WINE ===")
print(f"Dataset shape: {df_red.shape}")
print(f"Quality range: {y_red.min()} - {y_red.max()}")
print(f"Features: {features_red}")

# Standardize features manually
scaler_red = StandardScaler()
X_red_scaled = scaler_red.fit_transform(X_red)

# Create XGBoost regressor
xgb_red = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbosity=0
)

# Use cross validation function (5-fold, 1 repeat for now)
print("\nPerforming cross validation...")
result_red = cross_validate_model(xgb_red, X_red_scaled, y_red, 'Red', 'XGBoost', cv_folds=5, n_repeats=1)
all_results.append(result_red)

# Train on full data for feature importance and model saving
print("\nTraining on full dataset for feature importance...")
xgb_red.fit(X_red_scaled, y_red)

# Feature importance
importance_red = pd.DataFrame({
    'feature': features_red,
    'importance': xgb_red.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 most important features (trained on full dataset):")
print(importance_red.head(5).to_string(index=False))

# Save feature importance
importance_red.to_csv('results/xgb_importance_red.csv', index=False)
print(f"\nFeature importance saved to results/xgb_importance_red.csv")

# Save the model
joblib.dump(xgb_red, 'models/xgboost_red.pkl')
print(f"Model saved to models/xgboost_red.pkl")

# 2. WHITE WINE - XGBOOST WITH CROSS VALIDATION
print("\n" + "=" * 60)
print("WHITE WINE - XGBOOST WITH CROSS VALIDATION")
print("=" * 60)

# Read data directly
df_white = pd.read_csv('data/winequality-white.csv', sep=';')
X_white = df_white.drop('quality', axis=1)
y_white = df_white['quality']
features_white = X_white.columns.tolist()

print(f"\n=== WHITE WINE ===")
print(f"Dataset shape: {df_white.shape}")
print(f"Quality range: {y_white.min()} - {y_white.max()}")
print(f"Features: {features_white}")

# Standardize features manually
scaler_white = StandardScaler()
X_white_scaled = scaler_white.fit_transform(X_white)

# Create XGBoost regressor
xgb_white = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbosity=0
)

# Use cross validation function
print("\nPerforming cross validation...")
result_white = cross_validate_model(xgb_white, X_white_scaled, y_white, 'White', 'XGBoost', cv_folds=5, n_repeats=1)
all_results.append(result_white)

# Train on full data for feature importance and model saving
print("\nTraining on full dataset for feature importance...")
xgb_white.fit(X_white_scaled, y_white)

# Feature importance
importance_white = pd.DataFrame({
    'feature': features_white,
    'importance': xgb_white.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 most important features (trained on full dataset):")
print(importance_white.head(5).to_string(index=False))

# Save feature importance
importance_white.to_csv('results/xgb_importance_white.csv', index=False)
print(f"\nFeature importance saved to results/xgb_importance_white.csv")

# Save the model
joblib.dump(xgb_white, 'models/xgboost_white.pkl')
print(f"Model saved to models/xgboost_white.pkl")

# 3. SUMMARY
print("\n" + "=" * 60)
print("SUMMARY - CROSS VALIDATION RESULTS")
print("=" * 60)

results_df = pd.DataFrame(all_results)
# Display results with standard deviation
print(results_df[['model', 'dataset', 'MAD', 'MAD_std', 'MSE', 'RMSE', 'cv_method']].to_string(index=False))

# Save results to CSV
results_df.to_csv('results/xgboost_cv_results.csv', index=False)
print(f"\nResults saved to results/xgboost_cv_results.csv")