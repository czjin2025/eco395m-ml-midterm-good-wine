import sys
import os
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import load_and_standardize_data, cross_validate_model
from xgboost import XGBRegressor

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

all_results = []

# 1. RED WINE - XGBOOST
print("=" * 60)
print("RED WINE - XGBOOST WITH CROSS VALIDATION")
print("=" * 60)

X_red, y_red, features_red = load_and_standardize_data('red')

xgb_red = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbosity=0
)

print("\nPerforming cross validation...")
result_red = cross_validate_model(xgb_red, X_red, y_red, 'Red', 'XGBoost', cv_folds=5)
all_results.append(result_red)

print("\nTraining on full dataset for feature importance...")
xgb_red.fit(X_red, y_red)

importance_red = pd.DataFrame({
    'feature': features_red,
    'importance': xgb_red.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 most important features:")
print(importance_red.head(5).to_string(index=False))

importance_red.to_csv('results/xgb_importance_red.csv', index=False)
print(f"\nFeature importance saved to results/xgb_importance_red.csv")

joblib.dump(xgb_red, 'models/xgboost_red.pkl')
print(f"Model saved to models/xgboost_red.pkl")

# 2. WHITE WINE - XGBOOST
print("\n" + "=" * 60)
print("WHITE WINE - XGBOOST WITH CROSS VALIDATION")
print("=" * 60)

X_white, y_white, features_white = load_and_standardize_data('white')

xgb_white = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbosity=0
)

print("\nPerforming cross validation...")
result_white = cross_validate_model(xgb_white, X_white, y_white, 'White', 'XGBoost', cv_folds=5)
all_results.append(result_white)

print("\nTraining on full dataset for feature importance...")
xgb_white.fit(X_white, y_white)

importance_white = pd.DataFrame({
    'feature': features_white,
    'importance': xgb_white.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 most important features:")
print(importance_white.head(5).to_string(index=False))

importance_white.to_csv('results/xgb_importance_white.csv', index=False)
print(f"\nFeature importance saved to results/xgb_importance_white.csv")

joblib.dump(xgb_white, 'models/xgboost_white.pkl')
print(f"Model saved to models/xgboost_white.pkl")

# 3. SUMMARY
print("\n" + "=" * 60)
print("SUMMARY - CROSS VALIDATION RESULTS")
print("=" * 60)

results_df = pd.DataFrame(all_results)
print(results_df[['model', 'dataset', 'MAD', 'MAD_std', 'MSE', 'RMSE', 'cv_method']].to_string(index=False))

results_df.to_csv('results/xgboost_cv_results.csv', index=False)
print(f"\nResults saved to results/xgboost_cv_results.csv")