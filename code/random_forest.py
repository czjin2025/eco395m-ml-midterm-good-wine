import sys
import os
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import load_and_standardize_data, cross_validate_model
from sklearn.ensemble import RandomForestRegressor

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

all_results = []

print("Red Wine: Random Forest with Cross Validation")

X_red, y_red, features_red = load_and_standardize_data('red')

rf_red = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

result_red = cross_validate_model(
    rf_red, X_red, y_red, 
    'Red', 'Random Forest', 
    cv_folds=5, n_repeats=20
)
all_results.append(result_red)

rf_red.fit(X_red, y_red)

importance_red = pd.DataFrame({
    'feature': features_red,
    'importance': rf_red.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_red.head(5).to_string(index=False))

importance_red.to_csv('results/rf_importance_red.csv', index=False)
print(f"\nFeature importance saved to results/rf_importance_red.csv")

joblib.dump(rf_red, 'models/random_forest_red.pkl')
print(f"Model saved to models/random_forest_red.pkl")

print("White Wine: Random Forest with Cross Validation")

X_white, y_white, features_white = load_and_standardize_data('white')

rf_white = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

result_white = cross_validate_model(
    rf_white, X_white, y_white, 
    'White', 'Random Forest', 
    cv_folds=5, n_repeats=20
)
all_results.append(result_white)

rf_white.fit(X_white, y_white)

importance_white = pd.DataFrame({
    'feature': features_white,
    'importance': rf_white.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 most important features:")
print(importance_white.head(5).to_string(index=False))

importance_white.to_csv('results/rf_importance_white.csv', index=False)

joblib.dump(rf_white, 'models/random_forest_white.pkl')

print("Random Forest Cross Validation Results")

results_df = pd.DataFrame(all_results)
print(results_df[['model', 'dataset', 'MAD', 'MAD_std', 'MSE', 'RMSE', 'cv_method']].to_string(index=False))

results_df.to_csv('results/rf_cv_results.csv', index=False)