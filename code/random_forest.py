import sys
import os
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import cross_validate_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

all_results = []

print("=" * 60)
print("=" * 60)

df_red = pd.read_csv('data/winequality-red.csv', sep=';')
X_red = df_red.drop('quality', axis=1)
y_red = df_red['quality']
features_red = X_red.columns.tolist()

print(f"Dataset shape: {df_red.shape}")
print(f"Quality range: {y_red.min()} - {y_red.max()}")
print(f"Features: {features_red}")

scaler_red = StandardScaler()
X_red_scaled = scaler_red.fit_transform(X_red)

rf_red = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\n5-fold cross validation")

result_red = cross_validate_model(
    rf_red, X_red_scaled, y_red, 
    'Red', 'Random Forest', 
    cv_folds=5, n_repeats=1
)
all_results.append(result_red)

print("\nTrain")
rf_red.fit(X_red_scaled, y_red)

importance_red = pd.DataFrame({
    'feature': features_red,
    'importance': rf_red.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 most important features:")
print(importance_red.head(5).to_string(index=False))

importance_red.to_csv('results/rf_importance_red.csv', index=False)
print(f"\nFeature importance saved to results/rf_importance_red.csv")

joblib.dump(rf_red, 'models/random_forest_red.pkl')
print(f"Model saved to models/random_forest_red.pkl")

print("\n" + "=" * 60)
print("White Wine: Random Forest with Cross Validation”)
print("=" * 60)

df_white = pd.read_csv('data/winequality-white.csv', sep=';')
X_white = df_white.drop('quality', axis=1)
y_white = df_white['quality']
features_white = X_white.columns.tolist()

print(f"\nWhite Wine”)
print(f"Dataset shape: {df_white.shape}")
print(f"Quality range: {y_white.min()} - {y_white.max()}")
print(f"Features: {features_white}")

scaler_white = StandardScaler()
X_white_scaled = scaler_white.fit_transform(X_white)

rf_white = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\n5-fold cross validation")

result_white = cross_validate_model(
    rf_white, X_white_scaled, y_white, 
    'White', 'Random Forest', 
    cv_folds=5, n_repeats=1
)
all_results.append(result_white)

print("\nTrain")
rf_white.fit(X_white_scaled, y_white)

importance_white = pd.DataFrame({
    'feature': features_white,
    'importance': rf_white.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 most important features:")
print(importance_white.head(5).to_string(index=False))

importance_white.to_csv('results/rf_importance_white.csv', index=False)
print(f"\nFeature importance saved to results/rf_importance_white.csv")


joblib.dump(rf_white, 'models/random_forest_white.pkl')
print(f"Model saved to models/random_forest_white.pkl")

results_df = pd.DataFrame(all_results)

print(results_df[['model', 'dataset', 'MAD', 'MAD_std', 'MSE', 'RMSE', 'cv_method']].to_string(index=False))


results_df.to_csv('results/rf_cv_results.csv', index=False)
print(f"\nResults saved to results/rf_cv_results.csv")