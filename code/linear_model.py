import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# import from main.py
from main import cross_validate_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Create folders if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Store all results
all_results = []

# 1. RED WINE - LINEAR REGRESSION WITH CROSS VALIDATION
print("=" * 60)
print("RED WINE - LINEAR REGRESSION WITH CROSS VALIDATION")
print("=" * 60)

# Directly read data
df_red = pd.read_csv('data/winequality-red.csv', sep=';')
X_red = df_red.drop('quality', axis=1)
y_red = df_red['quality']
features_red = X_red.columns.tolist()

print(f"\n=== RED WINE ===")
print(f"Dataset shape: {df_red.shape}")
print(f"Quality range: {y_red.min()} - {y_red.max()}")
print(f"Features: {features_red}")

# Standardized features
scaler_red = StandardScaler()
X_red_scaled = scaler_red.fit_transform(X_red)

# Create linear regression model
lr_red = LinearRegression()

# Use the cross-validation function(5-fold, 1 repeat for now)
result_red = cross_validate_model(lr_red, X_red_scaled, y_red, 'Red', 'Linear Regression', cv_folds=5, n_repeats=1)
all_results.append(result_red)

# Train the complete dataset for feature coefficients and model saving.
lr_red.fit(X_red_scaled, y_red)

# The coefficients
print("\nCoefficients (trained on full dataset):")
coef_df_red = pd.DataFrame({
    'feature': features_red,
    'coefficient': lr_red.coef_
}).sort_values('coefficient', key=abs, ascending=False)
print(coef_df_red.head(10).to_string(index=False))

# Save coefficients
coef_df_red.to_csv('results/linear_coef_red.csv', index=False)
print(f"\nCoefficients saved to results/linear_coef_red.csv")

# Optional: statsmodels summary for more statistics
X_red_sm = sm.add_constant(X_red_scaled)
model_red_sm = sm.OLS(y_red, X_red_sm).fit()
print("\n" + "=" * 40)
print("Red Wine - Model Summary (Full Dataset)")
print("=" * 40)
print(f"R-squared: {model_red_sm.rsquared:.4f}")
print(f"Adjusted R-squared: {model_red_sm.rsquared_adj:.4f}")
print(f"F-statistic: {model_red_sm.fvalue:.2f}")
print(f"Prob (F-statistic): {model_red_sm.f_pvalue:.4e}")

# Save the model
joblib.dump(lr_red, 'models/linear_regression_red.pkl')
print(f"\nModel saved to models/linear_regression_red.pkl")

# 2. WHITE WINE - LINEAR REGRESSION WITH CROSS VALIDATION
print("\n" + "=" * 60)
print("WHITE WINE - LINEAR REGRESSION WITH CROSS VALIDATION")
print("=" * 60)

# Directly read data
df_white = pd.read_csv('data/winequality-white.csv', sep=';')
X_white = df_white.drop('quality', axis=1)
y_white = df_white['quality']
features_white = X_white.columns.tolist()

print(f"\n=== WHITE WINE ===")
print(f"Dataset shape: {df_white.shape}")
print(f"Quality range: {y_white.min()} - {y_white.max()}")
print(f"Features: {features_white}")

# Standardized features
scaler_white = StandardScaler()
X_white_scaled = scaler_white.fit_transform(X_white)

# Create linear regression model
lr_white = LinearRegression()

# Use the cross-validation function
result_white = cross_validate_model(lr_white, X_white_scaled, y_white, 'White', 'Linear Regression', cv_folds=5, n_repeats=1)
all_results.append(result_white)

# Train the complete dataset for feature coefficients and model saving.
lr_white.fit(X_white_scaled, y_white)

# The coefficients
print("\nCoefficients (trained on full dataset):")
coef_df_white = pd.DataFrame({
    'feature': features_white,
    'coefficient': lr_white.coef_
}).sort_values('coefficient', key=abs, ascending=False)
print(coef_df_white.head(10).to_string(index=False))

# Save coefficients
coef_df_white.to_csv('results/linear_coef_white.csv', index=False)
print(f"\nCoefficients saved to results/linear_coef_white.csv")

# Optional: statsmodels summary for more statistics
X_white_sm = sm.add_constant(X_white_scaled)
model_white_sm = sm.OLS(y_white, X_white_sm).fit()
print("\n" + "=" * 40)
print("White Wine - Model Summary (Full Dataset)")
print("=" * 40)
print(f"R-squared: {model_white_sm.rsquared:.4f}")
print(f"Adjusted R-squared: {model_white_sm.rsquared_adj:.4f}")
print(f"F-statistic: {model_white_sm.fvalue:.2f}")
print(f"Prob (F-statistic): {model_white_sm.f_pvalue:.4e}")

# Save the model
joblib.dump(lr_white, 'models/linear_regression_white.pkl')
print(f"\nModel saved to models/linear_regression_white.pkl")

# 3. SUMMARY
print("\n" + "=" * 60)
print("SUMMARY - CROSS VALIDATION RESULTS")
print("=" * 60)

results_df = pd.DataFrame(all_results)
print(results_df[['model', 'dataset', 'MAD', 'MAD_std', 'MSE', 'RMSE', 'cv_method']].to_string(index=False))

# Save results to CSV
results_df.to_csv('results/linear_cv_results.csv', index=False)
print(f"\nResults saved to results/linear_cv_results.csv")