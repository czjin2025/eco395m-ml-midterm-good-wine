import sys
import os
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import load_and_standardize_data, cross_validate_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

all_results = []

# 1. RED WINE - LINEAR REGRESSION
print("=" * 60)
print("RED WINE - LINEAR REGRESSION WITH CROSS VALIDATION")
print("=" * 60)

X_red, y_red, features_red = load_and_standardize_data('red')

lr_red = LinearRegression()

result_red = cross_validate_model(lr_red, X_red, y_red, 'Red', 'Linear Regression', cv_folds=5, n_repeats=1)
all_results.append(result_red)

lr_red.fit(X_red, y_red)

print("\nCoefficients (trained on full dataset):")
coef_df_red = pd.DataFrame({
    'feature': features_red,
    'coefficient': lr_red.coef_
}).sort_values('coefficient', key=abs, ascending=False)
print(coef_df_red.head(10).to_string(index=False))

coef_df_red.to_csv('results/linear_coef_red.csv', index=False)
print(f"\nCoefficients saved to results/linear_coef_red.csv")

X_red_sm = sm.add_constant(X_red)
model_red_sm = sm.OLS(y_red, X_red_sm).fit()
print("\n" + "=" * 40)
print("Red Wine - Model Summary (Full Dataset)")
print("=" * 40)
print(f"R-squared: {model_red_sm.rsquared:.4f}")
print(f"Adjusted R-squared: {model_red_sm.rsquared_adj:.4f}")
print(f"F-statistic: {model_red_sm.fvalue:.2f}")
print(f"Prob (F-statistic): {model_red_sm.f_pvalue:.4e}")

joblib.dump(lr_red, 'models/linear_regression_red.pkl')
print(f"\nModel saved to models/linear_regression_red.pkl")

# 2. WHITE WINE - LINEAR REGRESSION
print("\n" + "=" * 60)
print("WHITE WINE - LINEAR REGRESSION WITH CROSS VALIDATION")
print("=" * 60)

X_white, y_white, features_white = load_and_standardize_data('white')

lr_white = LinearRegression()

result_white = cross_validate_model(lr_white, X_white, y_white, 'White', 'Linear Regression', cv_folds=5, n_repeats=1)
all_results.append(result_white)

lr_white.fit(X_white, y_white)

print("\nCoefficients (trained on full dataset):")
coef_df_white = pd.DataFrame({
    'feature': features_white,
    'coefficient': lr_white.coef_
}).sort_values('coefficient', key=abs, ascending=False)
print(coef_df_white.head(10).to_string(index=False))

coef_df_white.to_csv('results/linear_coef_white.csv', index=False)
print(f"\nCoefficients saved to results/linear_coef_white.csv")

X_white_sm = sm.add_constant(X_white)
model_white_sm = sm.OLS(y_white, X_white_sm).fit()
print("\n" + "=" * 40)
print("White Wine - Model Summary (Full Dataset)")
print("=" * 40)
print(f"R-squared: {model_white_sm.rsquared:.4f}")
print(f"Adjusted R-squared: {model_white_sm.rsquared_adj:.4f}")
print(f"F-statistic: {model_white_sm.fvalue:.2f}")
print(f"Prob (F-statistic): {model_white_sm.f_pvalue:.4e}")

joblib.dump(lr_white, 'models/linear_regression_white.pkl')
print(f"\nModel saved to models/linear_regression_white.pkl")

print("\n" + "=" * 60)
print("SUMMARY - CROSS VALIDATION RESULTS")
print("=" * 60)

results_df = pd.DataFrame(all_results)
print(results_df[['model', 'dataset', 'MAD', 'MAD_std', 'MSE', 'RMSE', 'cv_method']].to_string(index=False))

results_df.to_csv('results/linear_cv_results.csv', index=False)
print(f"\nResults saved to results/linear_cv_results.csv")