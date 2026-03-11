# code/jin_linear_model.py
import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import from main.py
from main import load_and_preprocess_data, evaluate_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Store all results
all_results = []

# 1. RED WINE
print("=" * 60)
print("RED WINE - LINEAR REGRESSION")
print("=" * 60)

# Load red wine data
X_train_r, X_test_r, y_train_r, y_test_r, features_r = load_and_preprocess_data('red')

# Create linear regression object
lr_red = LinearRegression()

# Train the model
lr_red.fit(X_train_r, y_train_r)

# Make predictions
y_pred_r = lr_red.predict(X_test_r)

# Evaluate using our function
result_r = evaluate_model(y_test_r, y_pred_r, 'Linear Regression', 'Red')
all_results.append(result_r)

# The coefficients
print("\nCoefficients:")
coef_df_r = pd.DataFrame({
    'feature': features_r,
    'coefficient': lr_red.coef_
}).sort_values('coefficient', key=abs, ascending=False)
print(coef_df_r.to_string(index=False))

# Optional: statsmodels summary for more statistics
X_train_r_sm = sm.add_constant(X_train_r)
model_r_sm = sm.OLS(y_train_r, X_train_r_sm).fit()
print("\n" + "=" * 40)
print("Red Wine - Model Summary")
print("=" * 40)
print(f"R-squared: {model_r_sm.rsquared:.4f}")
print(f"Adjusted R-squared: {model_r_sm.rsquared_adj:.4f}")
print(f"F-statistic: {model_r_sm.fvalue:.2f}")
print(f"Prob (F-statistic): {model_r_sm.f_pvalue:.4e}")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the model
joblib.dump(lr_red, 'models/linear_regression_red.pkl')
print(f"\nModel saved to models/linear_regression_red.pkl")

# 2. WHITE WINE
print("\n" + "=" * 60)
print("WHITE WINE - LINEAR REGRESSION")
print("=" * 60)

# Load white wine data
X_train_w, X_test_w, y_train_w, y_test_w, features_w = load_and_preprocess_data('white')

# Create linear regression object
lr_white = LinearRegression()

# Train the model
lr_white.fit(X_train_w, y_train_w)

# Make predictions
y_pred_w = lr_white.predict(X_test_w)

# Evaluate using our function
result_w = evaluate_model(y_test_w, y_pred_w, 'Linear Regression', 'White')
all_results.append(result_w)

# The coefficients
print("\nCoefficients:")
coef_df_w = pd.DataFrame({
    'feature': features_w,
    'coefficient': lr_white.coef_
}).sort_values('coefficient', key=abs, ascending=False)
print(coef_df_w.to_string(index=False))

# Optional: statsmodels summary for more statistics
X_train_w_sm = sm.add_constant(X_train_w)
model_w_sm = sm.OLS(y_train_w, X_train_w_sm).fit()
print("\n" + "=" * 40)
print("White Wine - Model Summary")
print("=" * 40)
print(f"R-squared: {model_w_sm.rsquared:.4f}")
print(f"Adjusted R-squared: {model_w_sm.rsquared_adj:.4f}")
print(f"F-statistic: {model_w_sm.fvalue:.2f}")
print(f"Prob (F-statistic): {model_w_sm.f_pvalue:.4e}")

# Save the model
joblib.dump(lr_white, 'models/linear_regression_white.pkl')
print(f"\nModel saved to models/linear_regression_white.pkl")

# 3. SUMMARY
print("\n" + "=" * 60)
print("SUMMARY - ALL RESULTS")
print("=" * 60)

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

# Save results to CSV
# Save results to CSV
results_df.to_csv('results/linear_results.csv', index=False)
print(f"\nResults saved to code/linear_results.csv")