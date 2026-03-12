import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import from main.py
from main import load_and_standardize_data, cross_validate_model
from sklearn.svm import SVR

# Create folders
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Store all results
all_results = []

# ==================================================
# 1. RED WINE - SVM
# ==================================================
print("=" * 60)
print("RED WINE - SVM WITH CROSS VALIDATION")
print("=" * 60)

# Load and standardize red wine data
X_red, y_red, features_red = load_and_standardize_data('red')

# Create SVM model with RBF kernel
svm_red = SVR(
    kernel='rbf',      # Radial basis function kernel
    C=1.0,             # Regularization parameter
    gamma='scale',     # Kernel coefficient
    epsilon=0.1,       # Epsilon in the epsilon-SVR model
    cache_size=200     # Cache size for kernel computations
)

# Perform cross validation
print("\nPerforming 5-fold cross validation...")
result_red = cross_validate_model(
    svm_red, X_red, y_red, 
    'Red', 'SVM', 
    cv_folds=5, n_repeats=1
)
all_results.append(result_red)

# Train on full dataset for model saving
print("\nTraining on full dataset...")
svm_red.fit(X_red, y_red)

# Note: SVM doesn't have built-in feature importance like tree models
# We save the model for potential future use

# Save model
joblib.dump(svm_red, 'models/svm_red.pkl')
print(f"Model saved to models/svm_red.pkl")

# ==================================================
# 2. WHITE WINE - SVM
# ==================================================
print("\n" + "=" * 60)
print("WHITE WINE - SVM WITH CROSS VALIDATION")
print("=" * 60)

# Load and standardize white wine data
X_white, y_white, features_white = load_and_standardize_data('white')

# Create SVM model
svm_white = SVR(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    epsilon=0.1,
    cache_size=200
)

# Perform cross validation
print("\nPerforming 5-fold cross validation...")
result_white = cross_validate_model(
    svm_white, X_white, y_white, 
    'White', 'SVM', 
    cv_folds=5, n_repeats=1
)
all_results.append(result_white)

# Train on full dataset for model saving
print("\nTraining on full dataset...")
svm_white.fit(X_white, y_white)

# Save model
joblib.dump(svm_white, 'models/svm_white.pkl')
print(f"Model saved to models/svm_white.pkl")

# ==================================================
# 3. SUMMARY
# ==================================================
print("\n" + "=" * 60)
print("SUMMARY - SVM CROSS VALIDATION RESULTS")
print("=" * 60)

results_df = pd.DataFrame(all_results)
print(results_df[['model', 'dataset', 'MAD', 'MAD_std', 'MSE', 'RMSE', 'cv_method']].to_string(index=False))

# Save results
results_df.to_csv('results/svm_cv_results.csv', index=False)
print(f"\nResults saved to results/svm_cv_results.csv")

print("\n" + "=" * 60)
print("SVM MODEL COMPLETED SUCCESSFULLY!")
print("=" * 60)