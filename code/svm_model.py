import sys
import os
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import load_and_standardize_data, cross_validate_model
from sklearn.svm import SVR

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

all_results = []

print("Red Wine: SVM with Cross Validation")

X_red, y_red, features_red = load_and_standardize_data('red')

svm_red = SVR(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    epsilon=0.1,
    cache_size=200
)

result_red = cross_validate_model(
    svm_red, X_red, y_red, 
    'Red', 'SVM', 
    cv_folds=5, n_repeats=1
)
all_results.append(result_red)

svm_red.fit(X_red, y_red)

joblib.dump(svm_red, 'models/svm_red.pkl')

print("White Wine: SVM with Cross Validation")

X_white, y_white, features_white = load_and_standardize_data('white')

svm_white = SVR(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    epsilon=0.1,
    cache_size=200
)

result_white = cross_validate_model(
    svm_white, X_white, y_white, 
    'White', 'SVM', 
    cv_folds=5, n_repeats=1
)
all_results.append(result_white)

svm_white.fit(X_white, y_white)

joblib.dump(svm_white, 'models/svm_white.pkl')

print("SVM Cross Validation Results")

results_df = pd.DataFrame(all_results)
print(results_df[['model', 'dataset', 'MAD', 'MAD_std', 'MSE', 'RMSE', 'cv_method']].to_string(index=False))

results_df.to_csv('results/svm_cv_results.csv', index=False)