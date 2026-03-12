import sys, os, pandas as pd, numpy as np, joblib
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import load_and_standardize_data, cross_validate_model
from xgboost import XGBRegressor

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def run_xgboost(wine_type):
    """Run XGBoost for given wine type."""
    X, y, features = load_and_standardize_data(wine_type)
    
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, 
                       max_depth=5, random_state=42, verbosity=0)
    
    print(f"\n=== {wine_type.upper()} WINE ===")
    result = cross_validate_model(xgb, X, y, wine_type.capitalize(), 
                                  'XGBoost', cv_folds=5)
    
    xgb.fit(X, y)
    
    # Feature importance
    imp = pd.DataFrame({'feature': features, 'importance': xgb.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    imp.to_csv(f'results/xgb_importance_{wine_type}.csv', index=False)
    
    joblib.dump(xgb, f'models/xgboost_{wine_type}.pkl')
    
    return result

all_results = [run_xgboost(wine) for wine in ['red', 'white']]

# Summary
results_df = pd.DataFrame(all_results)
print("\n" + "="*60)
print("SUMMARY - XGBOOST RESULTS")
print(results_df[['model', 'dataset', 'MAD', 'MAD_std', 'MSE', 'RMSE']].to_string(index=False))
results_df.to_csv('results/xgboost_cv_results.csv', index=False)