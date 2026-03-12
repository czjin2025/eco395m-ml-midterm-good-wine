import sys, os, pandas as pd, numpy as np, joblib
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import cross_validate_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def run_linear_regression(wine_type):
    """Run linear regression for given wine type."""
    df = pd.read_csv(f'data/winequality-{wine_type}.csv', sep=';')
    X = df.drop('quality', axis=1)
    y = df['quality']
    features = X.columns.tolist()
    
    print(f"\n=== {wine_type.upper()} WINE ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Quality range: {y.min()} - {y.max()}")
    
    X_scaled = StandardScaler().fit_transform(X)
    lr = LinearRegression()
    
    result = cross_validate_model(lr, X_scaled, y, 
                                  wine_type.capitalize(), 
                                  'Linear Regression', cv_folds=5)
    
    lr.fit(X_scaled, y)
    
    # Save coefficients
    coef_df = pd.DataFrame({'feature': features, 'coefficient': lr.coef_})
    coef_df = coef_df.sort_values('coefficient', key=abs, ascending=False)
    coef_df.to_csv(f'results/linear_coef_{wine_type}.csv', index=False)
    
    # Save model
    joblib.dump(lr, f'models/linear_regression_{wine_type}.pkl')
    
    return result

# Run both
all_results = []
for wine in ['red', 'white']:
    all_results.append(run_linear_regression(wine))

# Summary
results_df = pd.DataFrame(all_results)
print("\n" + "="*60)
print("SUMMARY - CROSS VALIDATION RESULTS")
print(results_df[['model', 'dataset', 'MAD', 'MAD_std', 'MSE', 'RMSE']].to_string(index=False))
results_df.to_csv('results/linear_cv_results.csv', index=False)