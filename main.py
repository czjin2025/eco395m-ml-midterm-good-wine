import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import os

def load_and_standardize_data(wine_type='red'):
    if wine_type == 'red':
        file_path = 'data/winequality-red.csv'
    else:
        file_path = 'data/winequality-white.csv'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path, sep=';')
    
    print(f"\n=== {wine_type.upper()} WINE ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Quality range: {df['quality'].min()} - {df['quality'].max()}")
    
    X = df.drop('quality', axis=1)
    y = df['quality']
    feature_names = X.columns.tolist()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Features standardized: {len(feature_names)} features")
    print(f"Features: {feature_names}")
    
    return X_scaled, y, feature_names

def cross_validate_model(model, X, y, wine_type, model_name, cv_folds=5, n_repeats=1):
    mad_scorer = make_scorer(mean_absolute_error)
    mse_scorer = make_scorer(mean_squared_error)
    
    all_mad = []
    all_mse = []
    all_rmse = []
    
    print(f"\nPerforming {cv_folds}-fold cross validation" + 
          (f" ({n_repeats} repeats)" if n_repeats > 1 else ""))
    
    for repeat in range(n_repeats):
        if n_repeats > 1:
            print(f"  Repeat {repeat + 1}/{n_repeats}")
        
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42 + repeat)
        
        mad_scores = cross_val_score(model, X, y, cv=cv, scoring=mad_scorer)
        mse_scores = cross_val_score(model, X, y, cv=cv, scoring=mse_scorer)
        rmse_scores = np.sqrt(mse_scores)
        
        all_mad.extend(mad_scores)
        all_mse.extend(mse_scores)
        all_rmse.extend(rmse_scores)
    
    avg_mad = np.mean(all_mad)
    avg_mse = np.mean(all_mse)
    avg_rmse = np.mean(all_rmse)
    std_mad = np.std(all_mad)
    
    print(f"\n{model_name} - {wine_type} (CV Results)")
    print(f"MAD:  {avg_mad:.4f} (+/- {std_mad * 2:.4f})")
    print(f"MSE:  {avg_mse:.4f}")
    print(f"RMSE: {avg_rmse:.4f}")
    
    return {
    'model': model_name,
    'dataset': wine_type,
    'MAD': avg_mad,
    'MAD_std': std_mad,
    'MSE': avg_mse,
    'RMSE': avg_rmse, 
    'cv_method': f'{cv_folds}-fold' + (f' x {n_repeats}' if n_repeats > 1 else '')
}

# Cross Validation Functions
def cross_validate_model(model, X, y, wine_type, model_name, cv_folds=5, n_repeats=1):
    # Define scorers
    mad_scorer = make_scorer(mean_absolute_error)
    mse_scorer = make_scorer(mean_squared_error)
    
    all_mad = []
    all_mse = []
    all_rmse = []
    
    print(f"\nPerforming {cv_folds}-fold cross validation" + 
          (f" ({n_repeats} repeats)" if n_repeats > 1 else ""))
    
    for repeat in range(n_repeats):
        if n_repeats > 1:
            print(f"  Repeat {repeat + 1}/{n_repeats}")
        
        # Create KFold with different random state for each repeat
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42 + repeat)
        
        # Calculate scores
        mad_scores = cross_val_score(model, X, y, cv=cv, scoring=mad_scorer)
        mse_scores = cross_val_score(model, X, y, cv=cv, scoring=mse_scorer)
        rmse_scores = np.sqrt(mse_scores)
        
        all_mad.extend(mad_scores)
        all_mse.extend(mse_scores)
        all_rmse.extend(rmse_scores)
    
    # Calculate averages
    avg_mad = np.mean(all_mad)
    avg_mse = np.mean(all_mse)
    avg_rmse = np.mean(all_rmse)
    std_mad = np.std(all_mad)
    
    print(f"\n{model_name} - {wine_type} (CV Results)")
    print(f"MAD:  {avg_mad:.4f} (+/- {std_mad * 2:.4f})")
    print(f"MSE:  {avg_mse:.4f}")
    print(f"RMSE: {avg_rmse:.4f}")
    
    return {
        'model': model_name,
        'dataset': wine_type,
        'MAD': avg_mad,
        'MAD_std': std_mad,
        'MSE': avg_mse,
        'RMSE': avg_rmse,
        'cv_method': f'{cv_folds}-fold' + (f' x {n_repeats}' if n_repeats > 1 else '')
    }

def verify_data():
    print("Verifying data loading and standardization...")
    
    X_red, y_red, features_red = load_and_standardize_data('red')
    print(f"Red wine: {X_red.shape[0]} samples, {X_red.shape[1]} features")
    print(f"Target variable (quality) range: {y_red.min():.0f} - {y_red.max():.0f}")
    
    X_white, y_white, features_white = load_and_standardize_data('white')
    print(f"White wine: {X_white.shape[0]} samples, {X_white.shape[1]} features")
    print(f"Target variable (quality) range: {y_white.min():.0f} - {y_white.max():.0f}")
    
    print("\nAll functions ready to use.")
    return True

if __name__ == "__main__":
    verify_data()