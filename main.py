import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def load_and_preprocess_data(wine_type='red', test_size=0.2, random_state=42):
    """
    Load and preprocess wine quality data.
    
    Parameters:
        wine_type: str, 'red' or 'white'
        test_size: float, proportion of test set (default 0.2)
        random_state: int, random seed for reproducibility
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    """
    # Set file path based on wine type
    if wine_type == 'red':
        file_path = 'data/winequality-red.csv'
    else:
        file_path = 'data/winequality-white.csv'
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load data
    df = pd.read_csv(file_path, sep=';')
    
    print(f"\n=== {wine_type.upper()} WINE ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Quality range: {df['quality'].min()} - {df['quality'].max()}")
    
    # Separate features and target
    X = df.drop('quality', axis=1)
    y = df['quality']
    feature_names = X.columns.tolist()
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {feature_names}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

def evaluate_model(y_true, y_pred, model_name, dataset_name, verbose=True):
    """
    Calculate evaluation metrics: MAD, MSE, RMSE.
    
    Parameters:
        y_true: actual values
        y_pred: predicted values
        model_name: name of the model
        dataset_name: 'Red' or 'White'
        verbose: whether to print results
    
    Returns:
        dict with MAD, MSE, RMSE
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    if verbose:
        print(f"\n{model_name} - {dataset_name}")
        print(f"MAD: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
    
    return {
        'model': model_name,
        'dataset': dataset_name,
        'MAD': mae,
        'MSE': mse,
        'RMSE': rmse
    }

def verify_data():
    """
    Quick test to verify data loading works.
    """
    print("Verifying data loading...")
    
    # Test red wine
    X_train_r, X_test_r, y_train_r, y_test_r, features_r = load_and_preprocess_data('red')
    print("Red wine data loaded successfully")
    
    # Test white wine
    X_train_w, X_test_w, y_train_w, y_test_w, features_w = load_and_preprocess_data('white')
    print("White wine data loaded successfully")
    
    print("\nAll data loading functions are working correctly.")
    return True

if __name__ == "__main__":
    verify_data()