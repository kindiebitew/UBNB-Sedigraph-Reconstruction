import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from quantile_forest import RandomForestQuantileRegressor
import joblib

def load_processed_data(discharge_file, ssc_file):
    """Load processed discharge and SSC data.
    
    Args:
        discharge_file (str): Path to discharge CSV (expected columns: date, discharge, rainfall, temperature, eto).
        ssc_file (str): Path to SSC CSV (expected columns: date, ssc).
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    discharge_df = pd.read_csv(discharge_file)
    ssc_df = pd.read_csv(ssc_file)
    return discharge_df.merge(ssc_df, on='date')

def train_gradient_boosting(X, y):
    """Train Gradient Boosting for feature importance.
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable (SSC).
    Returns:
        GradientBoostingRegressor: Trained model.
    """
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X, y)
    return gb

def train_random_forest(X, y):
    """Train Random Forest for SSC prediction.
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable (SSC).
    Returns:
        RandomForestRegressor: Trained model.
    Notes:
        Hyperparameters tuned via 5-fold cross-validation (n_estimators=100, max_depth=10).
    """
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, y)
    return rf

def train_quantile_forest(X, y):
    """Train Quantile Random Forest for uncertainty estimation.
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable (SSC).
    Returns:
        RandomForestQuantileRegressor: Trained model.
    """
    qrf = RandomForestQuantileRegressor(n_estimators=200, random_state=42)
    qrf.fit(X, y)
    return qrf

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance.
    
    Args:
        model: Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
    Returns:
        tuple: R² score and MAE.
    """
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)

def train_models(discharge_file, ssc_file, output_dir='results'):
    """Train all models and save results.
    
    Args:
        discharge_file (str): Path to discharge CSV.
        ssc_file (str): Path to SSC CSV.
        output_dir (str): Directory to save results.
    """
    data = load_processed_data(discharge_file, ssc_file)
    X = data[['discharge', 'rainfall', 'temperature', 'eto']]  # ETo calculated via FAO Penman-Monteith
    y = data['ssc']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    gb = train_gradient_boosting(X_train, y_train)
    rf = train_random_forest(X_train, y_train)
    qrf = train_quantile_forest(X_train, y_train)
    
    # Save models
    joblib.dump(gb, f'{output_dir}/gb_model.pkl')
    joblib.dump(rf, f'{output_dir}/rf_model.pkl')
    joblib.dump(qrf, f'{output_dir}/qrf_model.pkl')
    
    # Evaluate models
    rf_r2, rf_mae = evaluate_model(rf, X_test, y_test)
    qrf_r2, qrf_mae = evaluate_model(qrf, X_test, y_test)
    
    print(f"RF R^2: {rf_r2:.2f}, MAE: {rf_mae:.2f}")
    print(f"QRF R^2: {qrf_r2:.2f}, MAE: {qrf_mae:.2f}")
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': gb.feature_importances_
    })
    feature_importance.to_csv(f'{output_dir}/feature_importance.csv', index=False)
    
    return gb, rf, qrf

def main():
    """Main function to train models for both watersheds."""
    for watershed in ['gilgel_abay', 'gumara']:
        train_models(
            f'data/processed_{watershed}_input.csv',
            f'data/{watershed}_ssc.csv',
            f'results/{watershed}'
        )

if __name__ == "__main__":
    main()