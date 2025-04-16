import pandas as pd
import numpy as np
import joblib

def load_models(model_dir):
    """Load trained RF and QRF models.
    
    Args:
        model_dir (str): Directory containing model files.
    Returns:
        tuple: Trained RF and QRF models.
    """
    rf = joblib.load(f'{model_dir}/rf_model.pkl')
    qrf = joblib.load(f'{model_dir}/qrf_model.pkl')
    return rf, qrf

def fit_power_law(ssc, discharge):
    """Fit power-law model SSC = a * Q^b.
    
    Args:
        ssc (np.array): Suspended sediment concentration (g/L).
        discharge (np.array): Discharge (m³/s).
    Returns:
        tuple: Coefficients a and b.
    Raises:
        ValueError: If ssc or discharge contains zeros or negatives.
    """
    if (ssc <= 0).any() or (discharge <= 0).any():
        raise ValueError("SSC and discharge must be positive.")
    log_ssc = np.log(ssc)
    log_discharge = np.log(discharge)
    b, log_a = np.polyfit(log_discharge, log_ssc, 1)
    a = np.exp(log_a)
    return a, b

def calculate_sediment_load(ssc, discharge):
    """Calculate daily sediment load (tons/day).
    
    Args:
        ssc (np.array): Suspended sediment concentration (g/L).
        discharge (np.array): Discharge (m³/s).
    Returns:
        np.array: Sediment load (tons/day).
    Notes:
        Conversion factor 86.4 converts g/s to tons/day (Section 3.4).
    """
    return ssc * discharge * 86.4

def reconstruct_sedigraph(discharge_file, model_dir, output_file):
    """Reconstruct sedigraph and calculate sediment loads.
    
    Args:
        discharge_file (str): Path to discharge CSV (expected columns: date, discharge, rainfall, temperature, eto).
        model_dir (str): Directory containing trained models.
        output_file (str): Path to save sedigraph results.
    """
    data = pd.read_csv(discharge_file)
    X = data[['discharge', 'rainfall', 'temperature', 'eto']]
    
    rf, qrf = load_models(model_dir)
    
    # Predict SSC
    ssc_rf = rf.predict(X)
    ssc_qrf = qrf.predict(X, quantiles=[0.05, 0.5, 0.95])
    
    # Fit power-law
    a, b = fit_power_law(ssc_rf, data['discharge'])
    
    # Calculate sediment loads
    loads = calculate_sediment_load(ssc_rf, data['discharge'])
    
    # Save results
    results = pd.DataFrame({
        'date': data['date'],
        'discharge': data['discharge'],
        'ssc_rf': ssc_rf,
        'ssc_qrf_lower': ssc_qrf[:, 0],
        'ssc_qrf_median': ssc_qrf[:, 1],
        'ssc_qrf_upper': ssc_qrf[:, 2],
        'sediment_load': loads
    })
    results.to_csv(output_file, index=False)
    
    # Save power-law coefficients
    pd.DataFrame({'a': [a], 'b': [b]}).to_csv(f'{model_dir}/power_law.csv', index=False)
    
    return a, b, results

def main():
    """Main function to reconstruct sedigraphs for both watersheds."""
    for watershed in ['gilgel_abay', 'gumara']:
        a, b, results = reconstruct_sedigraph(
            f'data/processed_{watershed}_input.csv',
            f'results/{watershed}',
            f'results/{watershed}_sedigraph.csv'
        )
        print(f"{watershed.capitalize()} Power-law: SSC = {a:.3f} * Q^{b:.3f}")

if __name__ == "__main__":
    main()