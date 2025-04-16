import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging

def load_data(file_path):
    """Load CSV data from file_path.
    
    Args:
        file_path (str): Path to CSV file.
    Returns:
        pd.DataFrame: Loaded data.
    Raises:
        FileNotFoundError: If file_path does not exist.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file {file_path} not found.")

def interpolate_discharge(df, column='discharge'):
    """Interpolate missing discharge data using linear interpolation.
    
    Args:
        df (pd.DataFrame): DataFrame with discharge data.
        column (str): Column name for discharge data (default: 'discharge').
    Returns:
        pd.DataFrame: DataFrame with interpolated discharge.
    """
    missing_pct = df[column].isna().mean() * 100
    if missing_pct > 5:
        print(f"Warning: {missing_pct:.1f}% missing data in {column}, exceeds expected 1.5-3.0%.")
    df[column] = df[column].interpolate(method='linear', limit_direction='both')
    return df

def krige_rainfall(df, stations, lat_col='latitude', lon_col='longitude', value_col='rainfall'):
    """Perform ordinary kriging for missing rainfall data.
    
    Args:
        df (pd.DataFrame): DataFrame with rainfall data.
        stations (list): List of station names.
        lat_col (str): Column name for latitude (default: 'latitude').
        lon_col (str): Column name for longitude (default: 'longitude').
        value_col (str): Column name for rainfall (default: 'rainfall').
    Returns:
        pd.DataFrame: DataFrame with kriged rainfall.
    Notes:
        Uses spherical variogram model as per paper's spatial interpolation method.
    """
    data = df[[lat_col, lon_col, value_col]].dropna()
    ok = OrdinaryKriging(
        data[lon_col], data[lat_col], data[value_col],
        variogram_model='spherical', verbose=False
    )
    missing = df[df[value_col].isna()]
    if not missing.empty:
        z, _ = ok.execute('points', missing[lon_col], missing[lat_col])
        df.loc[df[value_col].isna(), value_col] = z
    return df

def preprocess_data(discharge_file, rainfall_file, output_file):
    """Preprocess discharge and rainfall data.
    
    Args:
        discharge_file (str): Path to discharge CSV (expected columns: date, discharge).
        rainfall_file (str): Path to rainfall CSV (expected columns: date, station, latitude, longitude, rainfall).
        output_file (str): Path to save processed data.
    """
    discharge_df = load_data(discharge_file)
    rainfall_df = load_data(rainfall_file)
    
    # Interpolate discharge
    discharge_df = interpolate_discharge(discharge_df)
    
    # Krige rainfall
    stations = rainfall_df['station'].unique()
    for station in stations:
        station_df = rainfall_df[rainfall_df['station'] == station]
        rainfall_df.loc[rainfall_df['station'] == station] = krige_rainfall(station_df, stations)
    
    # Save processed data
    discharge_df.to_csv(output_file.replace('input', 'processed_discharge'), index=False)
    rainfall_df.to_csv(output_file.replace('input', 'processed_rainfall'), index=False)

def main():
    """Main function to preprocess data for both watersheds."""
    for watershed in ['gilgel_abay', 'gumara']:
        preprocess_data(
            f'data/{watershed}_discharge.csv',
            f'data/{watershed}_rainfall.csv',
            f'data/processed_{watershed}_input.csv'
        )

if __name__ == "__main__":
    main()