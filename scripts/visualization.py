import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(importance_file, output_file):
    """Plot feature importance from Gradient Boosting.
    
    Args:
        importance_file (str): Path to feature importance CSV.
        output_file (str): Path to save figure.
    """
    importance = pd.read_csv(importance_file)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='importance', y='feature', data=importance)
    plt.title('Feature Importance from Gradient Boosting (Figure 4)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_sediment_rating_curve(sedigraph_file, a, b, output_file):
    """Plot ML-based sediment rating curve.
    
    Args:
        sedigraph_file (str): Path to sedigraph CSV.
        a (float): Power-law coefficient a.
        b (float): Power-law exponent b.
        output_file (str): Path to save figure.
    """
    data = pd.read_csv(sedigraph_file)
    plt.figure(figsize=(8, 6))
    plt.scatter(data['discharge'], data['ssc_rf'], alpha=0.5, label='RF Predictions')
    x = np.linspace(data['discharge'].min(), data['discharge'].max(), 100)
    plt.plot(x, a * x**b, 'r-', label=f'Power-law: SSC = {a:.3f}Q^{b:.3f}')
    plt.xlabel('Discharge (m³/s)')
    plt.ylabel('SSC (g/L)')
    plt.title('ML-based Sediment Rating Curve (Figure 6)')
    plt.legend()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_qrf_uncertainty(sedigraph_file, output_file):
    """Plot sedigraph with QRF uncertainty bounds.
    
    Args:
        sedigraph_file (str): Path to sedigraph CSV.
        output_file (str): Path to save figure.
    """
    data = pd.read_csv(sedigraph_file)
    data['date'] = pd.to_datetime(data['date'])
    plt.figure(figsize=(10, 6))
    plt.plot(data['date'], data['ssc_rf'], label='RF Prediction')
    plt.fill_between(data['date'], data['ssc_qrf_lower'], data['ssc_qrf_upper'], 
                     alpha=0.3, label='QRF 5th-95th Percentile')
    plt.xlabel('Date')
    plt.ylabel('SSC (g/L)')
    plt.title('Sedigraph with QRF Uncertainty Bounds (Figure 7)')
    plt.legend()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_temporal_trends(sedigraph_file, output_file):
    """Plot annual sediment yield trends.
    
    Args:
        sedigraph_file (str): Path to sedigraph CSV.
        output_file (str): Path to save figure.
    """
    data = pd.read_csv(sedigraph_file)
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    annual_yield = data.groupby('year')['sediment_load'].sum() / 1e4  # Convert to tonnes/ha/yr
    plt.figure(figsize=(10, 6))
    annual_yield.plot()
    plt.title('Annual Sediment Yield (tonnes/ha/yr) (Figure 8)')
    plt.xlabel('Year')
    plt.ylabel('Sediment Yield')
    plt.savefig(output_file, dpi=300)
    plt.close()

def main():
    """Main function to generate visualizations for both watersheds."""
    for watershed in ['gilgel_abay', 'gumara']:
        plot_feature_importance(
            f'results/{watershed}/feature_importance.csv',
            f'figures/{watershed}_feature_importance.png'
        )
        power_law = pd.read_csv(f'results/{watershed}/power_law.csv')
        plot_sediment_rating_curve(
            f'results/{watershed}_sedigraph.csv',
            power_law['a'][0], power_law['b'][0],
            f'figures/{watershed}_rating_curve.png'
        )
        plot_qrf_uncertainty(
            f'results/{watershed}_sedigraph.csv',
            f'figures/{watershed}_qrf_uncertainty.png'
        )
        plot_temporal_trends(
            f'results/{watershed}_sedigraph.csv',
            f'figures/{watershed}_temporal_trends.png'
        )

if __name__ == "__main__":
    main()