import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, ccf
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt


# Load real and synthetic data

def load_timeVAE_data(real, synthetic, time_column='observation_date'):
    """
    Load real and synthetic time series data from CSV files.
    
    Args:
        real (str): Path to the real data CSV file.
        synthetic (str): Path to the synthetic data CSV file.
        time_column (str): Name of the time column in the CSV files.
    
    Returns:
        real_features (np.ndarray): Numerical features from the real data.
        synthetic_features (np.ndarray): Numerical features from the synthetic data.
    """
    # Load data
    real_data = pd.read_csv(real)
    synthetic_data = pd.read_csv(synthetic)
    
    # Convert time column to datetime
    real_data[time_column] = pd.to_datetime(real_data[time_column])
    synthetic_data[time_column] = pd.to_datetime(synthetic_data[time_column])
    
    # Extract feature columns (all columns except the time column)
    feature_columns = [col for col in real_data.columns if col != time_column]
    
    # Extract numerical features
    real_features = real_data[feature_columns].values
    synthetic_features = synthetic_data[feature_columns].values
    
    return real_features, synthetic_features, feature_columns

# Calculate statistics of real vs synthetic data
def compare_statistics(real, synthetic):
    real_stats = {
        'mean': np.mean(real, axis=0),
        'std': np.std(real, axis=0),
        'min': np.min(real, axis=0),
        'max': np.max(real, axis=0),
    }
    synthetic_stats = {
        'mean': np.mean(synthetic, axis=0),
        'std': np.std(synthetic, axis=0),
        'min': np.min(synthetic, axis=0),
        'max': np.max(synthetic, axis=0),
    }
    for key in real_stats:
        print(f"Real {key}: {real_stats[key]}")
        print(f"Synthetic {key}: {synthetic_stats[key]}")
        print()


# Autocorrelation for each feature
def evaluate_autocorrelation(real, synthetic, feature_columns, max_lag=10):
    for i, feature in enumerate(feature_columns):
        real_acf = acf(real[:, i], nlags=max_lag)
        synthetic_acf = acf(synthetic[:, i], nlags=max_lag)
        print(f"Autocorrelation for {feature}:")
        print(f"Real: {real_acf}")
        print(f"Synthetic: {synthetic_acf}")
        print()


# Cross-correlation between features
def evaluate_cross_correlation(real, synthetic, feature_columns, max_lag=10):
    for i, feature1 in enumerate(feature_columns):
        for j, feature2 in enumerate(feature_columns):
            if i != j:
                real_ccf = ccf(real[:, i], real[:, j], unbiased=False)[:max_lag]
                synthetic_ccf = ccf(synthetic[:, i], synthetic[:, j], unbiased=False)[:max_lag]
                print(f"Cross-correlation between {feature1} and {feature2}:")
                print(f"Real: {real_ccf}")
                print(f"Synthetic: {synthetic_ccf}")
                print()


# Autocorrelation for each feature
def evaluate_autocorrelation(real, synthetic, feature_columns, max_lag=10):
    for i, feature in enumerate(feature_columns):
        real_acf = acf(real[:, i], nlags=max_lag)
        synthetic_acf = acf(synthetic[:, i], nlags=max_lag)
        print(f"Autocorrelation for {feature}:")
        print(f"Real: {real_acf}")
        print(f"Synthetic: {synthetic_acf}")
        print()


# Cross-correlation between features
def evaluate_cross_correlation(real, synthetic, feature_columns, max_lag=10):
    for i, feature1 in enumerate(feature_columns):
        for j, feature2 in enumerate(feature_columns):
            if i != j:
                real_ccf = ccf(real[:, i], real[:, j], unbiased=False)[:max_lag]
                synthetic_ccf = ccf(synthetic[:, i], synthetic[:, j], unbiased=False)[:max_lag]
                print(f"Cross-correlation between {feature1} and {feature2}:")
                print(f"Real: {real_ccf}")
                print(f"Synthetic: {synthetic_ccf}")
                print()


# Jensen-Shannon Divergence for distribution comparison
def evaluate_diversity(real, synthetic, feature_columns):
    for i, feature in enumerate(feature_columns):
        jsd = jensenshannon(real[:, i], synthetic[:, i])
        print(f"Jensen-Shannon Divergence for {feature}: {jsd}")


# Plot time series
def plot_time_series(real, synthetic, feature_index, feature_name, save_dir='eval_plots'):
    """
    Plot real and synthetic time series for a given feature and save the plot to a directory.
    
    Args:
        real (np.ndarray): Real time series data.
        synthetic (np.ndarray): Synthetic time series data.
        feature_index (int): Index of the feature to plot.
        feature_name (str): Name of the feature (for labeling the plot).
        save_dir (str): Directory to save the plots.
    """
    try:
        # Convert save_dir to an absolute path
        save_dir = os.path.abspath(save_dir)
        print(f"Saving plots to: {save_dir}")
        
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Debugging: Print shapes of data
        print(f"Real data shape: {real.shape}")
        print(f"Synthetic data shape: {synthetic.shape}")
        
        # Plot the time series
        plt.figure(figsize=(10, 5))
        plt.plot(real[:, feature_index], label='Real')
        plt.plot(synthetic[:, feature_index], label='Synthetic', alpha=0.7)
        plt.title(feature_name)
        plt.legend()
        
        # Save the plot
        save_path = os.path.join(save_dir, f'{feature_name}.png')
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error plotting {feature_name}: {e}") 

    
def evaluate_synthetic_data(real, synthetic, feature_columns):
    print("Descriptive Statistics:")
    compare_statistics(real, synthetic)
    
    print("Temporal Dynamics:")
    evaluate_autocorrelation(real, synthetic, feature_columns)
    evaluate_cross_correlation(real, synthetic, feature_columns)
    
    print("Diversity and Coverage:")
    evaluate_diversity(real, synthetic, feature_columns)
    
    print("Visualization:")
    for i, feature in enumerate(feature_columns):
        plot_time_series(real, synthetic, i, feature)