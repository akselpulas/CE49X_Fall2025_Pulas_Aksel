"""
Lab 3: ERA5 Weather Data Analysis
Student: Aksel Pulas
Date: 2025-10-23

This script analyzes ERA5 wind data for Berlin and Munich.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')


# ============================================================================
# 1. LOAD AND EXPLORE DATASETS
# ============================================================================

def load_data(file_path, city_name):
    """
    Load ERA5 weather data from CSV file.
    
    Parameters:
        file_path (str): Path to CSV file
        city_name (str): Name of the city
    
    Returns:
        pd.DataFrame: Loaded data with timestamp as index
    """
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime (careful with pd.Timestamp)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Display basic information
        print(f"\n{'='*50}")
        print(f"{city_name} Dataset Information")
        print(f"{'='*50}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        
        # Handle missing values
        missing = df.isnull().sum()
        print(f"\nMissing values:\n{missing}")
        if missing.sum() > 0:
            print("Filling missing values with column mean...")
            df = df.fillna(df.mean())
        
        # Summary statistics
        print(f"\nSummary Statistics:")
        print(df[['u10m', 'v10m']].describe())
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None


# ============================================================================
# 2. COMPUTE TEMPORAL AGGREGATIONS
# ============================================================================

def calculate_wind_speed(u10m, v10m):
    """
    Calculate wind speed from u and v components.
    Formula: wind_speed = sqrt(u^2 + v^2)
    
    Parameters:
        u10m: Eastward wind component (m/s)
        v10m: Northward wind component (m/s)
    
    Returns:
        Wind speed magnitude (m/s)
    """
    return np.sqrt(u10m**2 + v10m**2)


def compute_aggregations(df, city_name):
    """
    Calculate monthly and seasonal averages.
    
    Parameters:
        df (pd.DataFrame): Weather data
        city_name (str): City name
    
    Returns:
        tuple: (monthly_avg, seasonal_avg)
    """
    # Add wind speed column
    df['wind_speed'] = calculate_wind_speed(df['u10m'], df['v10m'])
    
    # Add time features
    df['month'] = df['timestamp'].dt.month
    df['season'] = df['timestamp'].dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Calculate monthly averages
    monthly_avg = df.groupby('month')['wind_speed'].mean()
    
    # Calculate seasonal averages
    seasonal_avg = df.groupby('season')['wind_speed'].mean()
    
    print(f"\n{city_name} - Monthly Averages (m/s):")
    for month, speed in monthly_avg.items():
        print(f"  Month {month:2d}: {speed:.2f}")
    
    print(f"\n{city_name} - Seasonal Averages (m/s):")
    for season, speed in seasonal_avg.items():
        print(f"  {season:7s}: {speed:.2f}")
    
    return monthly_avg, seasonal_avg, df


# ============================================================================
# 3. STATISTICAL ANALYSIS
# ============================================================================

def find_extreme_events(df, city_name, n=5):
    """
    Identify periods with extreme wind conditions.
    
    Parameters:
        df (pd.DataFrame): Weather data
        city_name (str): City name
        n (int): Number of top events to show
    """
    print(f"\n{city_name} - Top {n} Highest Wind Speeds:")
    top_winds = df.nlargest(n, 'wind_speed')
    
    for idx, row in top_winds.iterrows():
        print(f"  {row['timestamp']}: {row['wind_speed']:.2f} m/s")


def calculate_diurnal_pattern(df):
    """
    Calculate diurnal (daily) patterns in wind speed.
    
    Parameters:
        df (pd.DataFrame): Weather data
    
    Returns:
        pd.Series: Average wind speed by hour (0-23)
    """
    df['hour'] = df['timestamp'].dt.hour
    hourly_avg = df.groupby('hour')['wind_speed'].mean()
    # Ensure all 24 hours are present
    hourly_avg = hourly_avg.reindex(range(24), fill_value=0)
    return hourly_avg


# ============================================================================
# 4. VISUALIZATION
# ============================================================================

def plot_monthly_comparison(berlin_monthly, munich_monthly):
    """Create time series plot of monthly average wind speeds."""
    plt.figure(figsize=(10, 5))
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x_labels = [months[m-1] for m in berlin_monthly.index]
    
    plt.plot(x_labels, berlin_monthly.values, marker='o', label='Berlin', linewidth=2)
    plt.plot(x_labels, munich_monthly.values, marker='s', label='Munich', linewidth=2)
    
    plt.xlabel('Month')
    plt.ylabel('Average Wind Speed (m/s)')
    plt.title('Monthly Average Wind Speeds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('monthly_comparison.png', dpi=300)
    plt.close()
    print("\n[Saved] monthly_comparison.png")


def plot_seasonal_comparison(berlin_seasonal, munich_seasonal):
    """Create seasonal comparison bar chart."""
    plt.figure(figsize=(10, 5))
    
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    berlin_vals = [berlin_seasonal.get(s, 0) for s in seasons]
    munich_vals = [munich_seasonal.get(s, 0) for s in seasons]
    
    x = np.arange(len(seasons))
    width = 0.35
    
    plt.bar(x - width/2, berlin_vals, width, label='Berlin')
    plt.bar(x + width/2, munich_vals, width, label='Munich')
    
    plt.xlabel('Season')
    plt.ylabel('Average Wind Speed (m/s)')
    plt.title('Seasonal Comparison')
    plt.xticks(x, seasons)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('seasonal_comparison.png', dpi=300)
    plt.close()
    print("[Saved] seasonal_comparison.png")


def plot_diurnal_pattern(berlin_hourly, munich_hourly):
    """Create diurnal pattern plot."""
    plt.figure(figsize=(10, 5))
    
    hours = range(24)
    plt.plot(hours, berlin_hourly.values, marker='o', label='Berlin', linewidth=2)
    plt.plot(hours, munich_hourly.values, marker='s', label='Munich', linewidth=2)
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Wind Speed (m/s)')
    plt.title('Diurnal Wind Speed Pattern')
    plt.xticks(range(0, 24, 2))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('diurnal_pattern.png', dpi=300)
    plt.close()
    print("[Saved] diurnal_pattern.png")


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    """Main analysis function."""
    print("\n" + "="*50)
    print("ERA5 WEATHER DATA ANALYSIS - LAB 3")
    print("="*50)
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths relative to script location
    berlin_file = os.path.join(script_dir, 'berlin_era5_wind_20241231_20241231.csv')
    munich_file = os.path.join(script_dir, 'munich_era5_wind_20241231_20241231.csv')
    
    # Check if files exist
    if not os.path.exists(berlin_file) or not os.path.exists(munich_file):
        print("\nError: Data files not found!")
        print(f"Looking in: {script_dir}")
        print(f"Expected files:")
        print(f"  - berlin_era5_wind_20241231_20241231.csv")
        print(f"  - munich_era5_wind_20241231_20241231.csv")
        return
    
    # 1. Load and explore datasets
    print("\n1. LOADING AND EXPLORING DATASETS")
    print("="*50)
    berlin_df = load_data(berlin_file, 'Berlin')
    munich_df = load_data(munich_file, 'Munich')
    
    if berlin_df is None or munich_df is None:
        return
    
    # 2. Compute temporal aggregations
    print("\n2. TEMPORAL AGGREGATIONS")
    print("="*50)
    berlin_monthly, berlin_seasonal, berlin_df = compute_aggregations(berlin_df, 'Berlin')
    munich_monthly, munich_seasonal, munich_df = compute_aggregations(munich_df, 'Munich')
    
    # Compare seasonal patterns
    print("\nSeasonal Pattern Comparison:")
    print(f"  Berlin windiest season: {berlin_seasonal.idxmax()} ({berlin_seasonal.max():.2f} m/s)")
    print(f"  Munich windiest season: {munich_seasonal.idxmax()} ({munich_seasonal.max():.2f} m/s)")
    
    # 3. Statistical analysis
    print("\n3. STATISTICAL ANALYSIS")
    print("="*50)
    find_extreme_events(berlin_df, 'Berlin')
    find_extreme_events(munich_df, 'Munich')
    
    print("\nDiurnal Patterns:")
    berlin_hourly = calculate_diurnal_pattern(berlin_df)
    munich_hourly = calculate_diurnal_pattern(munich_df)
    print(f"  Berlin peak hour: {berlin_hourly.idxmax()}:00 ({berlin_hourly.max():.2f} m/s)")
    print(f"  Munich peak hour: {munich_hourly.idxmax()}:00 ({munich_hourly.max():.2f} m/s)")
    
    # 4. Visualizations
    print("\n4. CREATING VISUALIZATIONS")
    print("="*50)
    plot_monthly_comparison(berlin_monthly, munich_monthly)
    plot_seasonal_comparison(berlin_seasonal, munich_seasonal)
    plot_diurnal_pattern(berlin_hourly, munich_hourly)
    
    # Summary
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print("\nKey Findings:")
    print(f"  Berlin avg: {berlin_df['wind_speed'].mean():.2f} m/s")
    print(f"  Munich avg: {munich_df['wind_speed'].mean():.2f} m/s")
    print("\nGenerated visualizations:")
    print("  - monthly_comparison.png")
    print("  - seasonal_comparison.png")
    print("  - diurnal_pattern.png")


if __name__ == "__main__":
    main()
