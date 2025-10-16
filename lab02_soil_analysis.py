# CE 49X - Lab 2: Soil Test Data Analysis

# Student Name: Aksel Pulas
# Student ID: 2020403090  
# Date: 2025-10-16

import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load the soil test dataset from a CSV file.
    
    Parameters:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame, or None if the file is not found.
    """
    # Implement data loading with error handling
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at path '{file_path}'. Please verify the location.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The provided CSV file is empty.")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while loading data: {e}")
        return None

def clean_data(df):
    """
    Clean the dataset by handling missing values and removing outliers from 'soil_ph'.
    
    For each column in ['soil_ph', 'nitrogen', 'phosphorus', 'moisture']:
    - Missing values are filled with the column mean.
    
    Additionally, remove outliers in 'soil_ph' that are more than 3 standard deviations from the mean.
    
    Parameters:
        df (pd.DataFrame): The raw DataFrame.
        
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df_cleaned = df.copy()
    
    # Fill missing values in each specified column with the column mean
    columns_to_clean = ['soil_ph', 'nitrogen', 'phosphorus', 'moisture']
    for col in columns_to_clean:
        if col in df_cleaned.columns:
            mean_value = df_cleaned[col].mean(skipna=True)
            df_cleaned[col] = df_cleaned[col].fillna(mean_value)
    
    # Remove outliers in 'soil_ph': values more than 3 standard deviations from the mean
    if 'soil_ph' in df_cleaned.columns:
        ph_mean = df_cleaned['soil_ph'].mean()
        ph_std = df_cleaned['soil_ph'].std(ddof=1)
        if pd.notna(ph_std) and ph_std > 0:
            z_threshold = 3
            within_bounds_mask = (df_cleaned['soil_ph'] >= ph_mean - z_threshold * ph_std) & \
                                 (df_cleaned['soil_ph'] <= ph_mean + z_threshold * ph_std)
            df_cleaned = df_cleaned.loc[within_bounds_mask].reset_index(drop=True)
    
    print(df_cleaned.head())
    return df_cleaned

def compute_statistics(df, column):
    """
    Compute and print descriptive statistics for the specified column.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column for which to compute statistics.
    """
    # Calculate descriptive statistics for the column
    series = df[column].dropna()
    min_val = series.min()
    max_val = series.max()
    mean_val = series.mean()
    median_val = series.median()
    std_val = series.std(ddof=1)
    
    print(f"\nDescriptive statistics for '{column}':")
    print(f"  Minimum: {min_val}")
    print(f"  Maximum: {max_val}")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Median: {median_val:.2f}")
    print(f"  Standard Deviation: {std_val:.2f}")

def main():
    # TODO: Update the file path to point to your soil_test.csv file
    file_path = 'soil_test.csv'  # Update this path as needed
    
    # Load the dataset using the load_data function
    df = load_data(file_path)
    if df is None:
        return
    
    # Clean the dataset using the clean_data function
    df_clean = clean_data(df)
    
    # Compute and display statistics for the 'soil_ph' column
    if 'soil_ph' in df_clean.columns:
        compute_statistics(df_clean, 'soil_ph')
    else:
        print("Column 'soil_ph' not found in the dataset.")
    
    # Compute statistics for other columns
    if 'nitrogen' in df_clean.columns:
        compute_statistics(df_clean, 'nitrogen')
    if 'phosphorus' in df_clean.columns:
        compute_statistics(df_clean, 'phosphorus')
    if 'moisture' in df_clean.columns:
        compute_statistics(df_clean, 'moisture')
    
if __name__ == '__main__':
    main()

# =============================================================================
# REFLECTION QUESTIONS
# =============================================================================
# Answer these questions in comments below:

# 1. What was the most challenging part of this lab?
# Answer: The most challenging part was implementing proper error handling for different types of CSV file issues (file not found, empty files, parsing errors) and understanding how to handle missing values and outliers in the data cleaning process. Ensuring the outlier removal logic worked correctly with the z-score approach was also challenging.

# 2. How could soil data analysis help civil engineers in real projects?
# Answer: Soil data analysis is crucial for civil engineers in foundation design, slope stability analysis, and construction planning. Understanding soil pH, nutrient content, and moisture levels helps engineers determine appropriate construction methods, select suitable materials, assess environmental impact, and ensure long-term stability of structures built on different soil types.

# 3. What additional features would make this soil analysis tool more useful?
# Answer: Additional features could include: data visualization (histograms, scatter plots, correlation matrices), classification of soil types based on parameters, trend analysis over time, export functionality for reports, integration with GIS systems for spatial analysis, and automated recommendations for soil treatment based on the analysis results.

# 4. How did error handling improve the robustness of your code?
# Answer: Error handling made the code more robust by gracefully managing various failure scenarios such as missing files, corrupted data, or unexpected data formats. This prevents the program from crashing and provides informative error messages to users, making it more professional and user-friendly. It also allows the program to continue running even when encountering minor data issues.