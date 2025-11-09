"""
Lab 4: Statistical Analysis
Descriptive Statistics and Probability Distributions

Author: Aksel Pulas
Date: November 2025

This script analyzes engineering data using:
- Descriptive statistics (mean, median, std, etc.)
- Probability distributions (Binomial, Poisson, Normal, Exponential)
- Bayes' theorem
- Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, binom, poisson, uniform, expon
import os
import warnings
warnings.filterwarnings('ignore')

# Set nice plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.switch_backend('Agg')  # Don't show plots, just save them


# ==============================================================================
# PART 1: DATA LOADING
# ==============================================================================

def load_data(file_path):
    """
    Load CSV data file.
    
    Args:
        file_path: Name of CSV file
    
    Returns:
        DataFrame with loaded data
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, file_path)
        data = pd.read_csv(full_path)
        print(f"✓ Loaded: {file_path} - Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"✗ Error loading {file_path}: {e}")
        return None


# ==============================================================================
# PART 2: DESCRIPTIVE STATISTICS
# ==============================================================================

def calculate_descriptive_stats(data, column='strength_mpa'):
    """
    Calculate all descriptive statistics.
    
    Returns dictionary with:
    - Central tendency: mean, median, mode
    - Spread: std, variance, range, IQR
    - Shape: skewness, kurtosis
    - Five-number summary: min, Q1, median, Q3, max
    """
    values = data[column].dropna()
    
    # Central tendency
    mean_val = np.mean(values)
    median_val = np.median(values)
    mode_val = stats.mode(values, keepdims=True).mode[0]
    
    # Spread
    std_val = np.std(values, ddof=1)
    variance_val = np.var(values, ddof=1)
    range_val = np.max(values) - np.min(values)
    
    # Quartiles
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr_val = q3 - q1
    
    # Shape
    skewness_val = stats.skew(values)
    kurtosis_val = stats.kurtosis(values)
    
    return {
        'mean': mean_val, 'median': median_val, 'mode': mode_val,
        'std': std_val, 'variance': variance_val, 'range': range_val,
        'q1': q1, 'q3': q3, 'iqr': iqr_val,
        'min': np.min(values), 'max': np.max(values),
        'skewness': skewness_val, 'kurtosis': kurtosis_val
    }


# ==============================================================================
# PART 3: PROBABILITY CALCULATIONS
# ==============================================================================

def calculate_probability_binomial(n, p, k):
    """
    Binomial probability: number of successes in n trials.
    Formula: P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
    
    Example: 100 components, 5% defect rate, P(3 defects)?
    """
    if isinstance(k, str) and '<=' in k:
        k_val = int(k.replace('<=', ''))
        return binom.cdf(k_val, n, p)
    return binom.pmf(k, n, p)


def calculate_probability_poisson(lambda_param, k):
    """
    Poisson probability: number of events in fixed interval.
    Formula: P(X=k) = (λ^k * e^(-λ)) / k!
    
    Example: Average 10 trucks/hour, P(8 trucks)?
    """
    if isinstance(k, str) and '>' in k:
        k_val = int(k.replace('>', ''))
        return 1 - poisson.cdf(k_val, lambda_param)
    return poisson.pmf(k, lambda_param)


def calculate_probability_normal(mean, std, x_lower=None, x_upper=None):
    """
    Normal probability.
    Formula: f(x) = (1/(σ√(2π))) * e^(-(x-μ)²/(2σ²))
    
    Example: Steel strength μ=250, σ=15, P(X>280)?
    """
    if x_lower is not None and x_upper is None:
        return 1 - norm.cdf(x_lower, mean, std)
    elif x_lower is None and x_upper is not None:
        return norm.cdf(x_upper, mean, std)
    return norm.cdf(x_upper, mean, std) - norm.cdf(x_lower, mean, std)


def calculate_probability_exponential(mean, x):
    """
    Exponential probability: time until failure.
    Formula: f(x) = λ * e^(-λx), where λ = 1/mean
    
    Example: Mean lifetime 1000h, P(fail before 500h)?
    """
    prob_before = expon.cdf(x, scale=mean)
    prob_after = 1 - prob_before
    return {'prob_before': prob_before, 'prob_after': prob_after}


def apply_bayes_theorem(prior, sensitivity, specificity):
    """
    Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)
    
    Example: Structural damage test
    - Prior: 5% structures have damage
    - Sensitivity: 95% (detects damage when present)
    - Specificity: 90% (correctly identifies no damage)
    - Question: If test positive, P(actual damage)?
    """
    # Total probability of positive test
    prob_positive = (sensitivity * prior) + ((1 - specificity) * (1 - prior))
    
    # Posterior probability
    posterior = (sensitivity * prior) / prob_positive
    
    return {
        'prior': prior,
        'posterior': posterior,
        'prob_positive': prob_positive
    }


# ==============================================================================
# PART 4: DISTRIBUTION FITTING
# ==============================================================================

def fit_distribution(data, column, distribution_type='normal'):
    """
    Fit normal distribution to data.
    Returns mean and standard deviation.
    """
    values = data[column].dropna()
    if distribution_type == 'normal':
        mu = np.mean(values)
        sigma = np.std(values, ddof=1)
        print(f"  Fitted: μ={mu:.2f}, σ={sigma:.2f}")
        return mu, sigma
    return None


# ==============================================================================
# PART 5: VISUALIZATIONS
# ==============================================================================

def plot_distribution(data, column, title, save_path):
    """Create histogram with mean, median, mode lines."""
    values = data[column].dropna()
    
    mean_val = np.mean(values)
    median_val = np.median(values)
    mode_val = stats.mode(values, keepdims=True).mode[0]
    
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.hist(values, bins=20, density=True, alpha=0.7, 
             color='skyblue', edgecolor='black')
    
    # Normal curve overlay
    x = np.linspace(values.min(), values.max(), 100)
    mu, sigma = mean_val, np.std(values, ddof=1)
    plt.plot(x, norm.pdf(x, mu, sigma), 'k-', linewidth=2, label='Normal Curve')
    
    # Central tendency lines
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                label=f'Median: {median_val:.2f}')
    plt.axvline(mode_val, color='orange', linestyle='--', linewidth=2, 
                label=f'Mode: {mode_val:.2f}')
    
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(save_path)}")


def plot_material_comparison(data, column, group_column, save_path):
    """Create boxplot comparing different materials."""
    plt.figure(figsize=(12, 7))
    data.boxplot(column=column, by=group_column, figsize=(12, 7))
    plt.suptitle('')
    plt.title(f'{column} by {group_column}', fontsize=14, fontweight='bold')
    plt.xlabel(group_column, fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(save_path)}")


def plot_distribution_fitting(data, column, fitted_params, save_path):
    """
    Plot data histogram with fitted normal distribution.
    Also generates synthetic data to compare with real data.
    """
    values = data[column].dropna()
    
    plt.figure(figsize=(12, 6))
    
    # Histogram of real data
    plt.hist(values, bins=25, density=True, alpha=0.6, 
             color='blue', edgecolor='black', label='Real Data')
    
    # Fitted curve and synthetic data
    if fitted_params:
        mu, sigma = fitted_params
        
        # Generate synthetic data from fitted distribution
        np.random.seed(42)
        synthetic_data = np.random.normal(mu, sigma, len(values))
        
        # Plot synthetic data histogram
        plt.hist(synthetic_data, bins=25, density=True, alpha=0.4, 
                 color='red', edgecolor='black', label='Synthetic Data')
        
        # Plot fitted curve
        x = np.linspace(min(values.min(), synthetic_data.min()), 
                       max(values.max(), synthetic_data.max()), 100)
        plt.plot(x, norm.pdf(x, mu, sigma), 'k-', linewidth=3, 
                 label=f'Fitted Normal (μ={mu:.2f}, σ={sigma:.2f})')
        
        # Print validation
        print(f"  Validation - Synthetic data: μ={np.mean(synthetic_data):.2f}, σ={np.std(synthetic_data, ddof=1):.2f}")
    
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution Fitting: Real vs Synthetic Data', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(save_path)}")


def plot_probability_distributions(save_path):
    """
    Create plots showing PMF/PDF and CDF for all distributions.
    Shows 6 distributions: Binomial, Poisson, Uniform, Normal, Exponential, and comparison.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Probability Distributions: PMF/PDF and CDF', 
                 fontsize=16, fontweight='bold')
    
    # 1. Binomial Distribution
    n, p = 100, 0.05
    x_binom = np.arange(0, 20)
    axes[0, 0].bar(x_binom, binom.pmf(x_binom, n, p), alpha=0.7, color='skyblue')
    axes[0, 0].set_title(f'Binomial PMF (n={n}, p={p})', fontweight='bold')
    axes[0, 0].set_xlabel('Number of Defects')
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Poisson Distribution
    lambda_param = 10
    x_poisson = np.arange(0, 25)
    axes[0, 1].bar(x_poisson, poisson.pmf(x_poisson, lambda_param), 
                   alpha=0.7, color='lightgreen')
    axes[0, 1].set_title(f'Poisson PMF (λ={lambda_param})', fontweight='bold')
    axes[0, 1].set_xlabel('Number of Events')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Normal Distribution (PDF)
    mean, std = 250, 15
    x_normal = np.linspace(200, 300, 200)
    axes[0, 2].plot(x_normal, norm.pdf(x_normal, mean, std), 'b-', linewidth=2)
    axes[0, 2].fill_between(x_normal, norm.pdf(x_normal, mean, std), alpha=0.3)
    axes[0, 2].set_title(f'Normal PDF (μ={mean}, σ={std})', fontweight='bold')
    axes[0, 2].set_xlabel('Strength (MPa)')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Exponential Distribution (PDF)
    mean_exp = 1000
    x_exp = np.linspace(0, 3000, 200)
    axes[1, 0].plot(x_exp, expon.pdf(x_exp, scale=mean_exp), 'r-', linewidth=2)
    axes[1, 0].fill_between(x_exp, expon.pdf(x_exp, scale=mean_exp), 
                            alpha=0.3, color='red')
    axes[1, 0].set_title(f'Exponential PDF (mean={mean_exp})', fontweight='bold')
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Uniform Distribution
    a, b = 0, 100
    x_uniform = np.linspace(a-10, b+10, 200)
    axes[1, 1].plot(x_uniform, uniform.pdf(x_uniform, a, b-a), 'purple', linewidth=2)
    axes[1, 1].fill_between(x_uniform, uniform.pdf(x_uniform, a, b-a), 
                            alpha=0.3, color='purple')
    axes[1, 1].set_title(f'Uniform PDF (a={a}, b={b})', fontweight='bold')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. CDF Comparison
    x_cdf = np.linspace(0, 20, 100)
    axes[1, 2].plot(x_cdf, binom.cdf(x_cdf, 100, 0.05), label='Binomial', linewidth=2)
    axes[1, 2].plot(x_cdf, poisson.cdf(x_cdf, 10), label='Poisson', linewidth=2)
    axes[1, 2].set_title('CDF Comparison', fontweight='bold')
    axes[1, 2].set_xlabel('Value')
    axes[1, 2].set_ylabel('Cumulative Probability')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(save_path)}")


def plot_concrete_boxplot(data, column, save_path):
    """
    Create boxplot for concrete strength showing quartiles and outliers.
    Task 1: Boxplot showing quartiles
    """
    values = data[column].dropna()
    
    plt.figure(figsize=(10, 7))
    
    # Create boxplot
    bp = plt.boxplot(values, vert=True, patch_artist=True, 
                     widths=0.5, showmeans=True)
    
    # Customize colors
    bp['boxes'][0].set_facecolor('skyblue')
    bp['boxes'][0].set_alpha(0.7)
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)
    bp['means'][0].set_marker('D')
    bp['means'][0].set_markerfacecolor('green')
    bp['means'][0].set_markersize(8)
    
    # Add labels
    stats = calculate_descriptive_stats(data, column)
    plt.text(1.3, stats['q1'], f"Q1: {stats['q1']:.2f}", fontsize=11, va='center')
    plt.text(1.3, stats['median'], f"Median: {stats['median']:.2f}", fontsize=11, va='center')
    plt.text(1.3, stats['q3'], f"Q3: {stats['q3']:.2f}", fontsize=11, va='center')
    plt.text(1.3, stats['min'], f"Min: {stats['min']:.2f}", fontsize=10, va='center')
    plt.text(1.3, stats['max'], f"Max: {stats['max']:.2f}", fontsize=10, va='center')
    
    plt.ylabel(column, fontsize=12)
    plt.title('Concrete Strength - Boxplot with Quartiles', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks([1], ['Concrete Strength'])
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(save_path)}")


def plot_probability_tree(bayes_result, save_path):
    """
    Create simple probability tree diagram for Bayes' theorem.
    Task 4: Visualize using probability tree
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    fig.suptitle("Bayes' Theorem: Probability Tree - Structural Damage Detection", 
                 fontsize=14, fontweight='bold')
    
    prior = bayes_result['prior']
    posterior = bayes_result['posterior']
    
    # Level 1: Start
    ax.plot([2, 2], [5, 5], 'ko', markersize=12)
    ax.text(1.5, 5, 'Start', fontsize=11, va='center')
    
    # Level 2: Damage / No Damage
    ax.plot([2, 4], [5, 7], 'b-', linewidth=2)
    ax.plot([2, 4], [5, 3], 'r-', linewidth=2)
    ax.plot([4, 4], [7, 7], 'bo', markersize=10)
    ax.plot([4, 4], [3, 3], 'ro', markersize=10)
    
    ax.text(3, 6.2, f'Damage\nP={prior:.2%}', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(3, 3.8, f'No Damage\nP={(1-prior):.2%}', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Level 3: Test Results from Damage
    ax.plot([4, 6], [7, 7.8], 'g-', linewidth=2)
    ax.plot([4, 6], [7, 6.2], 'orange', linewidth=1.5)
    ax.plot([6, 6], [7.8, 7.8], 'go', markersize=8)
    ax.plot([6, 6], [6.2, 6.2], marker='o', color='orange', markersize=8)
    
    ax.text(5, 7.5, 'Pos: 95%', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
    ax.text(5, 6.5, 'Neg: 5%', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
    
    # Level 3: Test Results from No Damage
    ax.plot([4, 6], [3, 3.8], 'orange', linewidth=1.5)
    ax.plot([4, 6], [3, 2.2], 'g-', linewidth=2)
    ax.plot([6, 6], [3.8, 3.8], marker='o', color='orange', markersize=8)
    ax.plot([6, 6], [2.2, 2.2], 'go', markersize=8)
    
    ax.text(5, 3.5, 'Pos: 10%', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
    ax.text(5, 2.5, 'Neg: 90%', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
    
    # Result Box
    result_text = f"""
    BAYES' THEOREM RESULT:
    
    P(Damage | Positive Test)
    = {posterior:.4f} ({posterior*100:.1f}%)
    
    Key Insight: Even with positive test,
    only {posterior*100:.1f}% chance of damage!
    
    Why? Low base rate (5%) +
    False positives exist (10%)
    """
    
    ax.text(5, 0.8, result_text, ha='center', va='center', fontsize=10,
            family='monospace', bbox=dict(boxstyle='round', facecolor='yellow', 
                                         edgecolor='red', linewidth=2, alpha=0.8))
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(save_path)}")


def plot_statistical_dashboard(stats_dict, save_path):
    """Create simple statistical summary dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Statistical Summary Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Central Tendency
    ax = axes[0, 0]
    ax.axis('off')
    ax.text(0.5, 0.9, 'Central Tendency', ha='center', fontsize=14, 
            fontweight='bold', transform=ax.transAxes)
    text = f"""
    Mean:   {stats_dict['mean']:.2f} MPa
    Median: {stats_dict['median']:.2f} MPa
    Mode:   {stats_dict['mode']:.2f} MPa
    """
    ax.text(0.1, 0.5, text, fontsize=12, transform=ax.transAxes, 
            verticalalignment='center', family='monospace')
    
    # 2. Spread
    ax = axes[0, 1]
    ax.axis('off')
    ax.text(0.5, 0.9, 'Spread Measures', ha='center', fontsize=14, 
            fontweight='bold', transform=ax.transAxes)
    text = f"""
    Std Dev: {stats_dict['std']:.2f} MPa
    Variance: {stats_dict['variance']:.2f}
    Range:   {stats_dict['range']:.2f} MPa
    IQR:     {stats_dict['iqr']:.2f} MPa
    """
    ax.text(0.1, 0.5, text, fontsize=12, transform=ax.transAxes, 
            verticalalignment='center', family='monospace')
    
    # 3. Shape
    ax = axes[1, 0]
    ax.axis('off')
    ax.text(0.5, 0.9, 'Shape Measures', ha='center', fontsize=14, 
            fontweight='bold', transform=ax.transAxes)
    text = f"""
    Skewness: {stats_dict['skewness']:.3f}
    Kurtosis: {stats_dict['kurtosis']:.3f}
    
    {"Symmetric" if abs(stats_dict['skewness']) < 0.5 else "Skewed"}
    {"Normal tails" if abs(stats_dict['kurtosis']) < 0.5 else "Heavy/Light tails"}
    """
    ax.text(0.1, 0.5, text, fontsize=12, transform=ax.transAxes, 
            verticalalignment='center', family='monospace')
    
    # 4. Five-Number Summary (bar chart)
    ax = axes[1, 1]
    five_num = [stats_dict['min'], stats_dict['q1'], stats_dict['median'], 
                stats_dict['q3'], stats_dict['max']]
    labels = ['Min', 'Q1', 'Median', 'Q3', 'Max']
    colors = ['red', 'orange', 'green', 'orange', 'red']
    bars = ax.bar(labels, five_num, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Strength (MPa)')
    ax.set_title('Five-Number Summary', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(save_path)}")


# ==============================================================================
# PART 6: REPORT GENERATION
# ==============================================================================

def create_statistical_report(concrete_data, material_data, output_file):
    """Generate comprehensive text report with all results."""
    stats_concrete = calculate_descriptive_stats(concrete_data, 'strength_mpa')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LAB 4: STATISTICAL ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Descriptive Statistics
        f.write("PART 1: CONCRETE STRENGTH STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean:     {stats_concrete['mean']:.2f} MPa\n")
        f.write(f"Median:   {stats_concrete['median']:.2f} MPa\n")
        f.write(f"Mode:     {stats_concrete['mode']:.2f} MPa\n")
        f.write(f"Std Dev:  {stats_concrete['std']:.2f} MPa\n")
        f.write(f"Variance: {stats_concrete['variance']:.2f}\n")
        f.write(f"Range:    {stats_concrete['range']:.2f} MPa\n")
        f.write(f"IQR:      {stats_concrete['iqr']:.2f} MPa\n")
        f.write(f"Skewness: {stats_concrete['skewness']:.3f}\n")
        f.write(f"Kurtosis: {stats_concrete['kurtosis']:.3f}\n\n")
        
        # Five-Number Summary
        f.write("Five-Number Summary:\n")
        f.write(f"  Min: {stats_concrete['min']:.2f} MPa\n")
        f.write(f"  Q1:  {stats_concrete['q1']:.2f} MPa\n")
        f.write(f"  Q2:  {stats_concrete['median']:.2f} MPa\n")
        f.write(f"  Q3:  {stats_concrete['q3']:.2f} MPa\n")
        f.write(f"  Max: {stats_concrete['max']:.2f} MPa\n\n")
        
        f.write("="*80 + "\n\n")
        
        # Material Comparison
        f.write("PART 2: MATERIAL COMPARISON\n")
        f.write("-"*80 + "\n")
        for material in material_data['material_type'].unique():
            subset = material_data[material_data['material_type'] == material]
            stats_mat = calculate_descriptive_stats(subset, 'yield_strength_mpa')
            f.write(f"\n{material}:\n")
            f.write(f"  Mean: {stats_mat['mean']:.2f} MPa\n")
            f.write(f"  Std:  {stats_mat['std']:.2f} MPa\n")
            f.write(f"  CV:   {(stats_mat['std']/stats_mat['mean']*100):.2f}%\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Probability Distributions
        f.write("PART 3: PROBABILITY CALCULATIONS\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Binomial (n=100, p=0.05):\n")
        f.write(f"  P(X=3)  = {calculate_probability_binomial(100, 0.05, 3):.4f}\n")
        f.write(f"  P(X≤5)  = {calculate_probability_binomial(100, 0.05, '<=5'):.4f}\n\n")
        
        f.write("Poisson (λ=10):\n")
        f.write(f"  P(X=8)  = {calculate_probability_poisson(10, 8):.4f}\n")
        f.write(f"  P(X>15) = {calculate_probability_poisson(10, '>15'):.4f}\n\n")
        
        f.write("Normal (μ=250, σ=15):\n")
        f.write(f"  P(X>280) = {calculate_probability_normal(250, 15, x_lower=280):.4f}\n")
        f.write(f"  95th percentile = {norm.ppf(0.95, 250, 15):.2f} MPa\n\n")
        
        exp_500 = calculate_probability_exponential(1000, 500)
        exp_1500 = calculate_probability_exponential(1000, 1500)
        f.write("Exponential (mean=1000h):\n")
        f.write(f"  P(T<500)  = {exp_500['prob_before']:.4f}\n")
        f.write(f"  P(T>1500) = {exp_1500['prob_after']:.4f}\n\n")
        
        f.write("="*80 + "\n\n")
        
        # Bayes' Theorem
        bayes = apply_bayes_theorem(0.05, 0.95, 0.90)
        f.write("PART 4: BAYES' THEOREM\n")
        f.write("-"*80 + "\n")
        f.write(f"Prior:     {bayes['prior']:.2%}\n")
        f.write(f"Posterior: {bayes['posterior']:.4f} ({bayes['posterior']*100:.2f}%)\n\n")
        f.write("Interpretation:\n")
        f.write(f"  Even with positive test, only {bayes['posterior']*100:.1f}% chance of damage.\n")
        f.write("  This shows importance of considering base rates!\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"  ✓ Saved: {os.path.basename(output_file)}")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """Main function that runs all analyses."""
    
    print("\n" + "="*80)
    print("LAB 4: STATISTICAL ANALYSIS")
    print("="*80 + "\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # STEP 1: Load data
    print("Step 1: Loading Data")
    print("-"*80)
    concrete_data = load_data('concrete_strength.csv')
    material_data = load_data('material_properties.csv')
    
    if concrete_data is None or material_data is None:
        print("Error: Could not load data files!")
        return
    
    # STEP 2: Descriptive Statistics
    print("\nStep 2: Descriptive Statistics")
    print("-"*80)
    stats = calculate_descriptive_stats(concrete_data, 'strength_mpa')
    
    print(f"\nCentral Tendency:")
    print(f"  Mean:   {stats['mean']:.2f} MPa")
    print(f"  Median: {stats['median']:.2f} MPa")
    print(f"  Mode:   {stats['mode']:.2f} MPa")
    
    print(f"\nSpread:")
    print(f"  Std Dev: {stats['std']:.2f} MPa")
    print(f"  IQR:     {stats['iqr']:.2f} MPa")
    
    print(f"\nShape:")
    print(f"  Skewness: {stats['skewness']:.3f}")
    print(f"  Kurtosis: {stats['kurtosis']:.3f}")
    
    # STEP 3: Probability Distributions
    print("\nStep 3: Probability Distributions")
    print("-"*80)
    
    print("\nBinomial (n=100, p=0.05):")
    print(f"  P(X=3)  = {calculate_probability_binomial(100, 0.05, 3):.4f}")
    print(f"  P(X≤5)  = {calculate_probability_binomial(100, 0.05, '<=5'):.4f}")
    
    print("\nPoisson (λ=10):")
    print(f"  P(X=8)  = {calculate_probability_poisson(10, 8):.4f}")
    print(f"  P(X>15) = {calculate_probability_poisson(10, '>15'):.4f}")
    
    print("\nNormal (μ=250, σ=15):")
    print(f"  P(X>280) = {calculate_probability_normal(250, 15, x_lower=280):.4f}")
    print(f"  95th %ile = {norm.ppf(0.95, 250, 15):.2f} MPa")
    
    print("\nExponential (mean=1000):")
    exp_result = calculate_probability_exponential(1000, 500)
    print(f"  P(T<500)  = {exp_result['prob_before']:.4f}")
    print(f"  P(T>1500) = {calculate_probability_exponential(1000, 1500)['prob_after']:.4f}")
    
    # STEP 4: Bayes' Theorem
    print("\nStep 4: Bayes' Theorem")
    print("-"*80)
    bayes = apply_bayes_theorem(0.05, 0.95, 0.90)
    print(f"\nPrior:     {bayes['prior']:.2%}")
    print(f"Posterior: {bayes['posterior']:.4f} ({bayes['posterior']*100:.1f}%)")
    print(f"\nInterpretation: Even with positive test, only {bayes['posterior']*100:.1f}% chance of damage!")
    
    # STEP 5: Material Comparison
    print("\nStep 5: Material Comparison")
    print("-"*80)
    for material in material_data['material_type'].unique():
        subset = material_data[material_data['material_type'] == material]
        mat_stats = calculate_descriptive_stats(subset, 'yield_strength_mpa')
        print(f"\n{material}: μ={mat_stats['mean']:.2f}, σ={mat_stats['std']:.2f} MPa")
    
    # STEP 6: Distribution Fitting
    print("\nStep 6: Distribution Fitting")
    print("-"*80)
    fitted = fit_distribution(concrete_data, 'strength_mpa', 'normal')
    
    # STEP 7: Create Visualizations
    print("\nStep 7: Creating Visualizations")
    print("-"*80)
    
    # Visualization 1: Histogram with central tendency
    plot_distribution(concrete_data, 'strength_mpa',
                     'Concrete Strength Distribution',
                     os.path.join(script_dir, 'concrete_strength_distribution.png'))
    
    # Visualization 2: Boxplot for concrete (Task 1)
    plot_concrete_boxplot(concrete_data, 'strength_mpa',
                         os.path.join(script_dir, 'concrete_strength_boxplot.png'))
    
    # Visualization 3: Material comparison
    plot_material_comparison(material_data, 'yield_strength_mpa', 'material_type',
                           os.path.join(script_dir, 'material_comparison_boxplot.png'))
    
    # Visualization 4: Probability distributions (all)
    plot_probability_distributions(os.path.join(script_dir, 'probability_distributions.png'))
    
    # Visualization 5: Distribution fitting with synthetic data (Task 5)
    plot_distribution_fitting(concrete_data, 'strength_mpa', fitted,
                            os.path.join(script_dir, 'distribution_fitting.png'))
    
    # Visualization 6: Probability tree for Bayes (Task 4)
    plot_probability_tree(bayes,
                         os.path.join(script_dir, 'probability_tree_bayes.png'))
    
    # Visualization 7: Statistical dashboard
    plot_statistical_dashboard(stats,
                              os.path.join(script_dir, 'statistical_summary_dashboard.png'))
    
    # STEP 8: Generate Report
    print("\nStep 8: Creating Report")
    print("-"*80)
    create_statistical_report(concrete_data, material_data,
                            os.path.join(script_dir, 'lab4_statistical_report.txt'))
    
    # DONE!
    print("\n" + "="*80)
    print("✓✓✓ LAB 4 COMPLETED SUCCESSFULLY - ALL TASKS DONE! ✓✓✓")
    print("="*80)
    print("\nGenerated Files (7 Visualizations + 1 Report):")
    print("  1. concrete_strength_distribution.png")
    print("  2. concrete_strength_boxplot.png (Task 1)")
    print("  3. material_comparison_boxplot.png")
    print("  4. probability_distributions.png")
    print("  5. distribution_fitting.png (with synthetic data - Task 5)")
    print("  6. probability_tree_bayes.png (Task 4)")
    print("  7. statistical_summary_dashboard.png")
    print("  8. lab4_statistical_report.txt")
    print("\n✓ All Lab Requirements Completed:")
    print("  ✓ Task 1: Concrete Analysis (with boxplot)")
    print("  ✓ Task 2: Material Comparison")
    print("  ✓ Task 3: Probability Modeling (all scenarios)")
    print("  ✓ Task 4: Bayes' Theorem (with probability tree)")
    print("  ✓ Task 5: Distribution Fitting (with synthetic data)")
    print("  ✓ All Required Functions")
    print("  ✓ 7 Professional Visualizations")
    print("  ✓ Comprehensive Report")
    print()


# ==============================================================================
# RUN THE PROGRAM
# ==============================================================================

if __name__ == "__main__":
    main()
