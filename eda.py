# eda_script.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings

# Suppress specific warnings that might arise from plotting empty data or statistical calculations
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Assume utils.py is in the same directory for load_and_preprocess_data
# For this script, we'll simplify to directly load the discharge data
# from discharge_data_cleaned.csv if utils.py's load_and_preprocess_data
# is complex or not readily available for standalone EDA.
# If load_and_preprocess_data is necessary due to specific cleaning in utils,
# ensure utils.py is accessible and its dependencies met.
# For simplicity and directness in EDA on the _cleaned_ CSV, we'll read it directly.

def load_discharge_data(file_path="discharge_data_cleaned.csv"):
    """
    Loads the discharge data, sets 'Date' as index, and ensures numeric types.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}. Please ensure it's in the same directory.")
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=True)
    df = df.set_index("Date").select_dtypes(include=[np.number])
    # Ensure column names are clean if they weren't already
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

def perform_eda(df_discharge, output_dir="eda_results"):
    """
    Performs Exploratory Data Analysis on the discharge DataFrame.
    Generates summary statistics, temporal trends, correlation analysis,
    and saves them to a JSON file and plots to sub-directories.

    Args:
        df_discharge (pd.DataFrame): The preprocessed discharge data.
        output_dir (str): Base directory to save EDA results (JSON and plots).
    """
    print("\n--- Starting EDA ---")
    
    # Create output directories
    json_output_path = os.path.join(output_dir, "eda_summary.json")
    plots_dir = os.path.join(output_dir, "plots")
    station_plots_dir = os.path.join(plots_dir, "station_plots")
    temporal_plots_dir = os.path.join(plots_dir, "temporal_plots")
    correlation_plots_dir = os.path.join(plots_dir, "correlation_plots")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(station_plots_dir, exist_ok=True)
    os.makedirs(temporal_plots_dir, exist_ok=True)
    os.makedirs(correlation_plots_dir, exist_ok=True)

    eda_summary = {}

    # 1. Dataset Overview
    print("\n1. Dataset Overview:")
    start_date = df_discharge.index.min().strftime('%Y-%m-%d')
    end_date = df_discharge.index.max().strftime('%Y-%m-%d')
    total_days = (df_discharge.index.max() - df_discharge.index.min()).days + 1
    num_stations = df_discharge.shape[1]
    overall_missing_percentage = df_discharge.isnull().sum().sum() / (df_discharge.shape[0] * df_discharge.shape[1]) * 100

    print(f"   Start Date: {start_date}")
    print(f"   End Date: {end_date}")
    print(f"   Total Days: {total_days}")
    print(f"   Number of Stations: {num_stations}")
    print(f"   Overall Missing Percentage: {overall_missing_percentage:.2f}%")
    print(f"   DataFrame Shape: {df_discharge.shape}")

    eda_summary["dataset_overview"] = {
        "start_date": start_date,
        "end_date": end_date,
        "total_days": total_days,
        "number_of_stations": num_stations,
        "overall_missing_percentage": round(overall_missing_percentage, 2)
    }

    # 2. Station-wise Statistics
    print("\n2. Station-wise Statistics:")
    station_stats = {}
    for col in df_discharge.columns:
        desc = df_discharge[col].describe()
        missing_count = df_discharge[col].isnull().sum()
        missing_perc = (missing_count / len(df_discharge)) * 100
        
        mean_val = desc.get('mean', np.nan)
        std_val = desc.get('std', np.nan)
        cv = (std_val / mean_val) if mean_val != 0 else np.nan

        station_stats[col] = {
            "mean_discharge": round(mean_val, 2),
            "median_discharge": round(df_discharge[col].median(), 2),
            "std_discharge": round(std_val, 2),
            "min_discharge": round(desc.get('min', np.nan), 2),
            "max_discharge": round(desc.get('max', np.nan), 2),
            "coeff_of_variation": round(cv, 2) if not np.isnan(cv) else None,
            "missing_percentage": round(missing_perc, 2)
        }
        print(f"   - {col}: Mean={station_stats[col]['mean_discharge']:.2f}, "
              f"Std={station_stats[col]['std_discharge']:.2f}, "
              f"Missing={station_stats[col]['missing_percentage']:.2f}%")
        
        # Plot histogram for each station
        plt.figure(figsize=(8, 5))
        sns.histplot(df_discharge[col].dropna(), kde=True, bins=50)
        plt.title(f'Discharge Distribution for {col}')
        plt.xlabel('Discharge')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.savefig(os.path.join(station_plots_dir, f'{col}_discharge_distribution.png'))
        plt.close()

    eda_summary["station_statistics"] = station_stats

    # 3. Temporal Trends
    print("\n3. Temporal Trends Analysis:")
    temporal_trends_summary = {}

    # Monthly Averages
    df_discharge['month'] = df_discharge.index.month
    monthly_avg = df_discharge.groupby('month').mean(numeric_only=True).mean(axis=1) # Mean across stations for each month
    temporal_trends_summary["monthly_averages_summary"] = monthly_avg.apply(lambda x: round(x, 2)).to_dict()
    print("   Monthly Average Discharge (across all stations):")
    print(monthly_avg)

    # Plot Monthly Averages (Box Plot)
    plt.figure(figsize=(12, 7))
    sns.boxplot(x=df_discharge.index.month, y=df_discharge.mean(axis=1), palette='viridis') # Mean daily discharge across stations
    plt.title('Monthly Distribution of Average Daily Discharge Across Stations')
    plt.xlabel('Month')
    plt.ylabel('Average Daily Discharge (Mean across stations)')
    plt.xticks(ticks=np.arange(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(os.path.join(temporal_plots_dir, 'monthly_discharge_boxplot.png'))
    plt.close()

    # Annual Averages
    df_discharge['year'] = df_discharge.index.year
    annual_avg = df_discharge.groupby('year').mean(numeric_only=True).mean(axis=1) # Mean across stations for each year
    temporal_trends_summary["annual_discharge_summary"] = annual_avg.apply(lambda x: round(x, 2)).to_dict()
    print("\n   Annual Average Discharge (across all stations):")
    print(annual_avg)

    # Plot Annual Averages (Line Plot)
    plt.figure(figsize=(10, 6))
    annual_avg.plot(marker='o', linestyle='-')
    plt.title('Annual Average Daily Discharge Across Stations')
    plt.xlabel('Year')
    plt.ylabel('Average Daily Discharge')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(os.path.join(temporal_plots_dir, 'annual_discharge_trend.png'))
    plt.close()

    # Overall Time Series for a few representative stations (e.g., top 3 by mean discharge)
    top_stations = df_discharge.mean(numeric_only=True).nlargest(3).index.tolist()
    if top_stations:
        plt.figure(figsize=(15, 8))
        for station in top_stations:
            df_discharge[station].plot(label=station, alpha=0.8)
        plt.title('Overall Discharge Trends for Top Stations')
        plt.xlabel('Date')
        plt.ylabel('Discharge')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(temporal_plots_dir, 'overall_discharge_trends.png'))
        plt.close()
    else:
        print("   No stations available to plot overall trends.")

    eda_summary["temporal_trends_summary"] = temporal_trends_summary

    # 4. Correlation Analysis
    print("\n4. Correlation Analysis:")
    correlation_matrix = df_discharge.drop(columns=['month', 'year'], errors='ignore').corr()
    
    # Fill NaN correlations with 0 or a very small number if they arise from no common non-NA data
    correlation_matrix = correlation_matrix.fillna(0)

    # Plot Correlation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Station Discharge Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(correlation_plots_dir, 'correlation_heatmap.png'))
    plt.close()

    # Extract highly/lowly correlated pairs
    correlation_pairs = correlation_matrix.stack().reset_index()
    correlation_pairs.columns = ['station1', 'station2', 'correlation']
    
    # Filter out self-correlations and duplicate pairs
    correlation_pairs = correlation_pairs[correlation_pairs['station1'] != correlation_pairs['station2']]
    correlation_pairs['sorted_pair'] = correlation_pairs.apply(lambda row: tuple(sorted([row['station1'], row['station2']])), axis=1)
    correlation_pairs = correlation_pairs.drop_duplicates(subset=['sorted_pair']).drop(columns=['sorted_pair'])

    # Sort by absolute correlation for top/bottom
    correlation_pairs['abs_correlation'] = correlation_pairs['correlation'].abs()
    top_correlated_pairs = correlation_pairs.nlargest(10, 'abs_correlation')
    low_correlated_pairs = correlation_pairs.nsmallest(10, 'abs_correlation') # Smallest (closest to 0 or negative)

    eda_summary["correlation_summary"] = {
        "top_10_correlated_pairs": top_correlated_pairs.drop(columns=['abs_correlation']).to_dict(orient='records'),
        "bottom_10_correlated_pairs": low_correlated_pairs.drop(columns=['abs_correlation']).to_dict(orient='records')
    }
    
    print("\nTop 10 Highly Correlated Station Pairs:")
    print(top_correlated_pairs[['station1', 'station2', 'correlation']].to_string(index=False, float_format="%.2f"))
    print("\nBottom 10 Correlated Station Pairs:")
    print(low_correlated_pairs[['station1', 'station2', 'correlation']].to_string(index=False, float_format="%.2f"))


    # Save summary to JSON
    with open(json_output_path, 'w') as f:
        json.dump(eda_summary, f, indent=4)
    print(f"\nEDA summary saved to: {json_output_path}")
    print(f"Plots saved to: {plots_dir}")
    print("\n--- EDA Complete ---")

if __name__ == "__main__":
    # Ensure discharge_data_cleaned.csv is in the same directory as this script.
    try:
        discharge_data = load_discharge_data()
        perform_eda(discharge_data)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure 'discharge_data_cleaned.csv' is in the same directory as 'eda_script.py'.")

