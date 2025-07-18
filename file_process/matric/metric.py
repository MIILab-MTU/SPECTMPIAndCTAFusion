import pandas as pd
import numpy as np
import os

def load_and_process_data(input_file):
    try:
        df = pd.read_excel(input_file, engine='openpyxl')
    except FileNotFoundError:
        print(f"Error: Input file {input_file} does not exist!")
        return None, None, None, None
    except Exception as e:
        print(f"Error reading {input_file}: {str(e)}")
        return None, None, None, None
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if not numeric_columns:
        print("Error: No numeric columns found!")
        return None, None, None, None
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(5)
    df_auto = df[df['ori'] == 'auto'][['patient_name'] + numeric_columns].dropna()
    df_manual = df[df['ori'] == 'manual'][['patient_name'] + numeric_columns].dropna()
    unique_names = df['patient_name'].unique()
    name_to_number = {name: str(i + 1) for i, name in enumerate(unique_names)}
    df_auto['number'] = df_auto['patient_name'].map(name_to_number)
    df_manual['number'] = df_manual['patient_name'].map(name_to_number)
    return df, df_auto, df_manual, numeric_columns

def calculate_statistics(df_auto, df_manual, numeric_columns):
    stats = []
    for col in numeric_columns:
        if not df_auto[col].empty:
            auto_mean = df_auto[col].mean()
            auto_var = df_auto[col].var(ddof=1)
            stats.append({
                'Column': col,
                'Type': 'auto',
                'Mean': round(auto_mean, 5),
                'Variance': round(auto_var, 5)
            })
        else:
            print(f"Warning: Column {col} auto data is empty.")
        if not df_manual[col].empty:
            manual_mean = df_manual[col].mean()
            manual_var = df_manual[col].var(ddof=1)
            stats.append({
                'Column': col,
                'Type': 'manual',
                'Mean': round(manual_mean, 5),
                'Variance': round(manual_var, 5)
            })
        else:
            print(f"Warning: Column {col} manual data is empty.")
    return pd.DataFrame(stats)

def main():
    input_file = r"eval_results_r5colorcpdpp.xlsx"
    output_file = f"stats_summary_cpdpp.xlsx"
    df, df_auto, df_manual, numeric_columns = load_and_process_data(input_file)
    if df is None:
        return
    stats_df = calculate_statistics(df_auto, df_manual, numeric_columns)
    if stats_df.empty:
        print("Error: Unable to calculate statistics, data is empty!")
        return
    print("\nData statistics results:")
    for col in numeric_columns:
        print(f"\nColumn: {col}")
        auto_stats = stats_df[(stats_df['Column'] == col) & (stats_df['Type'] == 'auto')]
        manual_stats = stats_df[(stats_df['Column'] == col) & (stats_df['Type'] == 'manual')]
        if not auto_stats.empty:
            print(f"  Auto - Mean: {auto_stats['Mean'].iloc[0]:.5f}, Variance: {auto_stats['Variance'].iloc[0]:.5f}")
        if not manual_stats.empty:
            print(f"  Manual - Mean: {manual_stats['Mean'].iloc[0]:.5f}, Variance: {manual_stats['Variance'].iloc[0]:.5f}")
    try:
        stats_df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\nStatistics results saved to {output_file}")
    except Exception as e:
        print(f"Error saving {output_file}: {str(e)}")

if __name__ == '__main__':
    main()