import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_process_data(input_file, column='affine_mse'):
    try:
        df = pd.read_excel(input_file, engine='openpyxl')
    except FileNotFoundError:
        print(f"Error: Input file {input_file} does not exist!")
        return None, None, None, None
    except Exception as e:
        print(f"Error reading {input_file}: {str(e)}")
        return None, None, None, None
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if column not in numeric_columns:
        print(f"Error: Column {column} is not numeric! Available numeric columns: {numeric_columns}")
        return None, None, None, numeric_columns
    df[column] = pd.to_numeric(df[column], errors='coerce').round(5)
    df_auto = df[df['ori'] == 'auto'][['patient_name', column]].dropna()
    df_manual = df[df['ori'] == 'manual'][['patient_name', column]].dropna()
    unique_names = df['patient_name'].unique()
    name_to_number = {name: str(i + 1) for i, name in enumerate(unique_names)}
    df_auto['number'] = df_auto['patient_name'].map(name_to_number)
    df_manual['number'] = df_manual['patient_name'].map(name_to_number)
    return df, df_auto, df_manual, numeric_columns

def sort_by_difference(df_auto, df_manual, column):
    df_merged = pd.merge(
        df_auto.rename(columns={column: 'auto_value'}),
        df_manual.rename(columns={column: 'manual_value'}),
        on=['patient_name', 'number'],
        how='inner'
    )
    df_merged['diff_value'] = df_merged['manual_value'] - df_merged['auto_value']
    df_merged['abs_diff'] = df_merged['diff_value'].abs()
    df_merged = df_merged.sort_values(by='diff_value', ascending=False)
    sorted_numbers = df_merged['number'].tolist()
    if not df_merged.empty:
        min_abs_diff_number = df_merged.loc[df_merged['abs_diff'].idxmin(), 'number']
    else:
        min_abs_diff_number = None
        print("Warning: No paired auto and manual data available for sorting!")
    return sorted_numbers, min_abs_diff_number

def plot_scatter(df_auto, df_manual, sorted_numbers, output_file, column):
    plt.figure(figsize=(12, 6))
    if not df_auto.empty:
        plt.scatter(df_auto['number'], df_auto[column], color='red', marker='o', label='Auto', s=100)
    else:
        print("Warning: No valid 'auto' data found.")
    if not df_manual.empty:
        plt.scatter(df_manual['number'], df_manual[column], color='blue', marker='o', label='Manual', s=100)
    else:
        print("Warning: No valid 'manual' data found.")
    plt.title(f'Data {column.upper()} Scatter Plot')
    plt.xlabel('Data Number')
    plt.ylabel(column.upper())
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Scatter plot saved to {output_file}")
    except Exception as e:
        print(f"Error saving {output_file}: {str(e)}")

def plot_line(df_auto, df_manual, sorted_numbers, output_file, column, min_abs_diff_number):
    plt.figure(figsize=(12, 6))
    if not df_auto.empty:
        plt.plot(df_auto['number'], df_auto[column], color='red', marker='o', label='Auto')
    else:
        print("Warning: No valid 'auto' data found.")
    if not df_manual.empty:
        plt.plot(df_manual['number'], df_manual[column], color='blue', marker='o', label='Manual')
    else:
        print("Warning: No valid 'manual' data found.")
    if min_abs_diff_number and sorted_numbers:
        min_abs_diff_idx = sorted_numbers.index(min_abs_diff_number)
        plt.axvspan(-0.5, min_abs_diff_idx + 0.5, facecolor='red', alpha=0.3, zorder=0)
        plt.axvspan(min_abs_diff_idx + 0.5, len(sorted_numbers) - 0.5, facecolor='blue', alpha=0.3, zorder=0)
    plt.title(f'Data {column.upper()} Line Plot (Sorted by Manual-Auto Difference)')
    plt.xlabel('Data Number')
    plt.ylabel(column.upper())
    plt.xticks(sorted_numbers, rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Line plot saved to {output_file}")
    except Exception as e:
        print(f"Error saving {output_file}: {str(e)}")

def plot_bar(df_auto, df_manual, sorted_numbers, output_file, column):
    plt.figure(figsize=(14, 6))
    bar_width = 0.35
    x = np.arange(len(sorted_numbers))
    if not df_auto.empty:
        plt.bar(x - bar_width / 2, df_auto[column], bar_width, color='red', label='Auto')
    else:
        print("Warning: No valid 'auto' data found.")
    if not df_manual.empty:
        plt.bar(x + bar_width / 2, df_manual[column], bar_width, color='blue', label='Manual')
    else:
        print("Warning: No valid 'manual' data found.")
    plt.title(f'Data {column.upper()} Comparison (Sorted by Manual-Auto Difference)')
    plt.xlabel('Data Number')
    plt.ylabel(column.upper())
    plt.xticks(x, sorted_numbers, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    try:
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Bar plot saved to {output_file}")
    except Exception as e:
        print(f"Error saving {output_file}: {str(e)}")

def plot_scatter_with_lines(df_auto, df_manual, sorted_numbers, output_file, column):
    plt.figure(figsize=(12, 6))
    if not df_auto.empty:
        plt.scatter(df_auto['number'], df_auto[column], color='red', marker='o', label='Auto', s=100)
    else:
        print("Warning: No valid 'auto' data found.")
    if not df_manual.empty:
        plt.scatter(df_manual['number'], df_manual[column], color='blue', marker='o', label='Manual', s=100)
    else:
        print("Warning: No valid 'manual' data found.")
    for i, number in enumerate(sorted_numbers):
        auto_value = df_auto[df_auto['number'] == number][column].values
        manual_value = df_manual[df_manual['number'] == number][column].values
        if len(auto_value) > 0 and len(manual_value) > 0:
            plt.plot([i, i], [auto_value[0], manual_value[0]], color='gray', linestyle='--')
    plt.title(f'Data {column.upper()} Scatter Plot (Sorted by Manual-Auto Difference)')
    plt.xlabel('Data Number')
    plt.ylabel(column.upper())
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Scatter plot with lines saved to {output_file}")
    except Exception as e:
        print(f"Error saving {output_file}: {str(e)}")

def main(plot_type='bar', sort_by_diff=True, column='affine_mse'):
    input_file = r"eval_results_r5colorcpdpp.xlsx"
    output_dir = r"path/to/plots/"
    os.makedirs(output_dir, exist_ok=True)
    df, df_auto, df_manual, numeric_columns = load_and_process_data(input_file, column)
    if df is None:
        return
    if sort_by_diff:
        sorted_numbers, min_abs_diff_number = sort_by_difference(df_auto, df_manual, column)
        if not sorted_numbers:
            return
        df_auto = df_auto[df_auto['number'].isin(sorted_numbers)]
        df_manual = df_manual[df_manual['number'].isin(sorted_numbers)]
        df_auto['number'] = pd.Categorical(df_auto['number'], categories=sorted_numbers, ordered=True)
        df_manual['number'] = pd.Categorical(df_manual['number'], categories=sorted_numbers, ordered=True)
        df_auto = df_auto.sort_values('number')
        df_manual = df_manual.sort_values('number')
    else:
        sorted_numbers = df_auto['number'].unique().tolist()
        sorted_numbers.sort()
        df_auto = df_auto.sort_values('number')
        df_manual = df_manual.sort_values('number')
        min_abs_diff_number = None
    plot_functions = {
        'scatter': plot_scatter,
        'line': lambda df_auto, df_manual, sorted_numbers, output_file, column: plot_line(
            df_auto, df_manual, sorted_numbers, output_file, column, min_abs_diff_number
        ),
        'bar': plot_bar,
        'scatter_with_lines': plot_scatter_with_lines
    }
    output_files = {
        'scatter': f"{output_dir}{column}_scatterplot.png",
        'line': f"{output_dir}{column}_lineplot.png",
        'bar': f"{output_dir}{column}_barplot.png",
        'scatter_with_lines': f"{output_dir}{column}_scatterplot_with_lines.png"
    }
    if plot_type not in plot_functions:
        print(f"Error: Invalid plot type {plot_type}! Supported: {list(plot_functions.keys())}")
        return
    plot_functions[plot_type](df_auto, df_manual, sorted_numbers, output_files[plot_type], column)

if __name__ == '__main__':
    main(plot_type='line', sort_by_diff=True, column='bcpdpp_jl')