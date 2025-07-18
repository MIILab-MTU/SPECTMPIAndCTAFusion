import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_process_data(input_file, reference_column='affine_jl'):
    try:
        df = pd.read_excel(input_file, engine='openpyxl')
    except FileNotFoundError:
        print(f"Error: Input file {input_file} does not exist!")
        return None, None, None, None
    except Exception as e:
        print(f"Error reading {input_file}: {str(e)}")
        return None, None, None, None
    jl_columns = ['affine_jl', 'rigid_jl', 'icp_jl', 'sicp_jl', 'bcpdpp_jl']
    missing_columns = [col for col in jl_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns: {missing_columns}")
        return None, None, None, None
    if reference_column not in jl_columns:
        print(f"Error: Reference column {reference_column} not in {jl_columns}!")
        return None, None, None, None
    for col in jl_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(5)
    df_auto = df[df['ori'] == 'auto'][['patient_name'] + jl_columns].dropna()
    df_manual = df[df['ori'] == 'manual'][['patient_name'] + jl_columns].dropna()
    unique_names = df['patient_name'].unique()
    name_to_number = {name: str(i + 1) for i, name in enumerate(unique_names)}
    df_auto['number'] = df_auto['patient_name'].map(name_to_number)
    df_manual['number'] = df_manual['patient_name'].map(name_to_number)
    return df, df_auto, df_manual, jl_columns

def sort_by_difference(df_auto, df_manual, reference_column):
    df_merged = pd.merge(
        df_auto.rename(columns={reference_column: 'auto_value'}),
        df_manual.rename(columns={reference_column: 'manual_value'}),
        on=['patient_name', 'number'],
        how='inner'
    )
    df_merged['diff_value'] = df_merged['manual_value'] - df_merged['auto_value']
    df_merged = df_merged.sort_values(by='diff_value', ascending=False)
    sorted_numbers = df_merged['number'].tolist()
    if not sorted_numbers:
        print("Warning: No paired auto and manual data available for sorting!")
    return sorted_numbers

def plot_jl_lines(df_auto, df_manual, sorted_numbers, output_dir, jl_columns):
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(jl_columns):
        if not df_auto[col].empty:
            plt.plot(df_auto['number'], df_auto[col], color=colors[i], marker='o', label=col.upper())
        else:
            print(f"Warning: Column {col} auto data is empty.")
    plt.title('Data JL Metrics (Auto) Line Plot')
    plt.xlabel('Data Number')
    plt.ylabel('JL Metric Value')
    plt.xticks(sorted_numbers, rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    auto_output = f"{output_dir}jl_auto_lineplot.png"
    try:
        plt.savefig(auto_output, dpi=300)
        plt.close()
        print(f"Auto line plot saved to {auto_output}")
    except Exception as e:
        print(f"Error saving {auto_output}: {str(e)}")
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(jl_columns):
        if not df_manual[col].empty:
            plt.plot(df_manual['number'], df_manual[col], color=colors[i], marker='o', label=col.upper())
        else:
            print(f"Warning: Column {col} manual data is empty.")
    plt.title('Data JL Metrics (Manual) Line Plot')
    plt.xlabel('Data Number')
    plt.ylabel('JL Metric Value')
    plt.xticks(sorted_numbers, rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    manual_output = f"{output_dir}jl_manual_lineplot.png"
    try:
        plt.savefig(manual_output, dpi=300)
        plt.close()
        print(f"Manual line plot saved to {manual_output}")
    except Exception as e:
        print(f"Error saving {manual_output}: {str(e)}")

def main():
    input_file = r"eval_results_r5colorcpdpp.xlsx"
    output_dir = r"path/to/plots/"
    reference_column = 'bcpdpp_jl'
    os.makedirs(output_dir, exist_ok=True)
    df, df_auto, df_manual, jl_columns = load_and_process_data(input_file, reference_column)
    if df is None:
        return
    sorted_numbers = sort_by_difference(df_auto, df_manual, reference_column)
    if not sorted_numbers:
        return
    df_auto = df_auto[df_auto['number'].isin(sorted_numbers)]
    df_manual = df_manual[df_manual['number'].isin(sorted_numbers)]
    df_auto['number'] = pd.Categorical(df_auto['number'], categories=sorted_numbers, ordered=True)
    df_manual['number'] = pd.Categorical(df_manual['number'], categories=sorted_numbers, ordered=True)
    df_auto = df_auto.sort_values('number')
    df_manual = df_manual.sort_values('number')
    plot_jl_lines(df_auto, df_manual, sorted_numbers, output_dir, jl_columns)

if __name__ == '__main__':
    main()