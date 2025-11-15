import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# --- Configuration ---
INPUT_FILE = "gemm_results.csv"
OUTPUT_DIR = "plots"
# ---------------------

def load_data(filepath):
    """
    Loads and prepares the benchmark data from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.", file=sys.stderr)
        print("Please run './run_all_tests.sh' first.", file=sys.stderr)
        return None
    
    if df.empty:
        print(f"Error: {filepath} is empty.", file=sys.stderr)
        return None

    # Ensure correct data types for plotting
    try:
        df['N'] = df['N'].astype(int)
        df['T'] = df['T'].astype(int)
        df['Time_Opt'] = df['Time_Opt'].astype(float)
        df['GFLOPS_Opt'] = df['GFLOPS_Opt'].astype(float)
        df['Time_Base'] = df['Time_Base'].astype(float)
        df['Speedup'] = df['Speedup'].astype(float)
    except KeyError as e:
        print(f"Error: Missing expected column {e}.", file=sys.stderr)
        print("CSV header should be: N,T,Time_Opt,GFLOPS_Opt,Time_Base,Speedup", file=sys.stderr)
        return None
        
    print("Data loaded successfully.")
    return df

def setup_plotting(output_dir):
    """
    Sets up seaborn theme and creates the output directory.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Plots will be saved to '{output_dir}/'")

def plot_gflops_vs_threads(df, output_dir):
    """
    Plot 1: Performance (GFLOPS) vs. Thread Count (Scalability)
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x='T',
        y='GFLOPS_Opt',
        hue='N',
        palette='viridis',
        marker='o'
    )
    plt.title('Performance vs. Thread Count (Optimized)', fontsize=16)
    plt.xlabel('Number of Threads (T)', fontsize=12)
    plt.ylabel('Performance (GFLOPS)', fontsize=12)
    plt.legend(title='Matrix Size (N)')
    plt.xticks(df['T'].unique())
    
    filename = os.path.join(output_dir, 'plot_gflops_vs_threads.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_gflops_vs_size(df, output_dir):
    """
    Plot 2: Performance (GFLOPS) vs. Problem Size (Cache Effects)
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x='N',
        y='GFLOPS_Opt',
        hue='T',
        palette='crest',
        marker='o'
    )
    plt.title('Performance vs. Problem Size (Optimized)', fontsize=16)
    plt.xlabel('Matrix Size (N)', fontsize=12)
    plt.ylabel('Performance (GFLOPS)', fontsize=12)
    plt.legend(title='Thread Count (T)')
    plt.xticks(df['N'].unique())
    
    filename = os.path.join(output_dir, 'plot_gflops_vs_size.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_time_vs_size(df, output_dir):
    """
    Plot 3: Execution Time vs. Problem Size (Log Scale)
    """
    plt.figure(figsize=(10, 6))
    
    # Plot Optimized time for all thread counts
    sns.lineplot(
        data=df, x='N', y='Time_Opt', hue='T',
        palette='flare', marker='o'
    )
    
    # Get baseline time (it's repeated, so just take one per N)
    df_base = df[['N', 'Time_Base']].drop_duplicates().sort_values('N')
    
    # Plot Baseline time
    sns.lineplot(
        data=df_base, x='N', y='Time_Base',
        color='black', linestyle='--', marker='x', label='Baseline'
    )
    
    plt.title('Execution Time vs. Problem Size', fontsize=16)
    plt.xlabel('Matrix Size (N)', fontsize=12)
    plt.ylabel('Time (seconds) - Log Scale', fontsize=12)
    plt.legend(title='Threads (T)')
    plt.yscale('log')
    plt.xticks(df['N'].unique())
    
    filename = os.path.join(output_dir, 'plot_time_vs_size.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_speedup_vs_threads(df, output_dir):
    """
    Plot 4: Speedup vs. Thread Count (Scalability)
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x='T',
        y='Speedup',
        hue='N',
        palette='viridis',
        marker='o'
    )
    plt.title('Speedup vs. Thread Count', fontsize=16)
    plt.xlabel('Number of Threads (T)', fontsize=12)
    plt.ylabel('Speedup (Baseline / Optimized)', fontsize=12)
    plt.legend(title='Matrix Size (N)')
    plt.xticks(df['T'].unique())
    
    filename = os.path.join(output_dir, 'plot_speedup_vs_threads.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_speedup_vs_size(df, output_dir):
    """
    Plot 5: Speedup vs. Problem Size
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x='N',
        y='Speedup',
        hue='T',
        palette='crest',
        marker='o'
    )
    plt.title('Speedup vs. Problem Size', fontsize=16)
    plt.xlabel('Matrix Size (N)', fontsize=12)
    plt.ylabel('Speedup (Baseline / Optimized)', fontsize=12)
    plt.legend(title='Thread Count (T)')
    plt.xticks(df['N'].unique())
    
    filename = os.path.join(output_dir, 'plot_speedup_vs_size.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def main():
    """
    Main function to load data and generate all plots.
    """
    df = load_data(INPUT_FILE)
    
    if df is not None:
        setup_plotting(OUTPUT_DIR)
        
        # Generate all plots
        plot_gflops_vs_threads(df, OUTPUT_DIR)
        plot_gflops_vs_size(df, OUTPUT_DIR)
        plot_time_vs_size(df, OUTPUT_DIR)
        plot_speedup_vs_threads(df, OUTPUT_DIR)
        plot_speedup_vs_size(df, OUTPUT_DIR)
        
        print("\nAll plots generated successfully.")

if __name__ == "__main__":
    main()