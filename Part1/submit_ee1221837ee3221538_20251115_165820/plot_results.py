import pandas as pd
import matplotlib.pyplot as plt
import os

# ======================
# CONFIGURATION
# ======================

CSV_FILE = "gemm_results.csv"
PLOT_DIR = "plots"

# ensure plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)


# ======================
# UTILITY HELPERS
# ======================

def save_plot(name):
    """Convenience wrapper to save figures into the plots/ directory."""
    plt.savefig(os.path.join(PLOT_DIR, name))
    plt.close()


def load_data():
    """Load CSV into a Pandas DataFrame."""
    df = pd.read_csv(CSV_FILE)
    return df


# ======================
# PLOTTING FUNCTIONS
# ======================

def plot_time_vs_size(df):
    plt.figure()
    plt.plot(df["N"], df["time_opt"], marker='o')
    plt.xlabel("Matrix Size N")
    plt.ylabel("Optimized Time (s)")
    plt.title("Optimized Runtime vs Matrix Size")
    save_plot("time_vs_size.png")


def plot_gflops_vs_size(df):
    plt.figure()
    plt.plot(df["N"], df["gflops_opt"], marker='o')
    plt.xlabel("Matrix Size N")
    plt.ylabel("GFLOPS")
    plt.title("GFLOPS vs Matrix Size")
    save_plot("gflops_vs_size.png")


def plot_speedup_vs_size(df):
    plt.figure()
    plt.plot(df["N"], df["speedup"], marker='o')
    plt.xlabel("Matrix Size N")
    plt.ylabel("Speedup (Baseline / Optimized)")
    plt.title("Speedup vs Matrix Size")
    save_plot("speedup_vs_size.png")


def plot_l1_miss_rate(df):
    plt.figure()
    plt.plot(df["N"], df["l1_miss_rate"], marker='o')
    plt.xlabel("Matrix Size N")
    plt.ylabel("L1 Miss Rate")
    plt.title("L1 Miss Rate vs Matrix Size")
    save_plot("l1_miss_rate_vs_n.png")


def plot_llc_miss_rate(df):
    plt.figure()
    plt.plot(df["N"], df["llc_miss_rate"], marker='o')
    plt.xlabel("Matrix Size N")
    plt.ylabel("LLC Miss Rate")
    plt.title("LLC Miss Rate vs Matrix Size")
    save_plot("llc_miss_rate_vs_n.png")


def plot_ipc_vs_threads(df):
    plt.figure()
    plt.plot(df["T"], df["ipc"], marker='o')
    plt.xlabel("Number of Threads")
    plt.ylabel("IPC (Instructions per Cycle)")
    plt.title("IPC vs Thread Count")
    save_plot("ipc_vs_threads.png")


def plot_branch_miss_rate(df):
    plt.figure()
    plt.plot(df["T"], df["branch_miss_rate"], marker='o')
    plt.xlabel("Number of Threads")
    plt.ylabel("Branch Miss Rate")
    plt.title("Branch Miss Rate vs Thread Count")
    save_plot("branch_miss_rate_vs_threads.png")


# Add-on modular plots (you can expand here):
def plot_llc_loads(df):
    plt.figure()
    plt.plot(df["N"], df["llc_load"], marker='o')
    plt.xlabel("Matrix Size N")
    plt.ylabel("LLC Loads")
    plt.title("LLC Loads vs Matrix Size")
    save_plot("llc_loads_vs_n.png")


# ======================
# MAIN DRIVER
# ======================

def main():
    df = load_data()

    # Core plots
    plot_time_vs_size(df)
    plot_gflops_vs_size(df)
    plot_speedup_vs_size(df)

    # Perf plots
    plot_l1_miss_rate(df)
    plot_llc_miss_rate(df)
    plot_ipc_vs_threads(df)
    plot_branch_miss_rate(df)

    # Optional modular plots
    # plot_llc_loads(df)

    print("All plots generated successfully into ./plots/")

if __name__ == "__main__":
    main()
