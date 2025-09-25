#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# === STYLE CONFIGURATION ===
PROGRAM_COLORS = {
    "hns_get": "#1f77b4",  # blue
    "hns_main": "#ff7f0e",  # orange
    "trilinos": "#2ca02c",  # green
}

TIMER_PATTERNS = {
    "comm_wait": "//",
    "comp_time": "",
    "data_proc_A": "\\\\",
    "data_proc_B": "..",
}

DEFAULT_COLOR = "#999999"
DEFAULT_PATTERN = ""

# === FUNCTIONS ===
def load_data(csv_paths):
    """Load and merge multiple CSV files into one DataFrame."""
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    return df


def plot_runtime_breakdown(df, savepath=None):
    """
    Plot runtime breakdown:
    - x-axis = nodes
    - grouped bars = programs
    - stacked segments = timers (excluding global_timer)
    """
    # keep only program phases, exclude global_timer
    df_parts = df[df["timer"] != "global_timer"]

    # aggregate over runs and ranks
    agg = (
        df_parts.groupby(["nodes", "program", "timer"], as_index=False)["avg"]
        .mean()
    )

    # pivot to wide for stacking
    pivot = agg.pivot_table(
        index=["nodes", "program"],
        columns="timer",
        values="avg",
        fill_value=0
    )

    timers = list(pivot.columns)
    programs = pivot.index.get_level_values("program").unique()
    nodes = pivot.index.get_level_values("nodes").unique()

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.8 / len(programs)

    # draw stacked bars
    for i, program in enumerate(programs):
        color = PROGRAM_COLORS.get(program, DEFAULT_COLOR)
        for j, node in enumerate(nodes):
            idx = (node, program)
            if idx not in pivot.index:
                continue
            vals = pivot.loc[idx, timers]
            bottom = 0
            for timer in timers:
                hatch = TIMER_PATTERNS.get(timer, DEFAULT_PATTERN)
                ax.bar(
                    j + i * bar_width - 0.4 + bar_width / 2,
                    vals[timer],
                    bar_width,
                    bottom=bottom,
                    color=color,
                    hatch=hatch,
                    edgecolor="black",
                )
                bottom += vals[timer]

    # axis labels
    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels(nodes)
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Runtime")
    ax.set_title("Runtime Breakdown by Program and Timer Phase")

    # === Legends ===
    program_patches = [
        mpatches.Patch(facecolor=PROGRAM_COLORS.get(p, DEFAULT_COLOR), label=p)
        for p in programs
    ]
    timer_patches = [
        mpatches.Patch(facecolor="white", edgecolor="black",
                       hatch=TIMER_PATTERNS.get(t, DEFAULT_PATTERN), label=t)
        for t in timers
    ]
    legend1 = ax.legend(handles=program_patches, title="Programs",
                        bbox_to_anchor=(1.05, 1), loc="upper left")
    legend2 = ax.legend(handles=timer_patches, title="Timers",
                        bbox_to_anchor=(1.05, 0.5), loc="upper left")
    ax.add_artist(legend1)

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark CSV data.")
    parser.add_argument("csv_files", nargs="+", help="Paths to CSV files.")
    args = parser.parse_args()

    df = load_data(args.csv_files)
    plot_runtime_breakdown(df, 'results/spgemm_comparison.png')


if __name__ == "__main__":
    main()
