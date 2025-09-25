#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools
import os

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

GRID_LINESTYLE = {
    "2x2x1": "--",
    "4x4x2": ":",
}

DEFAULT_COLOR = "#999999"
DEFAULT_PATTERN = ""
DEFAULT_LINESTYLE = "-"


# === FUNCTIONS ===
def load_data(csv_paths):
    """Load and merge multiple CSV files into one DataFrame."""
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    return df


def plot_runtime_breakdown_comparison(df: pd.DataFrame, savepath='results/breakdown_comparison.png'):
    """Runtime breakdown by nodes and programs (stacked barplot), comparing 'grid' per program."""
    import numpy as np
    import os

    df_parts = df[df["timer"] != "global_timer"]

    agg = (
        df_parts.groupby(["nodes", "program", "grid", "timer"], as_index=False)["avg"]
        .mean()
    )

    pivot = agg.pivot_table(
        index=["nodes", "program", "grid"], columns="timer", values="avg", fill_value=0
    )

    timers = list(pivot.columns)

    # ordered list of nodes (x-axis)
    nodes = sorted(pivot.index.get_level_values("nodes").unique())

    # elegant way to get unique (program, grid) combos present anywhere
    # droplevel(0) removes 'nodes' so we get pairs (program, grid)
    combos = list(pivot.index.droplevel("nodes").unique())
    # combos is a list of tuples: (program, grid)
    n_combos = len(combos)
    if n_combos == 0:
        raise ValueError("No (program, grid) combinations found in the data.")

    # bar width allocated per combo within a node group; keep some spacing (0.8 total group width)
    bar_width = 0.8 / n_combos

    # base x positions for each node
    x_base = np.arange(len(nodes))

    # offsets to center the group of bars at each node tick
    offsets = (np.arange(n_combos) - (n_combos - 1) / 2.0) * bar_width

    fig, ax = plt.subplots(figsize=(10, 6))

    # draw bars: iterate nodes (x groups) and combos (bars within group)
    for j, node in enumerate(nodes):
        for k, (program, grid) in enumerate(combos):
            idx = (node, program, grid)
            if idx not in pivot.index:
                # this (program,grid) not present for this node -> skip
                continue

            vals = pivot.loc[idx, timers]
            bottom = 0.0
            x = x_base[j] + offsets[k]
            color = PROGRAM_COLORS.get(program, DEFAULT_COLOR)

            # stacked segments for each timer
            for timer in timers:
                height = float(vals[timer])
                hatch = TIMER_PATTERNS.get(timer, DEFAULT_PATTERN)
                ax.bar(
                    x,
                    height,
                    bar_width * 0.95,  # slight spacing between bars
                    bottom=bottom,
                    color=color,
                    hatch=hatch,
                    edgecolor="black",
                )
                bottom += height

            # small label above bar indicating the grid (optional)
            total = float(vals.sum())
            ax.text(x, total * 1.02, str(grid), ha="center", va="bottom", fontsize=8)

    # x ticks centered on the node groups
    ax.set_xticks(x_base)
    ax.set_xticklabels(nodes)
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Runtime [ms]")
    ax.set_title("Runtime Breakdown and Comparison")

    # program legend (unique programs only)
    programs = sorted({p for p, g in combos})
    program_patches = [
        mpatches.Patch(facecolor=PROGRAM_COLORS.get(p, DEFAULT_COLOR), label=p)
        for p in programs
    ]
    timer_patches = [
        mpatches.Patch(
            facecolor="white",
            edgecolor="black",
            hatch=TIMER_PATTERNS.get(t, DEFAULT_PATTERN),
            label=t,
        )
        for t in timers
    ]
    legend1 = ax.legend(
        handles=program_patches, title="Programs", bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    legend2 = ax.legend(
        handles=timer_patches, title="Timers", bbox_to_anchor=(1.05, 0.5), loc="upper left"
    )
    ax.add_artist(legend1)

    plt.tight_layout()
    if savepath:
        dirname = os.path.dirname(savepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(savepath)
        plt.close(fig)
    else:
        plt.show()


def plot_rank_breakdown(df: pd.DataFrame, save_prefix="results/rank_breakdown/rank_breakdown"):
    """Plot breakdown barplot with ranks on x-axis, one plot per experiment."""
    df_parts = df[(df["timer"] != "global_timer") & (df['program'].str.contains('hns'))]

    # aggregate across runs
    agg = (
        df_parts.groupby(["nodes", "program", "grid", "rank", "timer"], as_index=False)["avg"]
        .mean()
    )

    for (nodes, program, grid), subdf in agg.groupby(["nodes", "program", "grid"]):
        pivot = subdf.pivot_table(
            index=["rank"], columns="timer", values="avg", fill_value=0
        )
        timers = list(pivot.columns)

        fig, ax = plt.subplots(figsize=(10, 6))
        color = PROGRAM_COLORS.get(program, DEFAULT_COLOR)
        bar_width = 0.8

        for r, vals in pivot.iterrows():
            bottom = 0
            for timer in timers:
                hatch = TIMER_PATTERNS.get(timer, DEFAULT_PATTERN)
                ax.bar(
                    r,
                    vals[timer],
                    bar_width,
                    bottom=bottom,
                    color=color,
                    hatch=hatch,
                    edgecolor="black",
                )
                bottom += vals[timer]

        ax.set_xlabel("Rank")
        ax.set_ylabel("Runtime")
        ax.set_title(f"Runtime Breakdown\nNodes={nodes}, Program={program}, Grid={grid}")

        timer_patches = [
            mpatches.Patch(
                facecolor="white",
                edgecolor="black",
                hatch=TIMER_PATTERNS.get(t, DEFAULT_PATTERN),
                label=t,
            )
            for t in timers
        ]
        ax.legend(handles=timer_patches, title="Timers", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True)

        plt.tight_layout()
        savepath = f"{save_prefix}_{program}{grid if grid != '-' else ''}_nodes{nodes}.png"
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
        plt.close(fig)


def plot_strong_scaling(df: pd.DataFrame, savepath="results/strong_scaling.png"):
    """Strong scaling line plot: total runtime vs number of GPUs (nodes)."""
    df_parts = df[df["timer"] != "global_timer"]
    df_parts["program_and_grid"] = df_parts["program"] + ' (' + df_parts["grid"] + ')'

    agg = (
        df_parts.groupby(["nodes", "program_and_grid"], as_index=False)["avg"]
        .sum()
    )
    print(agg)

    program_and_grids = agg["program_and_grid"].unique()

    fig, ax = plt.subplots(figsize=(8, 6))
    for program_and_grid in program_and_grids:
        sub = agg[agg["program_and_grid"] == program_and_grid].sort_values("nodes")
        program, grid = program_and_grid.split(' ')
        grid = grid[1:-1]
        ax.plot(
            sub["nodes"]*4,
            sub["avg"],
            marker="o",
            linestyle=GRID_LINESTYLE.get(grid, DEFAULT_LINESTYLE),
            color=PROGRAM_COLORS.get(program, DEFAULT_COLOR),
            label=program_and_grid,
        )

    ax.set_xticks((agg["nodes"]*4).unique())
    ax.set_xlabel("GPUs")
    ax.set_ylabel("Total Runtime [ms]")
    ax.set_title("Strong Scaling")
    ax.legend(title='Program (grid)')
    ax.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark CSV data.")
    parser.add_argument("csv_files", nargs="+", help="Paths to CSV files.")
    args = parser.parse_args()

    df = load_data(args.csv_files)

    plot_runtime_breakdown_comparison(df)
    #plot_rank_breakdown(df)
    plot_strong_scaling(df)


if __name__ == "__main__":
    main()
