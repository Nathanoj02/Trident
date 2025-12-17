#!/usr/bin/env python3
"""
Communication volume plotting script for HnS-SpGEMM.

Reads data files from results_commvol/ and generates per-process communication
volume plots comparing Trident (small grid) vs Sparse SUMMA (large grid).
"""

import argparse
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass
class CommVolResult:
    """Parsed communication volume result from a single file."""
    ngpus: int
    grid: str  # e.g., "2x2", "4x4"
    grid_size: int  # product of grid dimensions, e.g., 4 for "2x2"
    matrix: str
    # Map from process ID to total communication volume (bytes)
    process_comm_vol: dict[int, int]


def parse_grid(grid_str: str) -> int:
    """Parse grid string like '2x2' and return product (e.g., 4)."""
    parts = grid_str.split('x')
    return int(parts[0]) * int(parts[1])


def parse_filename(filename: str) -> tuple[int, str, str] | None:
    """
    Parse filename to extract ngpus, grid, and matrix.

    Expected format: comm_vol_<ngpus>_<grid>_<matrix>.out
    Example: comm_vol_16_4x4_archaea.out

    Returns (ngpus, grid, matrix) or None if parsing fails.
    """
    match = re.match(r'comm_vol_(\d+)_(\d+x\d+)_(.+)\.out$', filename)
    if not match:
        return None
    ngpus = int(match.group(1))
    grid = match.group(2)
    matrix = match.group(3)
    return ngpus, grid, matrix


def parse_file(filepath: str) -> dict[int, int]:
    """
    Parse a comm_vol output file and return per-process communication volume.

    Only processes lines with 'hier 0' and extracts the first byte value.
    Sums all byte values for the same process ID.

    Returns dict mapping process_id -> total_bytes
    """
    process_comm_vol: dict[int, int] = defaultdict(int)

    # Pattern to match data lines with hier 0
    # Example: <[p 0, t 0, m A, r 0, hier 0]>[internode_comm(...)] ... 834445320 B, ...
    line_pattern = re.compile(
        r'<\[p\s*(\d+),.*hier\s+0\]>'  # Match process ID and hier 0
        r'.*?'  # Skip to byte values
        r'(\d+)\s*B'  # First byte value
    )

    with open(filepath, 'r') as f:
        for line in f:
            if 'hier 0]>' not in line:
                continue
            match = line_pattern.search(line)
            if match:
                process_id = int(match.group(1))
                byte_value = int(match.group(2))
                process_comm_vol[process_id] += byte_value

    return dict(process_comm_vol)


def load_results(results_dir: str) -> list[CommVolResult]:
    """Load all comm_vol results from the directory."""
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Warning: Results directory '{results_dir}' does not exist")
        return results

    for filepath in results_path.glob('comm_vol_*.out'):
        parsed = parse_filename(filepath.name)
        if parsed is None:
            print(f"Warning: Could not parse filename: {filepath.name}")
            continue

        ngpus, grid, matrix = parsed
        process_comm_vol = parse_file(str(filepath))

        if not process_comm_vol:
            print(f"Warning: No data found in {filepath.name}")
            continue

        results.append(CommVolResult(
            ngpus=ngpus,
            grid=grid,
            grid_size=parse_grid(grid),
            matrix=matrix,
            process_comm_vol=process_comm_vol,
        ))

    return results


def group_results_for_plotting(
    results: list[CommVolResult]
) -> dict[tuple[int, str], tuple[CommVolResult | None, CommVolResult | None]]:
    """
    Group results by (ngpus, matrix) and identify Trident vs Sparse SUMMA.

    For each (ngpus, matrix) pair, find the two grid configurations:
    - Smaller grid = Trident
    - Larger grid = Sparse SUMMA

    Returns dict mapping (ngpus, matrix) -> (trident_result, sparse_summa_result)
    """
    # Group by (ngpus, matrix)
    groups: dict[tuple[int, str], list[CommVolResult]] = defaultdict(list)
    for result in results:
        key = (result.ngpus, result.matrix)
        groups[key].append(result)

    # For each group, identify Trident (small grid) and Sparse SUMMA (large grid)
    plot_groups: dict[tuple[int, str], tuple[CommVolResult | None, CommVolResult | None]] = {}
    for key, group_results in groups.items():
        if len(group_results) == 1:
            # Only one grid configuration - skip or include as single series
            print(f"Warning: Only one grid config for {key[1]} with {key[0]} GPUs, skipping")
            continue

        # Sort by grid size
        sorted_results = sorted(group_results, key=lambda r: r.grid_size)
        trident = sorted_results[0]  # Smallest grid
        sparse_summa = sorted_results[-1]  # Largest grid

        if len(group_results) > 2:
            print(f"Warning: More than 2 grid configs for {key[1]} with {key[0]} GPUs, "
                  f"using {trident.grid} (Trident) and {sparse_summa.grid} (Sparse SUMMA)")

        plot_groups[key] = (trident, sparse_summa)

    return plot_groups


# Visual style constants (consistent with other plotting scripts)
IMPL_COLORS = {
    'trident': '#F4A582',      # soft coral
    'sparse_summa': '#92C5A9',  # soft sage
}


def plot_comm_vol(
    results: list[CommVolResult],
    output_dir: str,
) -> None:
    """
    Generate communication volume plots.

    Creates one plot per (ngpus, matrix) with two series:
    - Trident (small grid)
    - Sparse SUMMA (large grid)
    """
    plot_groups = group_results_for_plotting(results)

    if not plot_groups:
        print("No data to plot")
        return

    for (ngpus, matrix), (trident, sparse_summa) in plot_groups.items():
        if trident is None or sparse_summa is None:
            continue

        # Create output directory
        plot_dir = Path(output_dir) / matrix
        plot_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot Trident
        trident_procs = sorted(trident.process_comm_vol.keys())
        trident_vols = [trident.process_comm_vol[p] / 1e6 for p in trident_procs]  # Convert to MB
        ax.plot(
            trident_procs, trident_vols,
            linewidth=2,
            color=IMPL_COLORS['trident'],
            label='Trident',
        )

        # Plot Sparse SUMMA
        summa_procs = sorted(sparse_summa.process_comm_vol.keys())
        summa_vols = [sparse_summa.process_comm_vol[p] / 1e6 for p in summa_procs]  # Convert to MB
        ax.plot(
            summa_procs, summa_vols,
            linewidth=2,
            color=IMPL_COLORS['sparse_summa'],
            label='Sparse SUMMA',
        )

        ax.set_xlabel('Process ID', fontsize=16)
        ax.set_ylabel('Internode Communication Volume (MB)', fontsize=16)
        ax.set_title(f'Internode Communication Volume -- {matrix} ({ngpus} GPUs)', fontsize=18)
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, alpha=0.3)

        # Set x-axis ticks to process IDs
        all_procs = sorted(set(trident_procs) | set(summa_procs))
        if len(all_procs) <= 20:
            ax.set_xticks(all_procs)

        plt.tight_layout()

        output_path = plot_dir / f'comm_vol_{ngpus}gpus.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot communication volume data from HnS-SpGEMM benchmarks'
    )
    parser.add_argument(
        '-o', '--output',
        default='plots_commvol',
        help='Output directory for plots (default: plots_commvol)',
    )
    parser.add_argument(
        '-r', '--results-dir',
        default='../results_commvol',
        help='Results directory (default: ../results_commvol)',
    )

    args = parser.parse_args()

    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} result files")

    if not results:
        print("No results to plot")
        return

    print(f"Generating plots to: {args.output}")
    plot_comm_vol(results, args.output)
    print("Done!")


if __name__ == '__main__':
    main()
