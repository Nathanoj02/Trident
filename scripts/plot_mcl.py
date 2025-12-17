#!/usr/bin/env python3
"""Plotting script for HnS-SpGEMM MCL benchmark results."""

import argparse
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Default results directory (relative to script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results_mcl')

# Implementation types
IMPL_TRIDENT = 'Trident'
IMPL_SPARSE_SUMMA = 'Sparse SUMMA'
IMPL_TRILINOS = 'Trilinos'

# Colors for implementations
IMPL_COLORS = {
    IMPL_TRILINOS: '#7A9DCF',        # soft blue
    IMPL_TRIDENT: '#F4A582',         # soft coral
    IMPL_SPARSE_SUMMA: '#92C5A9',    # soft sage
}

# Marker shapes for implementations
IMPL_MARKERS = {
    IMPL_TRILINOS: 'o',              # circle
    IMPL_TRIDENT: 's',               # square
    IMPL_SPARSE_SUMMA: '^',          # triangle up
}

# Implementation order for consistent plotting
IMPL_ORDER = [IMPL_TRILINOS, IMPL_TRIDENT, IMPL_SPARSE_SUMMA]

# Colors for GPU counts in per-iteration plots
GPU_COLORS = {
    4: '#4477AA',
    16: '#66CCEE',
    64: '#228833',
    256: '#CCBB44',
}

GPU_MARKERS = {
    4: 'o',
    16: 's',
    64: '^',
    256: 'D',
}


@dataclass
class HnsMclResult:
    """Parsed results from an HnS MCL output file."""
    filepath: str
    ngpus: int
    matrix: str
    implementation: str  # One of IMPL_TRIDENT or IMPL_SPARSE_SUMMA
    tol: str
    # SpGEMM times per iteration (starting from ITERATION 0)
    iteration_times_ms: list[float] = field(default_factory=list)


@dataclass
class TrilinosMclResult:
    """Parsed results from a Trilinos MCL output file."""
    filepath: str
    ngpus: int
    matrix: str
    tol: str
    # SpGEMM times per iteration (max across ranks, starting from ITERATION 0)
    iteration_times_ms: list[float] = field(default_factory=list)


def parse_hns_mcl_filename(filename: str) -> Optional[dict]:
    """Parse HnS MCL filename to extract metadata.

    Format: hns_mcl_<gpus>_<matrix>_<impl>_<tol>.out

    Implementation mapping:
    - 'async' in filename -> Trident
    - 'summa' in filename -> Sparse SUMMA
    """
    basename = os.path.basename(filename)
    pattern = r'hns_mcl_(\d+)_(.+)_(async|summa)_([0-9e.\-]+)\.out'
    match = re.match(pattern, basename)
    if not match:
        return None

    ngpus, matrix, impl_type, tol = match.groups()
    implementation = IMPL_TRIDENT if impl_type == 'async' else IMPL_SPARSE_SUMMA

    return {
        'ngpus': int(ngpus),
        'matrix': matrix,
        'implementation': implementation,
        'tol': tol,
    }


def parse_trilinos_mcl_filename(filename: str) -> Optional[dict]:
    """Parse Trilinos MCL filename to extract metadata.

    Format: trilinos_mcl_<gpus>_<matrix>_<tol>.out
    """
    basename = os.path.basename(filename)
    pattern = r'trilinos_mcl_(\d+)_(.+)_([0-9e.\-]+)\.out'
    match = re.match(pattern, basename)
    if not match:
        return None

    ngpus, matrix, tol = match.groups()

    return {
        'ngpus': int(ngpus),
        'matrix': matrix,
        'tol': tol,
    }


def parse_hns_mcl_file(filepath: str) -> Optional[HnsMclResult]:
    """Parse an HnS MCL output file to extract timing data."""
    meta = parse_hns_mcl_filename(filepath)
    if not meta:
        return None

    result = HnsMclResult(
        filepath=filepath,
        ngpus=meta['ngpus'],
        matrix=meta['matrix'],
        implementation=meta['implementation'],
        tol=meta['tol'],
    )

    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return None

    # Find ITERATION 0 marker and only parse after it
    iteration_start = content.find('ITERATION 0')
    if iteration_start == -1:
        print(f"Warning: No ITERATION 0 found in {filepath}")
        return None

    content_after_iter0 = content[iteration_start:]

    # Extract <Timer>[spgemm] times - one per iteration
    # Format: <Timer>[spgemm] 1226.827637 ms
    spgemm_pattern = r'<Timer>\[spgemm\]\s+([\d.]+)\s*ms'
    matches = re.findall(spgemm_pattern, content_after_iter0)

    if not matches:
        print(f"Warning: No spgemm timers found in {filepath}")
        return None

    result.iteration_times_ms = [float(m) for m in matches]
    return result


def parse_trilinos_mcl_file(filepath: str) -> Optional[TrilinosMclResult]:
    """Parse a Trilinos MCL output file to extract timing data."""
    meta = parse_trilinos_mcl_filename(filepath)
    if not meta:
        return None

    result = TrilinosMclResult(
        filepath=filepath,
        ngpus=meta['ngpus'],
        matrix=meta['matrix'],
        tol=meta['tol'],
    )

    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return None

    # Find ITERATION 0 marker and only parse after it
    iteration_start = content.find('ITERATION 0')
    if iteration_start == -1:
        print(f"Warning: No ITERATION 0 found in {filepath}")
        return None

    content_after_iter0 = content[iteration_start:]

    # Split content by ITERATION headers to process each iteration separately
    iteration_sections = re.split(r'===================== ITERATION \d+ ======================', content_after_iter0)

    # First element is empty (before ITERATION 0), skip it
    iteration_sections = iteration_sections[1:]

    # Extract max time across ranks for each iteration
    # Format: <Timer rank X>[mcl_spgemm] 5127.862793 ms
    spgemm_pattern = r'<Timer rank \d+>\[mcl_spgemm\]\s+([\d.]+)\s*ms'

    for section in iteration_sections:
        matches = re.findall(spgemm_pattern, section)
        if matches:
            times = [float(m) for m in matches]
            # Take max across ranks
            result.iteration_times_ms.append(max(times))

    if not result.iteration_times_ms:
        print(f"Warning: No mcl_spgemm timers found in {filepath}")
        return None

    return result


def load_all_results(results_dir: str) -> tuple[list[HnsMclResult], list[TrilinosMclResult]]:
    """Load all MCL results from the specified directory."""
    hns_results = []
    trilinos_results = []

    # Load HnS results
    hns_pattern = os.path.join(results_dir, 'hns_mcl_*.out')
    for filepath in glob(hns_pattern):
        result = parse_hns_mcl_file(filepath)
        if result and result.iteration_times_ms:
            hns_results.append(result)

    # Load Trilinos results
    trilinos_pattern = os.path.join(results_dir, 'trilinos_mcl_*.out')
    for filepath in glob(trilinos_pattern):
        result = parse_trilinos_mcl_file(filepath)
        if result and result.iteration_times_ms:
            trilinos_results.append(result)

    print(f"Loaded {len(hns_results)} HnS results and {len(trilinos_results)} Trilinos results")
    return hns_results, trilinos_results


def plot_strong_scaling(
    hns_results: list[HnsMclResult],
    trilinos_results: list[TrilinosMclResult],
    output_dir: str,
) -> None:
    """Generate strong scaling plots.

    One plot per (matrix, tol) combination.
    One series per implementation.
    Y-axis: sum of all iteration spgemm times.
    X-axis: GPU count.
    """
    plot_dir = os.path.join(output_dir, 'strong-scaling')

    # Group results by (matrix, tol)
    grouped_hns: dict[tuple[str, str], dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    grouped_trilinos: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)

    for result in hns_results:
        key = (result.matrix, result.tol)
        total_time = sum(result.iteration_times_ms)
        grouped_hns[key][result.implementation][result.ngpus] = total_time

    for result in trilinos_results:
        key = (result.matrix, result.tol)
        total_time = sum(result.iteration_times_ms)
        grouped_trilinos[key][result.ngpus] = total_time

    # Get all unique (matrix, tol) combinations
    all_keys = set(grouped_hns.keys()) | set(grouped_trilinos.keys())

    for matrix, tol in sorted(all_keys):
        matrix_dir = os.path.join(plot_dir, matrix)
        os.makedirs(matrix_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Path effect for black border on lines
        #line_border = [pe.Stroke(linewidth=4, foreground='black'), pe.Normal()]

        # Plot Trilinos
        if (matrix, tol) in grouped_trilinos:
            data = grouped_trilinos[(matrix, tol)]
            gpus = sorted(data.keys())
            times = [data[g] for g in gpus]
            ax.plot(
                gpus, times,
                marker=IMPL_MARKERS[IMPL_TRILINOS],
                color=IMPL_COLORS[IMPL_TRILINOS],
                label=IMPL_TRILINOS,
                linewidth=2,
                markersize=8,
            )

        # Plot HnS implementations
        if (matrix, tol) in grouped_hns:
            for impl in [IMPL_TRIDENT, IMPL_SPARSE_SUMMA]:
                if impl in grouped_hns[(matrix, tol)]:
                    data = grouped_hns[(matrix, tol)][impl]
                    gpus = sorted(data.keys())
                    times = [data[g] for g in gpus]
                    ax.plot(
                        gpus, times,
                        marker=IMPL_MARKERS[impl],
                        color=IMPL_COLORS[impl],
                        label=impl,
                        linewidth=2,
                        markersize=8,
                    )

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Number of GPUs', fontsize=12)
        ax.set_ylabel('Total SpGEMM Time (ms)', fontsize=12)
        ax.set_title(f'Strong Scaling: {matrix} (tol={tol})', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Set x-ticks to actual GPU counts
        all_gpus = set()
        if (matrix, tol) in grouped_trilinos:
            all_gpus.update(grouped_trilinos[(matrix, tol)].keys())
        if (matrix, tol) in grouped_hns:
            for impl_data in grouped_hns[(matrix, tol)].values():
                all_gpus.update(impl_data.keys())
        if all_gpus:
            ax.set_xticks(sorted(all_gpus))
            ax.set_xticklabels([str(g) for g in sorted(all_gpus)])

        plt.tight_layout()
        outpath = os.path.join(matrix_dir, f'strong_scaling_tol{tol}.png')
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {outpath}")


def plot_per_iteration(
    hns_results: list[HnsMclResult],
    trilinos_results: list[TrilinosMclResult],
    output_dir: str,
) -> None:
    """Generate per-iteration timing plots.

    One plot per (matrix, tol, implementation) combination.
    One series per GPU count.
    X-axis: MCL iteration.
    Y-axis: spgemm time for that iteration.
    """
    plot_dir = os.path.join(output_dir, 'per-iteration')

    # Group HnS results by (matrix, tol, implementation)
    grouped_hns: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(dict)

    for result in hns_results:
        key = (result.matrix, result.tol, result.implementation)
        grouped_hns[key][result.ngpus] = result.iteration_times_ms

    # Group Trilinos results by (matrix, tol)
    grouped_trilinos: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(dict)

    for result in trilinos_results:
        key = (result.matrix, result.tol)
        grouped_trilinos[key][result.ngpus] = result.iteration_times_ms

    # Plot HnS results
    for (matrix, tol, impl), gpu_data in sorted(grouped_hns.items()):
        impl_slug = impl.lower().replace(' ', '_')
        impl_dir = os.path.join(plot_dir, impl_slug, matrix)
        os.makedirs(impl_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        for ngpus in sorted(gpu_data.keys()):
            times = gpu_data[ngpus]
            iterations = list(range(len(times)))
            color = GPU_COLORS.get(ngpus, f'C{ngpus % 10}')
            marker = GPU_MARKERS.get(ngpus, 'o')
            ax.plot(
                iterations, times,
                marker=marker,
                color=color,
                label=f'{ngpus} GPUs',
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel('MCL Iteration', fontsize=12)
        ax.set_ylabel('SpGEMM Time (ms)', fontsize=12)
        ax.set_title(f'Per-Iteration Timing: {impl} - {matrix} (tol={tol})', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outpath = os.path.join(impl_dir, f'per_iteration_tol{tol}.png')
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {outpath}")

    # Plot Trilinos results
    for (matrix, tol), gpu_data in sorted(grouped_trilinos.items()):
        impl_dir = os.path.join(plot_dir, 'trilinos', matrix)
        os.makedirs(impl_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        for ngpus in sorted(gpu_data.keys()):
            times = gpu_data[ngpus]
            iterations = list(range(len(times)))
            color = GPU_COLORS.get(ngpus, f'C{ngpus % 10}')
            marker = GPU_MARKERS.get(ngpus, 'o')
            ax.plot(
                iterations, times,
                marker=marker,
                color=color,
                label=f'{ngpus} GPUs',
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel('MCL Iteration', fontsize=12)
        ax.set_ylabel('SpGEMM Time (ms)', fontsize=12)
        ax.set_title(f'Per-Iteration Timing: Trilinos - {matrix} (tol={tol})', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outpath = os.path.join(impl_dir, f'per_iteration_tol{tol}.png')
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {outpath}")


def get_available_plots() -> dict:
    """Return a dictionary of available plot types and their functions."""
    return {
        'strong-scaling': plot_strong_scaling,
        'per-iteration': plot_per_iteration,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Plot MCL benchmark results for HnS-SpGEMM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'plot_types',
        nargs='*',
        help='Plot types to generate. Use "all" for all plots.',
    )
    parser.add_argument(
        '-o', '--output',
        default='plots_mcl',
        help='Output directory for plots (default: plots_mcl)',
    )
    parser.add_argument(
        '-r', '--results-dir',
        default=DEFAULT_RESULTS_DIR,
        help=f'Results directory (default: {DEFAULT_RESULTS_DIR})',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available plot types',
    )

    args = parser.parse_args()
    available_plots = get_available_plots()

    if args.list:
        print("Available plot types:")
        for name in sorted(available_plots.keys()):
            print(f"  - {name}")
        print("  - all (generates all plot types)")
        return

    if not args.plot_types:
        parser.print_help()
        return

    # Determine which plots to generate
    if 'all' in args.plot_types:
        plot_types = list(available_plots.keys())
    else:
        plot_types = []
        for pt in args.plot_types:
            if pt not in available_plots:
                print(f"Error: Unknown plot type '{pt}'")
                print(f"Available: {', '.join(sorted(available_plots.keys()))}")
                return
            plot_types.append(pt)

    # Load all results
    hns_results, trilinos_results = load_all_results(args.results_dir)

    if not hns_results and not trilinos_results:
        print(f"Error: No results found in {args.results_dir}")
        return

    # Generate requested plots
    os.makedirs(args.output, exist_ok=True)

    for plot_type in plot_types:
        print(f"\nGenerating {plot_type} plots...")
        available_plots[plot_type](hns_results, trilinos_results, args.output)

    print(f"\nAll plots saved to: {args.output}")


if __name__ == '__main__':
    main()
