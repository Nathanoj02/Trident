#!/usr/bin/env python3
"""Plotting script for HnS-SpGEMM results."""

import argparse
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Figure configuration for LaTeX documents (4 plots side by side in double column)
FIGURE_SIZE = (6, 4)  # Wide and short
FIGURE_DPI = 200
TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 12

# Default results directory (relative to script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results_final')

# Implementation types
IMPL_TRIDENT = 'Trident'
IMPL_SPARSE_SUMMA = 'Sparse SUMMA'

# Colors for line plots (HnS implementations + Trilinos + CombBLAS)
IMPL_COLORS = {
    'Trilinos': '#7A9DCF',           # soft blue
    IMPL_TRIDENT: '#F4A582',         # soft coral
    IMPL_SPARSE_SUMMA: '#92C5A9',    # soft sage
    'CombBLAS': '#AA7DC4',           # soft purple
}

# Marker shapes for line plots (different shape per implementation)
IMPL_MARKERS = {
    'Trilinos': 'o',                 # circle
    IMPL_TRIDENT: 's',               # square
    IMPL_SPARSE_SUMMA: '^',          # triangle up
    'CombBLAS': 'D',                 # diamond
}

# Hatches for breakdown plots (to distinguish implementations)
IMPL_HATCHES = {
    IMPL_TRIDENT: '',          # solid
    IMPL_SPARSE_SUMMA: '///',  # diagonal lines
}

# Implementation order for consistent plotting
IMPL_ORDER = [IMPL_TRIDENT, IMPL_SPARSE_SUMMA]


@dataclass
class HnsResult:
    """Parsed results from an HnS output file."""
    filepath: str
    ngpus: int
    grid: str  # e.g., "4x4"
    matrix: str
    backend: str
    implementation: str  # One of IMPL_TRIDENT or IMPL_SPARSE_SUMMA
    is_permute: bool = False  # True if this is a permute version
    # Overall spgemm runtimes (one per round, excluding round 0)
    spgemm_times_ms: list[float] = field(default_factory=list)
    # Per-phase timings: {phase_name: {process_rank: [sum values per round]}}
    phase_timings: dict[str, dict[int, list[float]]] = field(default_factory=dict)


@dataclass
class TrilinosResult:
    """Parsed results from a Trilinos output file."""
    filepath: str
    ngpus: int
    matrix: str
    is_permute: bool = False  # True if this is a permute version
    # Average spgemm runtime from summary line
    spgemm_avg_ms: Optional[float] = None


@dataclass
class CombblasResult:
    """Parsed results from a CombBLAS output file."""
    filepath: str
    ngpus: int
    matrix: str
    # SpGEMM runtimes (one per round, excluding round 0)
    spgemm_times_ms: list[float] = field(default_factory=list)


def parse_hns_filename(filename: str) -> Optional[dict]:
    """Parse HnS filename to extract metadata.

    Format: hns_strong_<ngpus>_<grid>_<matrix>_<backend>_nospcomm_<async|summa>_<permute>.out

    Implementation mapping:
    - 'async' in filename -> Trident
    - 'summa' in filename -> Sparse SUMMA
    """
    basename = os.path.basename(filename)
    # hns_strong_16_2x2_mouse_gene_kokkos_nospcomm_async_nopermute.out
    # hns_strong_16_4x4_mouse_gene_kokkos_nospcomm_summa_permute.out
    pattern = r'hns_strong_(\d+)_(\d+x\d+)_(.+)_(\w+)_nospcomm_(async|summa)_(permute|nopermute)\.out'
    match = re.match(pattern, basename)
    if not match:
        return None
    scheduling = match.group(5)
    permute_str = match.group(6)
    # Map scheduling type to implementation name
    if scheduling == 'async':
        implementation = IMPL_TRIDENT
    else:  # summa
        implementation = IMPL_SPARSE_SUMMA
    return {
        'ngpus': int(match.group(1)),
        'grid': match.group(2),
        'matrix': match.group(3),
        'backend': match.group(4),
        'implementation': implementation,
        'is_permute': permute_str == 'permute',
    }


def parse_trilinos_filename(filename: str) -> Optional[dict]:
    """Parse Trilinos filename to extract metadata.

    Format: trilinos_strong_<matrix>_<ngpus>.out
    Or:     trilinos_strong_<matrix>_<ngpus>_permute.out
    """
    basename = os.path.basename(filename)
    # Try permute version first
    pattern_permute = r'trilinos_strong_(.+)_(\d+)_permute\.out'
    match = re.match(pattern_permute, basename)
    if match:
        return {
            'matrix': match.group(1),
            'ngpus': int(match.group(2)),
            'is_permute': True,
        }
    # Try non-permute version
    pattern = r'trilinos_strong_(.+)_(\d+)\.out'
    match = re.match(pattern, basename)
    if not match:
        return None
    return {
        'matrix': match.group(1),
        'ngpus': int(match.group(2)),
        'is_permute': False,
    }


def parse_combblas_filename(filename: str) -> Optional[dict]:
    """Parse CombBLAS filename to extract metadata.

    Format: combblas_strong_<matrix>_<ngpus>.out
    """
    basename = os.path.basename(filename)
    pattern = r'combblas_strong_(.+)_(\d+)\.out'
    match = re.match(pattern, basename)
    if not match:
        return None
    return {
        'matrix': match.group(1),
        'ngpus': int(match.group(2)),
    }


def parse_hns_file(filepath: str) -> Optional[HnsResult]:
    """Parse an HnS output file and extract timing/communication data."""
    meta = parse_hns_filename(filepath)
    if not meta:
        return None

    result = HnsResult(
        filepath=filepath,
        ngpus=meta['ngpus'],
        grid=meta['grid'],
        matrix=meta['matrix'],
        backend=meta['backend'],
        implementation=meta['implementation'],
        is_permute=meta['is_permute'],
    )

    with open(filepath, 'r') as f:
        content = f.read()

    # Track current round (starts at -1, incremented when we see "STARTING spgemm round:")
    current_round = -1

    # Temporary storage for phase timings per round
    # {phase: {process: {round: sum_value}}}
    round_phase_timings: dict[str, dict[int, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    # Temporary storage for single-value phase timings (like spm_time, spadd_time)
    # These need to be summed per round per process
    # {phase: {process: {round: accumulated_value}}}
    round_single_phase_timings: dict[str, dict[int, dict[int, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )

    # Regex patterns
    round_pattern = re.compile(r'STARTING spgemm round:\s*(\d+)')
    # <Timer>[spgemm] 1357.714355 ms (not the summary line with avg=)
    spgemm_timer_pattern = re.compile(r'<Timer>\[spgemm\]\s+([\d.]+)\s+ms')
    # <[process P]>[phase_name] ...,sum=X.X (summary format)
    phase_pattern = re.compile(r'<\[process\s+(\d+)\]>\[(\w+)\].*sum=([\d.]+)')
    # <[process P]>[phase_name] X.X ms (single value format, for spm_time, spadd_time, intranode_comm, etc.)
    # Note: intranode_comm appears in single-value format when grid is 1x1
    # Lines may have ANSI color codes (e.g., [96m...[0m) that need to be handled
    single_phase_pattern = re.compile(r'<\[process\s+(\d+)\]>\[(spm_time|spadd_time|wait_for_input|intranode_comm|task_queue_checkn|task_queue_incn|task_queue_pop)\]\s*([\d.]+)\s*ms')

    for line in content.split('\n'):
        # Check for round start
        round_match = round_pattern.search(line)
        if round_match:
            current_round = int(round_match.group(1))
            continue

        # Check for spgemm timer (skip round 0)
        spgemm_match = spgemm_timer_pattern.search(line)
        if spgemm_match and 'avg=' not in line:
            if current_round > 0:  # Skip round 0
                result.spgemm_times_ms.append(float(spgemm_match.group(1)))
            continue

        # Check for single-value phase timing (spm_time, spadd_time) - skip round 0
        single_phase_match = single_phase_pattern.search(line)
        if single_phase_match and current_round > 0:
            process = int(single_phase_match.group(1))
            phase_name = single_phase_match.group(2)
            time_val = float(single_phase_match.group(3))
            round_single_phase_timings[phase_name][process][current_round] += time_val
            continue

        # Check for summary phase timing (skip round 0)
        phase_match = phase_pattern.search(line)
        if phase_match and current_round > 0:
            process = int(phase_match.group(1))
            phase_name = phase_match.group(2)
            sum_val = float(phase_match.group(3))
            round_phase_timings[phase_name][process][current_round] = sum_val
            continue

    # Convert round-based data to list-based (sorted by round)
    # Phase timings (summary format)
    for phase_name, process_data in round_phase_timings.items():
        result.phase_timings[phase_name] = {}
        for process, round_data in process_data.items():
            rounds = sorted(round_data.keys())
            result.phase_timings[phase_name][process] = [round_data[r] for r in rounds]

    # Single-value phase timings (spm_time, spadd_time)
    for phase_name, process_data in round_single_phase_timings.items():
        result.phase_timings[phase_name] = {}
        for process, round_data in process_data.items():
            rounds = sorted(round_data.keys())
            result.phase_timings[phase_name][process] = [round_data[r] for r in rounds]

    return result


def parse_trilinos_file(filepath: str) -> Optional[TrilinosResult]:
    """Parse a Trilinos output file and extract timing data."""
    meta = parse_trilinos_filename(filepath)
    if not meta:
        return None

    result = TrilinosResult(
        filepath=filepath,
        ngpus=meta['ngpus'],
        matrix=meta['matrix'],
        is_permute=meta['is_permute'],
    )

    with open(filepath, 'r') as f:
        content = f.read()

    # Look for summary line: <Timer>[spgemm] n=10,avg=1902.223877,...
    pattern = re.compile(r'<Timer>\[spgemm\].*avg=([\d.]+)')
    match = pattern.search(content)
    if match:
        result.spgemm_avg_ms = float(match.group(1))

    return result


def parse_combblas_file(filepath: str) -> Optional[CombblasResult]:
    """Parse a CombBLAS output file and extract timing data.

    CombBLAS outputs multiple <Timer>[spgemm] X.X ms lines.
    The first is warmup (round 0), the rest are actual runs.
    Lines may have ANSI color codes that need to be stripped.
    """
    meta = parse_combblas_filename(filepath)
    if not meta:
        return None

    result = CombblasResult(
        filepath=filepath,
        ngpus=meta['ngpus'],
        matrix=meta['matrix'],
    )

    with open(filepath, 'r') as f:
        content = f.read()

    # Match <Timer>[spgemm] X.X ms lines, ignoring ANSI color codes
    # Format: [96m<Timer>[spgemm] 20165.000000 ms[0m
    pattern = re.compile(r'<Timer>\[spgemm\]\s+([\d.]+)\s+ms')
    matches = pattern.findall(content)

    # Skip round 0 (warmup), take the rest
    if len(matches) > 1:
        result.spgemm_times_ms = [float(m) for m in matches[1:]]
    elif len(matches) == 1:
        # Only one measurement, use it
        result.spgemm_times_ms = [float(matches[0])]

    return result


def load_all_results(results_dir: str = DEFAULT_RESULTS_DIR) -> tuple[list[HnsResult], list[TrilinosResult], list[CombblasResult]]:
    """Load all results from the given directory."""
    hns_results = []
    trilinos_results = []
    combblas_results = []

    for filepath in glob(os.path.join(results_dir, '*.out')):
        basename = os.path.basename(filepath)

        if basename.startswith('hns_strong_'):
            result = parse_hns_file(filepath)
            if result:
                hns_results.append(result)
        elif basename.startswith('trilinos_'):
            result = parse_trilinos_file(filepath)
            if result:
                trilinos_results.append(result)
        elif basename.startswith('combblas_'):
            result = parse_combblas_file(filepath)
            if result:
                combblas_results.append(result)

    return hns_results, trilinos_results, combblas_results


def _plot_runtime_comparison_impl(
    hns_for_matrix: list[HnsResult],
    trilinos_for_matrix: list[TrilinosResult],
    combblas_for_matrix: list[CombblasResult],
    matrix: str,
    output_dir: str,
    title_suffix: str = '',
    filename_suffix: str = '',
) -> None:
    """Internal implementation for runtime comparison plots."""
    import numpy as np

    if not hns_for_matrix and not trilinos_for_matrix and not combblas_for_matrix:
        return

    # Collect all unique GPU counts
    gpu_counts = set()
    for r in hns_for_matrix:
        gpu_counts.add(r.ngpus)
    for r in trilinos_for_matrix:
        gpu_counts.add(r.ngpus)
    for r in combblas_for_matrix:
        gpu_counts.add(r.ngpus)
    gpu_counts = sorted(gpu_counts)

    # Build data structure: {implementation: {ngpus: runtime_ms}}
    hns_data: dict[str, dict[int, float]] = defaultdict(dict)
    trilinos_data: dict[int, float] = {}
    combblas_data: dict[int, float] = {}

    # Trilinos data
    for r in trilinos_for_matrix:
        if r.spgemm_avg_ms is not None:
            trilinos_data[r.ngpus] = r.spgemm_avg_ms

    # CombBLAS data
    for r in combblas_for_matrix:
        if r.spgemm_times_ms:
            avg_time = sum(r.spgemm_times_ms) / len(r.spgemm_times_ms)
            combblas_data[r.ngpus] = avg_time

    # HnS data by implementation
    for r in hns_for_matrix:
        if r.spgemm_times_ms:
            avg_time = sum(r.spgemm_times_ms) / len(r.spgemm_times_ms)
            hns_data[r.implementation][r.ngpus] = avg_time

    # Create the plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Path effect for black outline on lines
    #line_outline = [pe.Stroke(linewidth=4, foreground='black'), pe.Normal()]

    # Plot Trilinos line
    if trilinos_data:
        x_vals = sorted(trilinos_data.keys())
        y_vals = [trilinos_data[g] for g in x_vals]
        ax.plot(x_vals, y_vals, marker=IMPL_MARKERS['Trilinos'], linestyle='-',
                label='Trilinos', color=IMPL_COLORS['Trilinos'],
                linewidth=2, markersize=6, markeredgecolor='black',
                markeredgewidth=0)

    # Plot CombBLAS line
    if combblas_data:
        x_vals = sorted(combblas_data.keys())
        y_vals = [combblas_data[g] for g in x_vals]
        ax.plot(x_vals, y_vals, marker=IMPL_MARKERS['CombBLAS'], linestyle='-',
                label='CombBLAS', color=IMPL_COLORS['CombBLAS'],
                linewidth=2, markersize=6, markeredgecolor='black',
                markeredgewidth=0)

    # Plot HnS lines for each implementation
    for impl in IMPL_ORDER:
        if impl in hns_data:
            gpu_data = hns_data[impl]
            x_vals = sorted(gpu_data.keys())
            y_vals = [gpu_data[g] for g in x_vals]
            ax.plot(x_vals, y_vals, marker=IMPL_MARKERS[impl], linestyle='-',
                    label=impl, color=IMPL_COLORS[impl],
                    linewidth=2, markersize=6, markeredgecolor='black',
                    markeredgewidth=0)

    ax.set_xlabel('Number of GPUs', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Runtime (ms)', fontsize=AXIS_LABEL_FONTSIZE)
    title = f'SpGEMM Runtime: {matrix}'
    if title_suffix:
        title += f' {title_suffix}'
    ax.set_title(title, fontsize=TITLE_FONTSIZE)

    # Set log scales
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)

    # Set x-axis ticks to the actual GPU counts
    ax.set_xticks(gpu_counts)
    ax.set_xticklabels([str(g) for g in gpu_counts], fontsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)

    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.grid(True, linestyle='--', alpha=0.7, which='both')

    # Save the plot
    plot_subdir = os.path.join(output_dir, 'runtime-comparison', matrix)
    os.makedirs(plot_subdir, exist_ok=True)
    filename = f'runtime_comparison{filename_suffix}.png'
    output_path = os.path.join(plot_subdir, filename)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"    Saved {output_path}")


def plot_runtime_comparison(
    hns_results: list[HnsResult],
    trilinos_results: list[TrilinosResult],
    combblas_results: list[CombblasResult],
    output_dir: str,
) -> None:
    """Line plot comparing Trident, Sparse SUMMA, Trilinos, and CombBLAS spgemm runtimes per matrix.

    Generates two types of plots:
    - Regular comparison (non-permute Trilinos) for all matrices
    - Permute comparison (permute Trilinos) only for HV15R
    """
    # Group results by matrix
    matrices = set()
    for r in hns_results:
        matrices.add(r.matrix)
    for r in trilinos_results:
        matrices.add(r.matrix)
    for r in combblas_results:
        matrices.add(r.matrix)

    for matrix in sorted(matrices):
        # Filter results for this matrix
        combblas_for_matrix = [r for r in combblas_results if r.matrix == matrix]

        # Regular comparison (non-permute) - for all matrices
        hns_nopermute = [r for r in hns_results
                         if r.matrix == matrix and not r.is_permute]
        trilinos_nopermute = [r for r in trilinos_results
                              if r.matrix == matrix and not r.is_permute]
        _plot_runtime_comparison_impl(
            hns_nopermute, trilinos_nopermute, combblas_for_matrix, matrix, output_dir
        )

        # Permute comparison - only for HV15R
        if matrix == 'HV15R':
            hns_permute = [r for r in hns_results
                           if r.matrix == matrix and r.is_permute]
            trilinos_permute = [r for r in trilinos_results
                                if r.matrix == matrix and r.is_permute]
            _plot_runtime_comparison_impl(
                hns_permute, trilinos_permute, combblas_for_matrix, matrix, output_dir,
                title_suffix='(Permute)',
                filename_suffix='_permute'
            )


def _get_comm_time_for_result(result: HnsResult) -> Optional[float]:
    """Calculate communication time for an HnS result.

    For Trident: comm_time = wait_for_input (internode) + intranode_comm
    For Sparse SUMMA: comm_time = bcast (broadcast)

    Returns max across processes of (average across rounds), or None if no data.
    """
    if result.implementation == IMPL_TRIDENT:
        # Sum of internode and intranode communication
        phases = ['wait_for_input', 'intranode_comm']
    elif result.implementation == IMPL_SPARSE_SUMMA:
        # Just broadcast time
        phases = ['bcast']
    else:
        return None

    # For each process, compute total comm time (sum of relevant phases, averaged across rounds)
    process_comm_times: dict[int, float] = defaultdict(float)

    for phase in phases:
        if phase not in result.phase_timings:
            continue
        for process, round_values in result.phase_timings[phase].items():
            if round_values:
                avg_val = sum(round_values) / len(round_values)
                process_comm_times[process] += avg_val

    if not process_comm_times:
        return None

    # Return max across processes
    return max(process_comm_times.values())


def plot_comm_time_comparison(
    hns_results: list[HnsResult],
    trilinos_results: list[TrilinosResult],
    combblas_results: list[CombblasResult],
    output_dir: str,
) -> None:
    """Line plot comparing communication time between Trident and Sparse SUMMA.

    For Trident: communication time = internode (wait_for_input) + intranode_comm
    For Sparse SUMMA: communication time = broadcast (bcast)

    One plot per matrix. X-axis: GPU count, Y-axis: communication time (ms).
    """
    import numpy as np

    # Group HnS results by matrix
    matrices = set(r.matrix for r in hns_results)

    for matrix in sorted(matrices):
        # Filter results for this matrix (non-permute only for consistency)
        hns_for_matrix = [r for r in hns_results
                         if r.matrix == matrix and not r.is_permute]

        if not hns_for_matrix:
            continue

        # Collect all unique GPU counts
        gpu_counts = sorted(set(r.ngpus for r in hns_for_matrix))

        # Build data structure: {implementation: {ngpus: comm_time_ms}}
        comm_data: dict[str, dict[int, float]] = defaultdict(dict)

        for r in hns_for_matrix:
            comm_time = _get_comm_time_for_result(r)
            if comm_time is not None:
                comm_data[r.implementation][r.ngpus] = comm_time

        if not comm_data:
            continue

        # Create the plot
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        # Plot lines for each implementation
        for impl in IMPL_ORDER:
            if impl not in comm_data:
                continue
            gpu_data = comm_data[impl]
            x_vals = sorted(gpu_data.keys())
            y_vals = [gpu_data[g] for g in x_vals]
            ax.plot(x_vals, y_vals, marker=IMPL_MARKERS[impl], linestyle='-',
                    label=impl, color=IMPL_COLORS[impl],
                    linewidth=2, markersize=6, markeredgecolor='black',
                    markeredgewidth=0)

        ax.set_xlabel('Number of GPUs', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel('Communication Time (ms)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(f'Communication Time: {matrix}', fontsize=TITLE_FONTSIZE)

        # Set log scales
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=10)

        # Set x-axis ticks to the actual GPU counts
        ax.set_xticks(gpu_counts)
        ax.set_xticklabels([str(g) for g in gpu_counts], fontsize=TICK_LABEL_FONTSIZE)
        ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)

        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.grid(True, linestyle='--', alpha=0.7, which='both')

        # Save the plot
        plot_subdir = os.path.join(output_dir, 'comm-time-comparison', matrix)
        os.makedirs(plot_subdir, exist_ok=True)
        output_path = os.path.join(plot_subdir, 'comm_time_comparison.png')
        fig.tight_layout()
        fig.savefig(output_path, dpi=FIGURE_DPI)
        plt.close(fig)
        print(f"    Saved {output_path}")


PHASE_DISPLAY_NAMES = {
    'wait_for_input': 'Internode Communication',
    'intranode_comm': 'Intranode Communication',
    'spm_time': 'Local SpGEMM',
    'spadd_time': 'Accumulation',
    'A_conversion': 'CSC Conversion',
    'bcast': 'Broadcast',
    'task_queue_checkn': 'Queue Check',
    'task_queue_incn': 'Queue Increment',
    'task_queue_pop': 'Queue Pop',
}

# Custom colors for phases (visually distinct, colorblind-friendly palette)
PHASE_COLORS = {
    'wait_for_input': '#4477AA',    # blue - internode comm
    'intranode_comm': '#66CCEE',    # cyan - intranode comm
    'spm_time': '#228833',          # green - local SpGEMM
    'spadd_time': '#CCBB44',        # yellow - accumulation
    'A_conversion': '#EE6677',      # red - CSC conversion
    'bcast': '#882255',             # wine - broadcast (Sparse SUMMA only)
    'task_queue_checkn': '#AA3377', # purple - queue check
    'task_queue_incn': '#BBBBBB',   # gray - queue increment
    'task_queue_pop': '#44AA99',    # teal - queue pop
}

# Phases that are queue-related (for filtering)
QUEUE_PHASES = {'task_queue_checkn', 'task_queue_incn', 'task_queue_pop'}

# Phases to exclude from Trident plots
TRIDENT_EXCLUDED_PHASES = {'A_conversion'}

# Phases to exclude from Sparse SUMMA plots
SPARSE_SUMMA_EXCLUDED_PHASES = {'A_conversion', 'wait_for_input'}


def plot_runtime_breakdown_per_process(
    hns_results: list[HnsResult],
    trilinos_results: list[TrilinosResult],
    combblas_results: list[CombblasResult],
    output_dir: str,
) -> None:
    """Stacked bar plot showing per-process runtime breakdown for HnS.

    Generates separate plots for each implementation (Trident, Sparse SUMMA).
    """
    import numpy as np
    from matplotlib.patches import Patch

    # Group results by (matrix, ngpus, implementation)
    groups: dict[tuple[str, int, str], HnsResult] = {}
    for r in hns_results:
        key = (r.matrix, r.ngpus, r.implementation)
        groups[key] = r

    for (matrix, ngpus, impl), result in sorted(groups.items()):
        # Collect all process ranks
        all_processes = set()
        for phase_data in result.phase_timings.values():
            all_processes.update(phase_data.keys())
        all_processes = sorted(all_processes)

        if not all_processes:
            continue

        # Collect all phases (only those in PHASE_DISPLAY_NAMES, exclude queue phases)
        # Also exclude implementation-specific excluded phases
        if impl == IMPL_TRIDENT:
            excluded = QUEUE_PHASES | TRIDENT_EXCLUDED_PHASES
        elif impl == IMPL_SPARSE_SUMMA:
            excluded = QUEUE_PHASES | SPARSE_SUMMA_EXCLUDED_PHASES
        else:
            excluded = QUEUE_PHASES
        all_phases = sorted([p for p in result.phase_timings.keys()
                            if p in PHASE_DISPLAY_NAMES and p not in excluded])

        if not all_phases:
            continue

        # Build data: {process: {phase: avg_value}}
        data: dict[int, dict[str, float]] = defaultdict(dict)

        for phase, process_data in result.phase_timings.items():
            if phase not in all_phases:
                continue
            for process, round_values in process_data.items():
                if round_values:
                    avg_val = sum(round_values) / len(round_values)
                    data[process][phase] = avg_val

        # Create the plot
        n_processes = len(all_processes)

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        bar_width = 0.8
        x = np.arange(n_processes)
        bottom = np.zeros(n_processes)

        # Plot stacked bars
        for phase in all_phases:
            values = []
            for proc in all_processes:
                val = data.get(proc, {}).get(phase, 0.0)
                values.append(val)
            values = np.array(values)

            display_name = PHASE_DISPLAY_NAMES.get(phase, phase)
            color = PHASE_COLORS.get(phase, '#888888')
            ax.bar(x, values, bar_width, bottom=bottom,
                   label=display_name, color=color,
                   edgecolor='black', linewidth=0.5)
            bottom += values

        ax.set_xlabel('Process Rank', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel('Runtime (ms)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(f'{matrix} ({ngpus} GPUs) - {impl}', fontsize=TITLE_FONTSIZE)
        ax.set_xticks(x[::max(1, len(x)//8)])  # Show subset of ticks if many processes
        ax.set_xticklabels([all_processes[i] for i in range(0, len(all_processes), max(1, len(all_processes)//8))],
                          fontsize=TICK_LABEL_FONTSIZE)
        ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)

        # Create legend for phases
        phase_handles = [Patch(facecolor=PHASE_COLORS.get(phase, '#888888'), edgecolor='black',
                               label=PHASE_DISPLAY_NAMES.get(phase, phase))
                        for phase in all_phases]
        ax.legend(handles=phase_handles, title='Phase', loc='upper right',
                  fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE)

        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Save to implementation-specific subdirectory
        impl_dir_name = impl.lower().replace(' ', '_')
        plot_subdir = os.path.join(output_dir, 'runtime-breakdown-per-process', impl_dir_name, matrix)
        os.makedirs(plot_subdir, exist_ok=True)
        filename = f'{ngpus}gpus.png'
        output_path = os.path.join(plot_subdir, filename)
        fig.tight_layout()
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved {output_path}")


def plot_runtime_breakdown_overall(
    hns_results: list[HnsResult],
    trilinos_results: list[TrilinosResult],
    combblas_results: list[CombblasResult],
    output_dir: str,
) -> None:
    """Stacked bar plot showing overall runtime breakdown for HnS.

    Shows both Trident and Sparse SUMMA side-by-side in the same plot,
    with different hatches to distinguish implementations.
    """
    import numpy as np
    from matplotlib.patches import Patch

    # Group results by matrix
    matrix_results: dict[str, list[HnsResult]] = defaultdict(list)
    for r in hns_results:
        matrix_results[r.matrix].append(r)

    for matrix, results in sorted(matrix_results.items()):
        if not results:
            continue

        # Collect all unique GPU counts
        gpu_counts = sorted(set(r.ngpus for r in results))

        # Determine which implementations are present
        impls_present = sorted(set(r.implementation for r in results),
                               key=lambda x: IMPL_ORDER.index(x) if x in IMPL_ORDER else 999)

        if not impls_present:
            continue

        # Collect all phases across both implementations
        # Use union of phases from both implementations (excluding implementation-specific exclusions)
        all_phases_set = set()
        for r in results:
            if r.implementation == IMPL_TRIDENT:
                excluded = QUEUE_PHASES | TRIDENT_EXCLUDED_PHASES
            elif r.implementation == IMPL_SPARSE_SUMMA:
                excluded = QUEUE_PHASES | SPARSE_SUMMA_EXCLUDED_PHASES
            else:
                excluded = QUEUE_PHASES
            for phase in r.phase_timings.keys():
                if phase in PHASE_DISPLAY_NAMES and phase not in excluded:
                    all_phases_set.add(phase)
        all_phases = sorted(all_phases_set)

        if not all_phases:
            continue

        # Build data: {impl: {ngpus: {phase: avg_value}}}
        data: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

        for r in results:
            impl = r.implementation
            # Determine excluded phases for this implementation
            if impl == IMPL_TRIDENT:
                excluded = QUEUE_PHASES | TRIDENT_EXCLUDED_PHASES
            elif impl == IMPL_SPARSE_SUMMA:
                excluded = QUEUE_PHASES | SPARSE_SUMMA_EXCLUDED_PHASES
            else:
                excluded = QUEUE_PHASES

            for phase in all_phases:
                if phase in excluded:
                    continue
                if phase in r.phase_timings:
                    # For each process, compute average across rounds
                    process_avgs = []
                    for process, round_values in r.phase_timings[phase].items():
                        if round_values:
                            avg_val = sum(round_values) / len(round_values)
                            process_avgs.append(avg_val)
                    # Take average across processes
                    if process_avgs:
                        data[impl][r.ngpus][phase] = sum(process_avgs) / len(process_avgs)

        # Create the plot
        n_gpu_counts = len(gpu_counts)
        n_impls = len(impls_present)

        fig, ax = plt.subplots(figsize=(max(8, n_gpu_counts * 1.5), 5))

        bar_width = 0.35
        x = np.arange(n_gpu_counts)

        # Plot grouped stacked bars for each implementation
        for impl_idx, impl in enumerate(impls_present):
            offset = (impl_idx - (n_impls - 1) / 2) * bar_width
            bottom = np.zeros(n_gpu_counts)
            hatch = IMPL_HATCHES.get(impl, '')

            for phase in all_phases:
                values = []
                for ngpus in gpu_counts:
                    val = data.get(impl, {}).get(ngpus, {}).get(phase, 0.0)
                    values.append(val)
                values = np.array(values)

                color = PHASE_COLORS.get(phase, '#888888')
                ax.bar(x + offset, values, bar_width, bottom=bottom,
                       color=color, edgecolor='black', linewidth=0.5,
                       hatch=hatch)
                bottom += values

        ax.set_xlabel('Number of GPUs', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel('Runtime (ms)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(f'Runtime Breakdown: {matrix}', fontsize=TITLE_FONTSIZE)
        ax.set_xticks(x)
        ax.set_xticklabels(gpu_counts, fontsize=TICK_LABEL_FONTSIZE)
        ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)

        # Create legend with two sections: phases and implementations
        legend_handles = []

        # Phase legend entries
        for phase in all_phases:
            legend_handles.append(Patch(
                facecolor=PHASE_COLORS.get(phase, '#888888'),
                edgecolor='black',
                label=PHASE_DISPLAY_NAMES.get(phase, phase)
            ))

        # Add separator (empty entry)
        legend_handles.append(Patch(facecolor='none', edgecolor='none', label=''))

        # Implementation legend entries (showing hatch patterns)
        for impl in impls_present:
            legend_handles.append(Patch(
                facecolor='white',
                edgecolor='black',
                hatch=IMPL_HATCHES.get(impl, ''),
                label=impl
            ))

        ax.legend(handles=legend_handles, loc='upper right',
                  fontsize=LEGEND_FONTSIZE)

        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Save plot
        plot_subdir = os.path.join(output_dir, 'runtime-breakdown-overall', matrix)
        os.makedirs(plot_subdir, exist_ok=True)
        output_path = os.path.join(plot_subdir, 'runtime_breakdown_overall.png')
        fig.tight_layout()
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved {output_path}")


def plot_runtime_per_process_stack(
    hns_results: list[HnsResult],
    trilinos_results: list[TrilinosResult],
    combblas_results: list[CombblasResult],
    output_dir: str,
) -> None:
    """Stackplot showing per-phase runtime breakdown per process for each HnS implementation.

    Generates separate plots for each implementation (Trident, Sparse SUMMA).
    """
    import numpy as np
    from matplotlib.patches import Patch

    # Group results by (matrix, ngpus, implementation)
    groups: dict[tuple[str, int, str], HnsResult] = {}
    for r in hns_results:
        key = (r.matrix, r.ngpus, r.implementation)
        groups[key] = r

    for (matrix, ngpus, impl), result in sorted(groups.items()):
        # Collect all process ranks
        all_processes = set()
        for phase_data in result.phase_timings.values():
            all_processes.update(phase_data.keys())
        all_processes = sorted(all_processes)

        if not all_processes:
            continue

        # Collect all phases (only those in PHASE_DISPLAY_NAMES, exclude queue phases)
        # Also exclude implementation-specific excluded phases
        if impl == IMPL_TRIDENT:
            excluded = QUEUE_PHASES | TRIDENT_EXCLUDED_PHASES
        elif impl == IMPL_SPARSE_SUMMA:
            excluded = QUEUE_PHASES | SPARSE_SUMMA_EXCLUDED_PHASES
        else:
            excluded = QUEUE_PHASES
        all_phases = sorted([p for p in result.phase_timings.keys()
                            if p in PHASE_DISPLAY_NAMES and p not in excluded])

        if not all_phases:
            continue

        # Build data: {phase: {process: avg_value}}
        data: dict[str, dict[int, float]] = defaultdict(dict)

        for phase, process_data in result.phase_timings.items():
            if phase not in all_phases:
                continue
            for process, round_values in process_data.items():
                if round_values:
                    avg_val = sum(round_values) / len(round_values)
                    data[phase][process] = avg_val

        # Create the plot
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        x = np.array(all_processes)

        # Build stack data: list of arrays, one per phase
        stack_data = []
        colors = []
        for phase in all_phases:
            values = [data.get(phase, {}).get(proc, 0.0) for proc in all_processes]
            stack_data.append(values)
            colors.append(PHASE_COLORS.get(phase, '#888888'))

        if stack_data:
            ax.stackplot(x, stack_data, colors=colors, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Process Rank', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel('Runtime (ms)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(f'{matrix} ({ngpus} GPUs) - {impl}', fontsize=TITLE_FONTSIZE)
        ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Create legend for phases
        phase_handles = [Patch(facecolor=PHASE_COLORS.get(phase, '#888888'), edgecolor='black',
                               label=PHASE_DISPLAY_NAMES.get(phase, phase))
                        for phase in all_phases]
        ax.legend(handles=phase_handles, title='Phase', loc='upper right',
                  fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE)

        # Save to implementation-specific subdirectory
        impl_dir_name = impl.lower().replace(' ', '_')
        plot_subdir = os.path.join(output_dir, 'runtime-per-process-stack', impl_dir_name, matrix)
        os.makedirs(plot_subdir, exist_ok=True)
        filename = f'{ngpus}gpus.png'
        output_path = os.path.join(plot_subdir, filename)
        fig.tight_layout()
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved {output_path}")


def get_available_plots():
    """Return a dictionary of available plot types and their functions."""
    return {
        'runtime-comparison': plot_runtime_comparison,
        'comm-time-comparison': plot_comm_time_comparison,
        'runtime-breakdown-per-process': plot_runtime_breakdown_per_process,
        'runtime-breakdown-overall': plot_runtime_breakdown_overall,
        'runtime-per-process-stack': plot_runtime_per_process_stack,
    }


def print_geomean_speedups(
    hns_results: list[HnsResult],
    trilinos_results: list[TrilinosResult],
    combblas_results: list[CombblasResult],
) -> None:
    """Print geometric mean speedup of Trident over Trilinos, CombBLAS, and Sparse SUMMA at 256 GPUs."""
    import math

    target_gpus = 256
    target_matrices = ['archaea', 'eukarya', 'isolates_subgraph4', 'mouse_gene']

    # Build lookup tables for runtimes at 256 GPUs
    # Trident times: {matrix: avg_ms}
    trident_times: dict[str, float] = {}
    for r in hns_results:
        if r.ngpus == target_gpus and r.implementation == IMPL_TRIDENT and r.matrix in target_matrices:
            if r.spgemm_times_ms:
                trident_times[r.matrix] = sum(r.spgemm_times_ms) / len(r.spgemm_times_ms)

    # Sparse SUMMA times: {matrix: avg_ms}
    summa_times: dict[str, float] = {}
    for r in hns_results:
        if r.ngpus == target_gpus and r.implementation == IMPL_SPARSE_SUMMA and r.matrix in target_matrices:
            if r.spgemm_times_ms:
                summa_times[r.matrix] = sum(r.spgemm_times_ms) / len(r.spgemm_times_ms)

    # Trilinos times: {matrix: avg_ms}
    trilinos_times: dict[str, float] = {}
    for r in trilinos_results:
        if r.ngpus == target_gpus and r.matrix in target_matrices and not r.is_permute:
            if r.spgemm_avg_ms is not None:
                trilinos_times[r.matrix] = r.spgemm_avg_ms

    # CombBLAS times: {matrix: avg_ms}
    combblas_times: dict[str, float] = {}
    for r in combblas_results:
        if r.ngpus == target_gpus and r.matrix in target_matrices:
            if r.spgemm_times_ms:
                combblas_times[r.matrix] = sum(r.spgemm_times_ms) / len(r.spgemm_times_ms)

    print(f"\n{'='*70}")
    print(f"Trident Speedups at {target_gpus} GPUs")
    print(f"{'='*70}")

    # Calculate speedups for each baseline
    baselines = [
        ('Trilinos', trilinos_times),
        ('CombBLAS', combblas_times),
        ('Sparse SUMMA', summa_times),
    ]

    for baseline_name, baseline_times in baselines:
        speedups = []
        print(f"\nTrident vs {baseline_name}:")
        for matrix in target_matrices:
            if matrix in trident_times and matrix in baseline_times:
                speedup = baseline_times[matrix] / trident_times[matrix]
                speedups.append(speedup)
                print(f"  {matrix}: {speedup:.2f}x ({baseline_times[matrix]:.1f} ms / {trident_times[matrix]:.1f} ms)")
            else:
                missing = []
                if matrix not in trident_times:
                    missing.append("Trident")
                if matrix not in baseline_times:
                    missing.append(baseline_name)
                print(f"  {matrix}: N/A (missing: {', '.join(missing)})")

        if speedups:
            geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
            print(f"  Geometric mean: {geomean:.2f}x")
        else:
            print(f"  Geometric mean: N/A (no data)")

    print(f"{'='*70}\n")


def main():
    available_plots = get_available_plots()

    parser = argparse.ArgumentParser(
        description='Generate plots for HnS-SpGEMM results.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'plots',
        nargs='*',
        choices=list(available_plots.keys()) + ['all'],
        help='Plot type(s) to generate. Use "all" to generate all plots.'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available plot types'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='.',
        help='Output directory for plots (default: current directory)'
    )
    parser.add_argument(
        '--results-dir', '-r',
        default=DEFAULT_RESULTS_DIR,
        help=f'Directory containing result files (default: {DEFAULT_RESULTS_DIR})'
    )

    args = parser.parse_args()

    if args.list:
        print("Available plot types:")
        for name in available_plots:
            func = available_plots[name]
            doc = func.__doc__ or "No description"
            print(f"  {name}: {doc.split(chr(10))[0]}")
        return

    if not args.plots:
        parser.print_help()
        return

    # Load all results once
    print(f"Loading results from {args.results_dir}...")
    hns_results, trilinos_results, combblas_results = load_all_results(args.results_dir)
    print(f"  Loaded {len(hns_results)} HnS, {len(trilinos_results)} Trilinos, and {len(combblas_results)} CombBLAS results.")

    # Print geometric mean speedups
    print_geomean_speedups(hns_results, trilinos_results, combblas_results)

    # Determine which plots to generate
    if 'all' in args.plots:
        plots_to_make = list(available_plots.keys())
    else:
        plots_to_make = args.plots

    # Generate each requested plot
    for plot_name in plots_to_make:
        print(f"Generating {plot_name}...")
        plot_func = available_plots[plot_name]
        plot_func(hns_results, trilinos_results, combblas_results, args.output_dir)
        print(f"  Done.")


if __name__ == '__main__':
    main()
