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

# Default results directory (relative to script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results_wave4')

# Colors for HnS configurations (dimension, spcomm)
CONFIG_COLORS = {
    ('2D', True): '#4C72B0',    # steel blue
    ('2D', False): '#55A868',   # sage green
    ('3D', True): '#C44E52',    # soft red
    ('3D', False): '#8172B3',   # muted purple
}

# Hatching patterns for workstealing vs async
WORKSTEALING_HATCHES = {
    True: '',       # workstealing: solid
    False: '///',   # async: diagonal lines
}


def grid_size(grid: str) -> int:
    """Convert grid string like '4x4' to total size (16)."""
    parts = grid.split('x')
    return int(parts[0]) * int(parts[1])


def get_dimension_label(grid: str, all_grids_for_ngpus: set[str]) -> str:
    """Determine if a grid is '2D' (larger) or '3D' (smaller) for a given GPU count.

    For a given GPU count, the larger grid is 2D, smaller is 3D.
    """
    sizes = {g: grid_size(g) for g in all_grids_for_ngpus}
    max_size = max(sizes.values())
    return '2D' if sizes[grid] == max_size else '3D'


def get_config_label(dim: str, spcomm: bool, workstealing: bool) -> str:
    """Get display label for an HnS configuration."""
    spcomm_str = 'spcomm' if spcomm else 'nospcomm'
    sched_str = 'workstealing' if workstealing else 'async'
    return f'HnS {dim} {spcomm_str} {sched_str}'


@dataclass
class HnsResult:
    """Parsed results from an HnS output file."""
    filepath: str
    ngpus: int
    grid: str  # e.g., "4x4"
    matrix: str
    backend: str
    spcomm: bool
    workstealing: bool
    # Overall spgemm runtimes (one per round, excluding round 0)
    spgemm_times_ms: list[float] = field(default_factory=list)
    # Per-phase timings: {phase_name: {process_rank: [sum values per round]}}
    phase_timings: dict[str, dict[int, list[float]]] = field(default_factory=dict)
    # Internode comm bytes: {process_rank: [total bytes per round]}
    internode_bytes: dict[int, list[float]] = field(default_factory=dict)


@dataclass
class TrilinosResult:
    """Parsed results from a Trilinos output file."""
    filepath: str
    ngpus: int
    matrix: str
    # Average spgemm runtime from summary line
    spgemm_avg_ms: Optional[float] = None


def parse_hns_filename(filename: str) -> Optional[dict]:
    """Parse HnS filename to extract metadata.

    Format: hns_strong_<ngpus>_<grid>_<matrix>_<backend>_<spcomm_mode>.out
    """
    basename = os.path.basename(filename)
    # hns_strong_16_4x4_mouse_gene_kokkos_spcomm.out
    pattern = r'hns_strong_(\d+)_(\d+x\d+)_(.+)_(\w+)_(spcomm|nospcomm)_(workstealing|async)\.out'
    match = re.match(pattern, basename)
    if not match:
        return None
    return {
        'ngpus': int(match.group(1)),
        'grid': match.group(2),
        'matrix': match.group(3),
        'backend': match.group(4),
        'spcomm': match.group(5) == 'spcomm',
        'workstealing': match.group(6) == 'workstealing'
    }


def parse_trilinos_filename(filename: str) -> Optional[dict]:
    """Parse Trilinos filename to extract metadata.

    Format: trilinos_strong_<matrix>_<ngpus>.out
    """
    basename = os.path.basename(filename)
    pattern = r'trilinos_strong_(.+)_(\d+)\.out'
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
        spcomm=meta['spcomm'],
        workstealing=meta['workstealing'],
    )

    with open(filepath, 'r') as f:
        content = f.read()

    # Track current round (starts at -1, incremented when we see "STARTING spgemm round:")
    current_round = -1

    # Temporary storage for internode bytes per round per process
    # {round: {process: total_bytes}}
    round_internode_bytes: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))

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
    # <[process P]>[phase_name] X.X ms (single value format, for spm_time, spadd_time)
    single_phase_pattern = re.compile(r'<\[process\s+(\d+)\]>\[(spm_time|spadd_time|wait_for_input|intranode_comm)\]\s+([\d.]+)\s+ms')
    # <[p P, ...]> ... XXXXX B, YYYYY B, ... (capture second B value)
    internode_pattern = re.compile(r'<\[p\s+(\d+),.*?\]>.*?\d+\s+B,\s*(\d+)\s+B')

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

        # Check for internode communication (skip round 0)
        internode_match = internode_pattern.search(line)
        if internode_match and current_round > 0:
            process = int(internode_match.group(1))
            bytes_val = float(internode_match.group(2))
            round_internode_bytes[current_round][process] += bytes_val
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

    # Internode bytes
    all_rounds = sorted(round_internode_bytes.keys())
    all_processes = set()
    for rd in round_internode_bytes.values():
        all_processes.update(rd.keys())

    for process in all_processes:
        result.internode_bytes[process] = [
            round_internode_bytes[r].get(process, 0.0) for r in all_rounds
        ]

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
    )

    with open(filepath, 'r') as f:
        content = f.read()

    # Look for summary line: <Timer>[spgemm] n=10,avg=1902.223877,...
    pattern = re.compile(r'<Timer>\[spgemm\].*avg=([\d.]+)')
    match = pattern.search(content)
    if match:
        result.spgemm_avg_ms = float(match.group(1))

    return result


def filter_hns_results(
    results: list[HnsResult],
    spcomm_filter: str = 'all',
    scheduling_filter: str = 'all'
) -> list[HnsResult]:
    """Filter HnS results based on spcomm and scheduling mode.

    Args:
        results: List of HnsResult objects
        spcomm_filter: 'all', 'spcomm', or 'nospcomm'
        scheduling_filter: 'all', 'workstealing', or 'async'

    Returns:
        Filtered list of HnsResult objects
    """
    filtered = results

    if spcomm_filter != 'all':
        spcomm_value = (spcomm_filter == 'spcomm')
        filtered = [r for r in filtered if r.spcomm == spcomm_value]

    if scheduling_filter != 'all':
        workstealing_value = (scheduling_filter == 'workstealing')
        filtered = [r for r in filtered if r.workstealing == workstealing_value]

    return filtered


def load_all_results(results_dir: str = DEFAULT_RESULTS_DIR) -> tuple[list[HnsResult], list[TrilinosResult]]:
    """Load all results from the given directory."""
    hns_results = []
    trilinos_results = []

    for filepath in glob(os.path.join(results_dir, '*.out')):
        basename = os.path.basename(filepath)

        if basename.startswith('hns_'):
            result = parse_hns_file(filepath)
            if result:
                hns_results.append(result)
        elif basename.startswith('trilinos_'):
            result = parse_trilinos_file(filepath)
            if result:
                trilinos_results.append(result)

    return hns_results, trilinos_results


def plot_runtime_comparison(
    hns_results: list[HnsResult],
    trilinos_results: list[TrilinosResult],
    output_dir: str,
) -> None:
    """Bar plot comparing HnS vs Trilinos spgemm runtimes per matrix."""
    import numpy as np

    # Group results by matrix
    matrices = set()
    for r in hns_results:
        matrices.add(r.matrix)
    for r in trilinos_results:
        matrices.add(r.matrix)

    for matrix in sorted(matrices):
        # Filter results for this matrix
        hns_for_matrix = [r for r in hns_results if r.matrix == matrix]
        trilinos_for_matrix = [r for r in trilinos_results if r.matrix == matrix]

        if not hns_for_matrix and not trilinos_for_matrix:
            continue

        # Collect all unique GPU counts
        gpu_counts = set()
        for r in hns_for_matrix:
            gpu_counts.add(r.ngpus)
        for r in trilinos_for_matrix:
            gpu_counts.add(r.ngpus)
        gpu_counts = sorted(gpu_counts)

        # For each GPU count, determine which grids map to 2D/3D
        grids_by_ngpus: dict[int, set[str]] = defaultdict(set)
        for r in hns_for_matrix:
            grids_by_ngpus[r.ngpus].add(r.grid)

        # Build data structure: {(dim, spcomm, workstealing): {ngpus: runtime_ms}}
        # Also track Trilinos separately
        hns_data: dict[tuple[str, bool, bool], dict[int, float]] = defaultdict(dict)
        trilinos_data: dict[int, float] = {}

        # Trilinos data
        for r in trilinos_for_matrix:
            if r.spgemm_avg_ms is not None:
                trilinos_data[r.ngpus] = r.spgemm_avg_ms

        # HnS data with 2D/3D labels
        for r in hns_for_matrix:
            if r.spgemm_times_ms:
                dim = get_dimension_label(r.grid, grids_by_ngpus[r.ngpus])
                avg_time = sum(r.spgemm_times_ms) / len(r.spgemm_times_ms)
                hns_data[(dim, r.spcomm, r.workstealing)][r.ngpus] = avg_time

        # Define config order: Trilinos first, then HnS configs
        config_order = [
            ('2D', True, True), ('2D', True, False),
            ('2D', False, True), ('2D', False, False),
            ('3D', True, True), ('3D', True, False),
            ('3D', False, True), ('3D', False, False)
        ]
        hns_configs_present = [c for c in config_order if c in hns_data]

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Determine bar positions
        n_configs = 1 + len(hns_configs_present)  # Trilinos + HnS configs
        bar_width = 0.8 / n_configs
        x = np.arange(len(gpu_counts))

        # Plot Trilinos bars
        values = [trilinos_data.get(g, np.nan) for g in gpu_counts]
        offset = (0 - n_configs / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width, label='Trilinos',
               color='tab:gray', edgecolor='black', linewidth=1)

        # Track which workstealing values we've added to legend
        legend_added = set()

        # Plot HnS bars with hatches
        for i, (dim, spcomm, workstealing) in enumerate(hns_configs_present, start=1):
            gpu_data = hns_data[(dim, spcomm, workstealing)]
            values = [gpu_data.get(g, np.nan) for g in gpu_counts]
            offset = (i - n_configs / 2 + 0.5) * bar_width

            # Only add legend label for first occurrence of each workstealing value
            if workstealing not in legend_added:
                label = 'Workstealing' if workstealing else 'Async'
                legend_added.add(workstealing)
            else:
                label = None

            bars = ax.bar(x + offset, values, bar_width, label=label,
                          color=CONFIG_COLORS[(dim, spcomm)],
                          hatch=WORKSTEALING_HATCHES[workstealing],
                          edgecolor='black', linewidth=1)

            # Add text labels above bars
            spcomm_str = 'spcomm' if spcomm else 'nospcomm'
            bar_label = f'{dim}\n{spcomm_str}'
            for j, (bar, value) in enumerate(zip(bars, values)):
                if not np.isnan(value):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           bar_label, ha='center', va='bottom', fontsize=7,
                           rotation=0)

        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Runtime (ms)')
        ax.set_title(f'SpGEMM Runtime Comparison: {matrix}')
        ax.set_xticks(x)
        ax.set_xticklabels(gpu_counts)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the plot
        plot_subdir = os.path.join(output_dir, 'runtime-comparison', matrix)
        os.makedirs(plot_subdir, exist_ok=True)
        output_path = os.path.join(plot_subdir, 'runtime_comparison.png')
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"    Saved {output_path}")


PHASE_DISPLAY_NAMES = {
    'wait_for_input': 'Internode Communication',
    'intranode_comm': 'Intranode Communication',
    'spm_time': 'Local SpGEMM',
    'spadd_time': 'Accumulation',
    'A_conversion': 'CSC Conversion',
}


def plot_runtime_breakdown_per_process(
    hns_results: list[HnsResult],
    trilinos_results: list[TrilinosResult],
    output_dir: str,
) -> None:
    """Stacked bar plot showing per-process runtime breakdown for HnS."""
    import numpy as np
    from matplotlib.patches import Patch

    # Group HnS results by (matrix, ngpus)
    groups: dict[tuple[str, int], list[HnsResult]] = defaultdict(list)
    for r in hns_results:
        groups[(r.matrix, r.ngpus)].append(r)

    for (matrix, ngpus), results in sorted(groups.items()):
        # Determine 2D/3D mapping for this GPU count
        all_grids = set(r.grid for r in results)

        # Collect all process ranks
        all_processes = set()
        for r in results:
            for phase_data in r.phase_timings.values():
                all_processes.update(phase_data.keys())
        all_processes = sorted(all_processes)

        if not all_processes:
            continue

        # Collect all phases across all results (only those in PHASE_DISPLAY_NAMES)
        all_phases = set()
        for r in results:
            all_phases.update(r.phase_timings.keys())
        all_phases = sorted([p for p in all_phases if p in PHASE_DISPLAY_NAMES])

        # Build data: {(dim, spcomm, workstealing): {process: {phase: avg_value}}}
        data: dict[tuple[str, bool, bool], dict[int, dict[str, float]]] = {}

        for r in results:
            dim = get_dimension_label(r.grid, all_grids)
            key = (dim, r.spcomm, r.workstealing)
            if key not in data:
                data[key] = defaultdict(dict)

            for phase, process_data in r.phase_timings.items():
                for process, round_values in process_data.items():
                    if round_values:
                        avg_val = sum(round_values) / len(round_values)
                        data[key][process][phase] = avg_val

        # Order configs consistently
        config_order = [
            ('2D', True, True), ('2D', True, False),
            ('2D', False, True), ('2D', False, False),
            ('3D', True, True), ('3D', True, False),
            ('3D', False, True), ('3D', False, False)
        ]
        configs_present = [c for c in config_order if c in data]

        # Create the plot
        n_configs = len(configs_present)
        n_processes = len(all_processes)

        fig, ax = plt.subplots(figsize=(max(12, n_processes * 0.5), 6))

        bar_width = 0.8 / n_configs
        x = np.arange(n_processes)

        # Use a colormap for phases
        phase_colors = plt.cm.tab10.colors

        # Plot stacked bars for each configuration
        for i, (dim, spcomm, workstealing) in enumerate(configs_present):
            process_data = data[(dim, spcomm, workstealing)]
            offset = (i - n_configs / 2 + 0.5) * bar_width
            bottom = np.zeros(n_processes)
            hatch = WORKSTEALING_HATCHES[workstealing]

            for j, phase in enumerate(all_phases):
                values = []
                for proc in all_processes:
                    val = process_data.get(proc, {}).get(phase, 0.0)
                    values.append(val)
                values = np.array(values)

                display_name = PHASE_DISPLAY_NAMES.get(phase, phase)
                # Only add phase label for the first config to avoid duplicate legend entries
                phase_label = display_name if i == 0 else None
                ax.bar(x + offset, values, bar_width, bottom=bottom,
                       label=phase_label, color=phase_colors[j % len(phase_colors)],
                       hatch=hatch, edgecolor='black', linewidth=0.5)
                bottom += values

        ax.set_xlabel('Process Rank')
        ax.set_ylabel('Runtime (ms)')
        ax.set_title(f'Per-Process Runtime Breakdown: {matrix} ({ngpus} GPUs)')
        ax.set_xticks(x)
        ax.set_xticklabels(all_processes)

        # Create two legends: one for phases, one for configs (hatches)
        phase_handles = [Patch(facecolor=phase_colors[j % len(phase_colors)], edgecolor='black',
                               label=PHASE_DISPLAY_NAMES.get(phase, phase))
                        for j, phase in enumerate(all_phases)]
        config_handles = [Patch(facecolor='white', edgecolor='black',
                                hatch=WORKSTEALING_HATCHES[workstealing],
                                label=get_config_label(dim, spcomm, workstealing))
                         for dim, spcomm, workstealing in configs_present]

        leg1 = ax.legend(handles=phase_handles, title='Phase', loc='upper left')
        ax.add_artist(leg1)
        ax.legend(handles=config_handles, title='Config', loc='upper right')

        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plot_subdir = os.path.join(output_dir, 'runtime-breakdown-per-process', matrix)
        os.makedirs(plot_subdir, exist_ok=True)
        output_path = os.path.join(plot_subdir, f'{ngpus}gpus.png')
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved {output_path}")


def plot_runtime_breakdown_overall(
    hns_results: list[HnsResult],
    trilinos_results: list[TrilinosResult],
    output_dir: str,
) -> None:
    """Stacked bar plot showing overall runtime breakdown (max across processes) for HnS."""
    import numpy as np
    from matplotlib.patches import Patch

    # Group HnS results by matrix
    matrices = set(r.matrix for r in hns_results)

    for matrix in sorted(matrices):
        # Filter results for this matrix
        results_for_matrix = [r for r in hns_results if r.matrix == matrix]

        if not results_for_matrix:
            continue

        # Collect all unique GPU counts
        gpu_counts = sorted(set(r.ngpus for r in results_for_matrix))

        # For each GPU count, determine which grids map to 2D/3D
        grids_by_ngpus: dict[int, set[str]] = defaultdict(set)
        for r in results_for_matrix:
            grids_by_ngpus[r.ngpus].add(r.grid)

        # Collect all phases (only those in PHASE_DISPLAY_NAMES)
        all_phases = set()
        for r in results_for_matrix:
            all_phases.update(r.phase_timings.keys())
        all_phases = sorted([p for p in all_phases if p in PHASE_DISPLAY_NAMES])

        # Build data: {(dim, spcomm, workstealing): {ngpus: {phase: max_avg_value}}}
        data: dict[tuple[str, bool, bool], dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

        for r in results_for_matrix:
            dim = get_dimension_label(r.grid, grids_by_ngpus[r.ngpus])
            key = (dim, r.spcomm, r.workstealing)

            for phase in all_phases:
                if phase in r.phase_timings:
                    # For each process, compute average across rounds
                    process_avgs = []
                    for process, round_values in r.phase_timings[phase].items():
                        if round_values:
                            avg_val = sum(round_values) / len(round_values)
                            process_avgs.append(avg_val)
                    # Take max across processes
                    if process_avgs:
                        data[key][r.ngpus][phase] = max(process_avgs)

        # Order configs consistently
        config_order = [
            ('2D', True, True), ('2D', True, False),
            ('2D', False, True), ('2D', False, False),
            ('3D', True, True), ('3D', True, False),
            ('3D', False, True), ('3D', False, False)
        ]
        configs_present = [c for c in config_order if c in data]

        # Create the plot
        n_configs = len(configs_present)
        n_gpu_counts = len(gpu_counts)

        fig, ax = plt.subplots(figsize=(max(10, n_gpu_counts * 2), 6))

        bar_width = 0.8 / n_configs
        x = np.arange(n_gpu_counts)

        # Use a colormap for phases
        phase_colors = plt.cm.tab10.colors

        # Plot stacked bars for each configuration
        for i, (dim, spcomm, workstealing) in enumerate(configs_present):
            gpu_data = data[(dim, spcomm, workstealing)]
            offset = (i - n_configs / 2 + 0.5) * bar_width
            bottom = np.zeros(n_gpu_counts)
            hatch = WORKSTEALING_HATCHES[workstealing]

            for j, phase in enumerate(all_phases):
                values = []
                for ngpus in gpu_counts:
                    val = gpu_data.get(ngpus, {}).get(phase, 0.0)
                    values.append(val)
                values = np.array(values)

                display_name = PHASE_DISPLAY_NAMES.get(phase, phase)
                # Only add phase label for the first config to avoid duplicate legend entries
                phase_label = display_name if i == 0 else None
                ax.bar(x + offset, values, bar_width, bottom=bottom,
                       label=phase_label, color=phase_colors[j % len(phase_colors)],
                       hatch=hatch, edgecolor='black', linewidth=0.5)
                bottom += values

            # Add text labels above each stacked bar
            spcomm_str = 'spcomm' if spcomm else 'nospcomm'
            bar_label = f'{dim}\n{spcomm_str}'
            for k, (x_pos, height) in enumerate(zip(x + offset, bottom)):
                if height > 0:
                    ax.text(x_pos, height, bar_label,
                           ha='center', va='bottom', fontsize=7, rotation=0)

        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Runtime (ms)')
        ax.set_title(f'Overall Runtime Breakdown: {matrix}')
        ax.set_xticks(x)
        ax.set_xticklabels(gpu_counts)

        # Create two legends: one for phases, one for workstealing mode
        phase_handles = [Patch(facecolor=phase_colors[j % len(phase_colors)], edgecolor='black',
                               label=PHASE_DISPLAY_NAMES.get(phase, phase))
                        for j, phase in enumerate(all_phases)]

        # Create workstealing legend entries (only unique values)
        workstealing_values = sorted(set(ws for _, _, ws in configs_present), reverse=True)
        workstealing_handles = [Patch(facecolor='white', edgecolor='black',
                                      hatch=WORKSTEALING_HATCHES[ws],
                                      label='Workstealing' if ws else 'Async')
                               for ws in workstealing_values]

        leg1 = ax.legend(handles=phase_handles, title='Phase', loc='upper left')
        ax.add_artist(leg1)
        ax.legend(handles=workstealing_handles, title='Scheduling', loc='upper right')

        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plot_subdir = os.path.join(output_dir, 'runtime-breakdown-overall', matrix)
        os.makedirs(plot_subdir, exist_ok=True)
        output_path = os.path.join(plot_subdir, 'runtime_breakdown_overall.png')
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved {output_path}")


def plot_comm_volume_per_process(
    hns_results: list[HnsResult],
    trilinos_results: list[TrilinosResult],
    output_dir: str,
) -> None:
    """Bar plot showing per-process internode communication volume for HnS."""
    import numpy as np

    # Group HnS results by (matrix, ngpus)
    groups: dict[tuple[str, int], list[HnsResult]] = defaultdict(list)
    for r in hns_results:
        groups[(r.matrix, r.ngpus)].append(r)

    for (matrix, ngpus), results in sorted(groups.items()):
        # Determine 2D/3D mapping for this GPU count
        all_grids = set(r.grid for r in results)

        # Collect all process ranks
        all_processes = set()
        for r in results:
            all_processes.update(r.internode_bytes.keys())
        all_processes = sorted(all_processes)

        if not all_processes:
            continue

        # Build data: {(dim, spcomm, workstealing): {process: avg_bytes}}
        data: dict[tuple[str, bool, bool], dict[int, float]] = {}

        for r in results:
            dim = get_dimension_label(r.grid, all_grids)
            key = (dim, r.spcomm, r.workstealing)
            if key not in data:
                data[key] = {}

            for process, round_values in r.internode_bytes.items():
                if round_values:
                    avg_val = sum(round_values) / len(round_values)
                    data[key][process] = avg_val

        # Order configs consistently
        config_order = [
            ('2D', True, True), ('2D', True, False),
            ('2D', False, True), ('2D', False, False),
            ('3D', True, True), ('3D', True, False),
            ('3D', False, True), ('3D', False, False)
        ]
        configs_present = [c for c in config_order if c in data]

        # Create the plot
        n_configs = len(configs_present)
        n_processes = len(all_processes)

        fig, ax = plt.subplots(figsize=(max(12, n_processes * 0.5), 6))

        bar_width = 0.8 / n_configs
        x = np.arange(n_processes)

        # Plot bars for each configuration
        for i, (dim, spcomm, workstealing) in enumerate(configs_present):
            process_data = data[(dim, spcomm, workstealing)]
            offset = (i - n_configs / 2 + 0.5) * bar_width
            values = [process_data.get(proc, 0.0) for proc in all_processes]
            label = get_config_label(dim, spcomm, workstealing)
            ax.bar(x + offset, values, bar_width, label=label,
                   color=CONFIG_COLORS[(dim, spcomm)],
                   hatch=WORKSTEALING_HATCHES[workstealing],
                   edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Process Rank')
        ax.set_ylabel('Communication Volume (bytes)')
        ax.set_title(f'Per-Process Internode Communication Volume: {matrix} ({ngpus} GPUs)')
        ax.set_xticks(x)
        ax.set_xticklabels(all_processes)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Use scientific notation for y-axis if values are large
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(6, 6))

        plot_subdir = os.path.join(output_dir, 'comm-volume-per-process', matrix)
        os.makedirs(plot_subdir, exist_ok=True)
        output_path = os.path.join(plot_subdir, f'{ngpus}gpus.png')
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved {output_path}")


def get_available_plots():
    """Return a dictionary of available plot types and their functions."""
    return {
        'runtime-comparison': plot_runtime_comparison,
        'runtime-breakdown-per-process': plot_runtime_breakdown_per_process,
        'runtime-breakdown-overall': plot_runtime_breakdown_overall,
        'comm-volume-per-process': plot_comm_volume_per_process,
    }


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
    parser.add_argument(
        '--spcomm',
        choices=['all', 'spcomm', 'nospcomm'],
        default='all',
        help='Filter HnS results by spcomm mode (default: all)'
    )
    parser.add_argument(
        '--scheduling',
        choices=['all', 'workstealing', 'async'],
        default='all',
        help='Filter HnS results by scheduling mode (default: all)'
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
    hns_results, trilinos_results = load_all_results(args.results_dir)
    print(f"  Loaded {len(hns_results)} HnS results and {len(trilinos_results)} Trilinos results.")

    # Apply filters
    hns_results = filter_hns_results(hns_results, args.spcomm, args.scheduling)
    print(f"  After filtering: {len(hns_results)} HnS results (spcomm={args.spcomm}, scheduling={args.scheduling})")

    # Determine which plots to generate
    if 'all' in args.plots:
        plots_to_make = list(available_plots.keys())
    else:
        plots_to_make = args.plots

    # Generate each requested plot
    for plot_name in plots_to_make:
        print(f"Generating {plot_name}...")
        plot_func = available_plots[plot_name]
        plot_func(hns_results, trilinos_results, args.output_dir)
        print(f"  Done.")


if __name__ == '__main__':
    main()
