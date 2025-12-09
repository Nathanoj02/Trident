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
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results_wave6')

# Implementation types
IMPL_NONE = 'None'
IMPL_PERMUTE = 'Permute'
IMPL_WORKSTEALING = 'Workstealing'

# Accumthread implementation types
IMPL_ACCUMTHREAD_NONE = 'Accumthread None'
IMPL_ACCUMTHREAD_PERMUTE = 'Accumthread Permute'

# Colors for line plots (HnS implementations + Trilinos)
IMPL_COLORS = {
    'Trilinos': '#7A9DCF',      # soft blue
    IMPL_NONE: '#F4A582',       # soft coral
    IMPL_PERMUTE: '#92C5A9',    # soft sage
    IMPL_WORKSTEALING: '#C9A9D4',  # soft lavender
    IMPL_ACCUMTHREAD_NONE: '#E8A838',     # golden orange
    IMPL_ACCUMTHREAD_PERMUTE: '#5DADE2',  # sky blue
}

# Hatches for breakdown plots (to distinguish implementations)
IMPL_HATCHES = {
    IMPL_NONE: '',          # solid
    IMPL_PERMUTE: '///',    # diagonal lines
    IMPL_WORKSTEALING: '...',  # dots
}

# Hatches for accumthread breakdown plots
IMPL_ACCUMTHREAD_HATCHES = {
    IMPL_ACCUMTHREAD_NONE: '',       # solid
    IMPL_ACCUMTHREAD_PERMUTE: '///',  # diagonal lines
}


def get_implementation_type(scheduling: str, permute: bool) -> str:
    """Determine implementation type from scheduling mode and permute flag.

    Returns one of: IMPL_NONE, IMPL_PERMUTE, IMPL_WORKSTEALING
    """
    if scheduling == 'workstealing':
        return IMPL_WORKSTEALING
    elif permute:
        return IMPL_PERMUTE
    else:
        return IMPL_NONE


def get_accumthread_implementation_type(permute: bool) -> str:
    """Determine accumthread implementation type from permute flag.

    Returns one of: IMPL_ACCUMTHREAD_NONE, IMPL_ACCUMTHREAD_PERMUTE
    """
    if permute:
        return IMPL_ACCUMTHREAD_PERMUTE
    else:
        return IMPL_ACCUMTHREAD_NONE


@dataclass
class HnsResult:
    """Parsed results from an HnS output file."""
    filepath: str
    ngpus: int
    grid: str  # e.g., "4x4"
    matrix: str
    backend: str
    implementation: str  # One of IMPL_NONE, IMPL_PERMUTE, IMPL_WORKSTEALING, or accumthread variants
    is_accumthread: bool = False  # True if this is an accumthread result
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
    # Average spgemm runtime from summary line
    spgemm_avg_ms: Optional[float] = None


def parse_hns_filename(filename: str) -> Optional[dict]:
    """Parse HnS filename to extract metadata.

    Format: hns_strong_<ngpus>_<grid>_<matrix>_<backend>_nospcomm_<scheduling>_<permute>.out
    """
    basename = os.path.basename(filename)
    # hns_strong_16_2x2_mouse_gene_kokkos_nospcomm_async_nopermute.out
    pattern = r'hns_strong_(\d+)_(\d+x\d+)_(.+)_(\w+)_nospcomm_(workstealing|async)_(permute|nopermute)\.out'
    match = re.match(pattern, basename)
    if not match:
        return None
    scheduling = match.group(5)
    permute = match.group(6) == 'permute'
    return {
        'ngpus': int(match.group(1)),
        'grid': match.group(2),
        'matrix': match.group(3),
        'backend': match.group(4),
        'implementation': get_implementation_type(scheduling, permute),
    }


def parse_hns_accumthread_filename(filename: str) -> Optional[dict]:
    """Parse HnS accumthread filename to extract metadata.

    Format: hns_accumthread_strong_<ngpus>_<grid>_<matrix>_<backend>_nospcomm_async_<permute>.out
    Example: hns_accumthread_strong_64_4x4_HV15R_kokkos_nospcomm_async_nopermute.out
    """
    basename = os.path.basename(filename)
    pattern = r'hns_accumthread_strong_(\d+)_(\d+x\d+)_(.+)_(\w+)_nospcomm_async_(permute|nopermute)\.out'
    match = re.match(pattern, basename)
    if not match:
        return None
    permute = match.group(5) == 'permute'
    return {
        'ngpus': int(match.group(1)),
        'grid': match.group(2),
        'matrix': match.group(3),
        'backend': match.group(4),
        'implementation': get_accumthread_implementation_type(permute),
        'is_accumthread': True,
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


def parse_hns_file(filepath: str, is_accumthread: bool = False) -> Optional[HnsResult]:
    """Parse an HnS output file and extract timing/communication data."""
    if is_accumthread:
        meta = parse_hns_accumthread_filename(filepath)
    else:
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
        is_accumthread=meta.get('is_accumthread', False),
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
    # <[process P]>[phase_name] X.X ms (single value format, for spm_time, spadd_time, queue phases)
    single_phase_pattern = re.compile(r'<\[process\s+(\d+)\]>\[(spm_time|spadd_time|wait_for_input|intranode_comm|task_queue_checkn|task_queue_incn|task_queue_pop)\]\s+([\d.]+)\s+ms')

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
    )

    with open(filepath, 'r') as f:
        content = f.read()

    # Look for summary line: <Timer>[spgemm] n=10,avg=1902.223877,...
    pattern = re.compile(r'<Timer>\[spgemm\].*avg=([\d.]+)')
    match = pattern.search(content)
    if match:
        result.spgemm_avg_ms = float(match.group(1))

    return result


def load_all_results(results_dir: str = DEFAULT_RESULTS_DIR) -> tuple[list[HnsResult], list[TrilinosResult]]:
    """Load all results from the given directory."""
    hns_results = []
    trilinos_results = []

    for filepath in glob(os.path.join(results_dir, '*.out')):
        basename = os.path.basename(filepath)

        if basename.startswith('hns_accumthread_'):
            result = parse_hns_file(filepath, is_accumthread=True)
            if result:
                hns_results.append(result)
        elif basename.startswith('hns_'):
            result = parse_hns_file(filepath, is_accumthread=False)
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
    """Line plot comparing HnS vs Trilinos spgemm runtimes per matrix."""
    import numpy as np

    # Group results by matrix
    matrices = set()
    for r in hns_results:
        matrices.add(r.matrix)
    for r in trilinos_results:
        matrices.add(r.matrix)

    # Implementation order for consistent plotting (regular HnS + accumthread)
    impl_order = [IMPL_NONE, IMPL_PERMUTE, IMPL_WORKSTEALING]
    accumthread_impl_order = [IMPL_ACCUMTHREAD_NONE, IMPL_ACCUMTHREAD_PERMUTE]

    # Markers for different implementation types
    impl_markers = {
        IMPL_NONE: 'o',
        IMPL_PERMUTE: 'o',
        IMPL_WORKSTEALING: 'o',
        IMPL_ACCUMTHREAD_NONE: 's',      # square
        IMPL_ACCUMTHREAD_PERMUTE: 's',   # square
    }

    for matrix in sorted(matrices):
        # Filter results for this matrix (separate regular and accumthread)
        hns_for_matrix = [r for r in hns_results if r.matrix == matrix and not r.is_accumthread]
        accumthread_for_matrix = [r for r in hns_results if r.matrix == matrix and r.is_accumthread]
        trilinos_for_matrix = [r for r in trilinos_results if r.matrix == matrix]

        if not hns_for_matrix and not accumthread_for_matrix and not trilinos_for_matrix:
            continue

        # Collect all unique GPU counts
        gpu_counts = set()
        for r in hns_for_matrix:
            gpu_counts.add(r.ngpus)
        for r in accumthread_for_matrix:
            gpu_counts.add(r.ngpus)
        for r in trilinos_for_matrix:
            gpu_counts.add(r.ngpus)
        gpu_counts = sorted(gpu_counts)

        # Build data structure: {implementation: {ngpus: runtime_ms}}
        hns_data: dict[str, dict[int, float]] = defaultdict(dict)
        trilinos_data: dict[int, float] = {}

        # Trilinos data
        for r in trilinos_for_matrix:
            if r.spgemm_avg_ms is not None:
                trilinos_data[r.ngpus] = r.spgemm_avg_ms

        # HnS data by implementation (regular)
        for r in hns_for_matrix:
            if r.spgemm_times_ms:
                avg_time = sum(r.spgemm_times_ms) / len(r.spgemm_times_ms)
                hns_data[r.implementation][r.ngpus] = avg_time

        # HnS data by implementation (accumthread)
        for r in accumthread_for_matrix:
            if r.spgemm_times_ms:
                avg_time = sum(r.spgemm_times_ms) / len(r.spgemm_times_ms)
                hns_data[r.implementation][r.ngpus] = avg_time

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot Trilinos line
        if trilinos_data:
            x_vals = sorted(trilinos_data.keys())
            y_vals = [trilinos_data[g] for g in x_vals]
            ax.plot(x_vals, y_vals, 'o-', label='Trilinos',
                    color=IMPL_COLORS['Trilinos'], linewidth=2, markersize=8)

        # Plot HnS lines for each regular implementation
        for impl in impl_order:
            if impl in hns_data:
                gpu_data = hns_data[impl]
                x_vals = sorted(gpu_data.keys())
                y_vals = [gpu_data[g] for g in x_vals]
                marker = impl_markers.get(impl, 'o')
                ax.plot(x_vals, y_vals, f'{marker}-', label=f'HnS {impl}',
                        color=IMPL_COLORS[impl], linewidth=2, markersize=8)

        # Plot HnS lines for each accumthread implementation
        for impl in accumthread_impl_order:
            if impl in hns_data:
                gpu_data = hns_data[impl]
                x_vals = sorted(gpu_data.keys())
                y_vals = [gpu_data[g] for g in x_vals]
                marker = impl_markers.get(impl, 's')
                ax.plot(x_vals, y_vals, f'{marker}-', label=f'HnS {impl}',
                        color=IMPL_COLORS[impl], linewidth=2, markersize=8)

        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Runtime (ms)')
        ax.set_title(f'SpGEMM Runtime Comparison: {matrix}')

        # Set log scales
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=10)

        # Set x-axis ticks to the actual GPU counts
        ax.set_xticks(gpu_counts)
        ax.set_xticklabels([str(g) for g in gpu_counts])

        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7, which='both')

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
    'task_queue_checkn': '#AA3377', # purple - queue check
    'task_queue_incn': '#BBBBBB',   # gray - queue increment
    'task_queue_pop': '#44AA99',    # teal - queue pop
}

# Phases that are queue-related (for filtering)
QUEUE_PHASES = {'task_queue_checkn', 'task_queue_incn', 'task_queue_pop'}


def _plot_runtime_breakdown_per_process_impl(
    results_list: list[HnsResult],
    output_dir: str,
    impl_order: list[str],
    hatches_dict: dict[str, str],
    filename_suffix: str = '',
    title_suffix: str = '',
) -> None:
    """Internal implementation for per-process runtime breakdown plots."""
    import numpy as np
    from matplotlib.patches import Patch

    # Group results by (matrix, ngpus)
    groups: dict[tuple[str, int], list[HnsResult]] = defaultdict(list)
    for r in results_list:
        groups[(r.matrix, r.ngpus)].append(r)

    for (matrix, ngpus), results in sorted(groups.items()):
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

        # Build data: {implementation: {process: {phase: avg_value}}}
        data: dict[str, dict[int, dict[str, float]]] = {}

        for r in results:
            impl = r.implementation
            if impl not in data:
                data[impl] = defaultdict(dict)

            for phase, process_data in r.phase_timings.items():
                for process, round_values in process_data.items():
                    if round_values:
                        avg_val = sum(round_values) / len(round_values)
                        data[impl][process][phase] = avg_val

        # Get implementations present in order
        impls_present = [impl for impl in impl_order if impl in data]

        if not impls_present:
            continue

        # Create the plot
        n_impls = len(impls_present)
        n_processes = len(all_processes)

        fig, ax = plt.subplots(figsize=(max(12, n_processes * 0.5), 6))

        bar_width = 0.8 / n_impls
        x = np.arange(n_processes)

        # Plot stacked bars for each implementation
        for i, impl in enumerate(impls_present):
            process_data = data[impl]
            offset = (i - n_impls / 2 + 0.5) * bar_width
            bottom = np.zeros(n_processes)
            hatch = hatches_dict.get(impl, '')

            for phase in all_phases:
                values = []
                for proc in all_processes:
                    val = process_data.get(proc, {}).get(phase, 0.0)
                    values.append(val)
                values = np.array(values)

                display_name = PHASE_DISPLAY_NAMES.get(phase, phase)
                color = PHASE_COLORS.get(phase, '#888888')
                # Only add phase label for the first implementation to avoid duplicate legend entries
                phase_label = display_name if i == 0 else None
                ax.bar(x + offset, values, bar_width, bottom=bottom,
                       label=phase_label, color=color,
                       hatch=hatch, edgecolor='black', linewidth=0.5)
                bottom += values

        ax.set_xlabel('Process Rank')
        ax.set_ylabel('Runtime (ms)')
        title = f'Per-Process Runtime Breakdown: {matrix} ({ngpus} GPUs)'
        if title_suffix:
            title += f' {title_suffix}'
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(all_processes)

        # Create two legends: one for phases, one for implementations (hatches)
        phase_handles = [Patch(facecolor=PHASE_COLORS.get(phase, '#888888'), edgecolor='black',
                               label=PHASE_DISPLAY_NAMES.get(phase, phase))
                        for phase in all_phases]
        impl_handles = [Patch(facecolor='white', edgecolor='black',
                              hatch=hatches_dict.get(impl, ''),
                              label=f'HnS {impl}')
                       for impl in impls_present]

        leg1 = ax.legend(handles=phase_handles, title='Phase', loc='upper left')
        ax.add_artist(leg1)
        ax.legend(handles=impl_handles, title='Implementation', loc='upper right')

        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plot_subdir = os.path.join(output_dir, 'runtime-breakdown-per-process', matrix)
        os.makedirs(plot_subdir, exist_ok=True)
        filename = f'{ngpus}gpus{filename_suffix}.png'
        output_path = os.path.join(plot_subdir, filename)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved {output_path}")


def plot_runtime_breakdown_per_process(
    hns_results: list[HnsResult],
    trilinos_results: list[TrilinosResult],
    output_dir: str,
) -> None:
    """Stacked bar plot showing per-process runtime breakdown for HnS.

    Generates separate plots for regular HnS and accumthread implementations.
    """
    # Regular HnS results
    regular_results = [r for r in hns_results if not r.is_accumthread]
    impl_order = [IMPL_NONE, IMPL_PERMUTE, IMPL_WORKSTEALING]
    _plot_runtime_breakdown_per_process_impl(
        regular_results, output_dir, impl_order, IMPL_HATCHES
    )

    # Accumthread results
    accumthread_results = [r for r in hns_results if r.is_accumthread]
    accumthread_impl_order = [IMPL_ACCUMTHREAD_NONE, IMPL_ACCUMTHREAD_PERMUTE]
    _plot_runtime_breakdown_per_process_impl(
        accumthread_results, output_dir, accumthread_impl_order,
        IMPL_ACCUMTHREAD_HATCHES, filename_suffix='_accumthread',
        title_suffix='(Accumthread)'
    )


def _plot_runtime_breakdown_overall_impl(
    hns_results: list[HnsResult],
    output_dir: str,
    include_queue: bool,
    subdir_name: str,
    impl_order: list[str],
    hatches_dict: dict[str, str],
    title_suffix: str = '',
    filename_suffix: str = '',
) -> None:
    """Internal implementation for overall runtime breakdown plots."""
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

        # Collect all phases (only those in PHASE_DISPLAY_NAMES, optionally filter queue)
        all_phases = set()
        for r in results_for_matrix:
            all_phases.update(r.phase_timings.keys())
        all_phases = sorted([p for p in all_phases if p in PHASE_DISPLAY_NAMES])
        if not include_queue:
            all_phases = [p for p in all_phases if p not in QUEUE_PHASES]

        # Build data: {implementation: {ngpus: {phase: max_avg_value}}}
        data: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

        for r in results_for_matrix:
            impl = r.implementation

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
                        data[impl][r.ngpus][phase] = max(process_avgs)

        # Get implementations present in order
        impls_present = [impl for impl in impl_order if impl in data]

        if not impls_present:
            continue

        # Create the plot
        n_impls = len(impls_present)
        n_gpu_counts = len(gpu_counts)

        fig, ax = plt.subplots(figsize=(max(10, n_gpu_counts * 2), 6))

        bar_width = 0.8 / n_impls
        x = np.arange(n_gpu_counts)

        # Plot stacked bars for each implementation
        for i, impl in enumerate(impls_present):
            gpu_data = data[impl]
            offset = (i - n_impls / 2 + 0.5) * bar_width
            bottom = np.zeros(n_gpu_counts)
            hatch = hatches_dict.get(impl, '')

            for phase in all_phases:
                values = []
                for ngpus in gpu_counts:
                    val = gpu_data.get(ngpus, {}).get(phase, 0.0)
                    values.append(val)
                values = np.array(values)

                display_name = PHASE_DISPLAY_NAMES.get(phase, phase)
                color = PHASE_COLORS.get(phase, '#888888')
                # Only add phase label for the first implementation to avoid duplicate legend entries
                phase_label = display_name if i == 0 else None
                ax.bar(x + offset, values, bar_width, bottom=bottom,
                       label=phase_label, color=color,
                       hatch=hatch, edgecolor='black', linewidth=0.5)
                bottom += values

        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Runtime (ms)')
        title = f'Overall Runtime Breakdown: {matrix}'
        if title_suffix:
            title += f' {title_suffix}'
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(gpu_counts)

        # Create two legends: one for phases, one for implementations
        phase_handles = [Patch(facecolor=PHASE_COLORS.get(phase, '#888888'), edgecolor='black',
                               label=PHASE_DISPLAY_NAMES.get(phase, phase))
                        for phase in all_phases]

        impl_handles = [Patch(facecolor='white', edgecolor='black',
                              hatch=hatches_dict.get(impl, ''),
                              label=f'HnS {impl}')
                       for impl in impls_present]

        leg1 = ax.legend(handles=phase_handles, title='Phase', loc='upper left')
        ax.add_artist(leg1)
        ax.legend(handles=impl_handles, title='Implementation', loc='upper right')

        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plot_subdir = os.path.join(output_dir, subdir_name, matrix)
        os.makedirs(plot_subdir, exist_ok=True)
        filename = f'runtime_breakdown_overall{filename_suffix}.png'
        output_path = os.path.join(plot_subdir, filename)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved {output_path}")


def plot_runtime_breakdown_overall(
    hns_results: list[HnsResult],
    trilinos_results: list[TrilinosResult],
    output_dir: str,
) -> None:
    """Stacked bar plot showing overall runtime breakdown (max across processes) for HnS.

    Generates versions for regular HnS and accumthread, each with/without queue phases.
    """
    # Regular HnS results
    regular_results = [r for r in hns_results if not r.is_accumthread]
    impl_order = [IMPL_NONE, IMPL_PERMUTE, IMPL_WORKSTEALING]

    # Regular HnS: Version without queue phases
    _plot_runtime_breakdown_overall_impl(
        regular_results, output_dir,
        include_queue=False,
        subdir_name='runtime-breakdown-overall',
        impl_order=impl_order,
        hatches_dict=IMPL_HATCHES,
    )
    # Regular HnS: Version with queue phases
    _plot_runtime_breakdown_overall_impl(
        regular_results, output_dir,
        include_queue=True,
        subdir_name='runtime-breakdown-overall-queue',
        impl_order=impl_order,
        hatches_dict=IMPL_HATCHES,
        title_suffix='(with Queue)',
    )

    # Accumthread results
    accumthread_results = [r for r in hns_results if r.is_accumthread]
    accumthread_impl_order = [IMPL_ACCUMTHREAD_NONE, IMPL_ACCUMTHREAD_PERMUTE]

    # Accumthread: Version without queue phases
    _plot_runtime_breakdown_overall_impl(
        accumthread_results, output_dir,
        include_queue=False,
        subdir_name='runtime-breakdown-overall',
        impl_order=accumthread_impl_order,
        hatches_dict=IMPL_ACCUMTHREAD_HATCHES,
        title_suffix='(Accumthread)',
        filename_suffix='_accumthread',
    )
    # Accumthread: Version with queue phases
    _plot_runtime_breakdown_overall_impl(
        accumthread_results, output_dir,
        include_queue=True,
        subdir_name='runtime-breakdown-overall-queue',
        impl_order=accumthread_impl_order,
        hatches_dict=IMPL_ACCUMTHREAD_HATCHES,
        title_suffix='(Accumthread, with Queue)',
        filename_suffix='_accumthread',
    )


def _plot_runtime_per_process_stack_impl(
    results_list: list[HnsResult],
    output_dir: str,
    impl_order: list[str],
    filename_suffix: str = '',
    title_suffix: str = '',
) -> None:
    """Internal implementation for per-process stack plots."""
    import numpy as np
    from matplotlib.patches import Patch

    # Group results by (matrix, ngpus)
    groups: dict[tuple[str, int], list[HnsResult]] = defaultdict(list)
    for r in results_list:
        groups[(r.matrix, r.ngpus)].append(r)

    for (matrix, ngpus), results in sorted(groups.items()):
        # Build data: {(implementation, grid): {phase: {process: avg_value}}}
        data: dict[tuple[str, str], dict[str, dict[int, float]]] = {}

        # Collect all phases present
        all_phases_set = set()

        for r in results:
            key = (r.implementation, r.grid)
            if key not in data:
                data[key] = defaultdict(lambda: defaultdict(float))

            for phase, process_data in r.phase_timings.items():
                # Only include phases we care about
                if phase not in PHASE_DISPLAY_NAMES:
                    continue
                all_phases_set.add(phase)
                for process, round_values in process_data.items():
                    if round_values:
                        avg_val = sum(round_values) / len(round_values)
                        data[key][phase][process] = avg_val

        # Sort phases consistently
        all_phases = sorted([p for p in all_phases_set if p in PHASE_DISPLAY_NAMES])

        # Get (impl, grid) keys present, ordered by implementation then grid
        keys_present = []
        for impl in impl_order:
            impl_keys = sorted([k for k in data.keys() if k[0] == impl], key=lambda k: k[1])
            keys_present.extend(impl_keys)

        if not keys_present:
            continue

        # Collect all process ranks across all configurations
        all_processes = set()
        for phase_data in data.values():
            for process_data in phase_data.values():
                all_processes.update(process_data.keys())
        all_processes = sorted(all_processes)

        if not all_processes:
            continue

        # Create vertically stacked subplots with fixed figure size
        n_configs = len(keys_present)
        fig, axes = plt.subplots(n_configs, 1, figsize=(12, 3 * n_configs), sharex=True)

        # Handle single subplot case
        if n_configs == 1:
            axes = [axes]

        x = np.array(all_processes)

        # Find global y-max for consistent y-axis across subplots (sum of all phases)
        y_max = 0
        for key in keys_present:
            phase_data = data[key]
            totals = [0.0] * len(all_processes)
            for phase in all_phases:
                for j, proc in enumerate(all_processes):
                    totals[j] += phase_data.get(phase, {}).get(proc, 0.0)
            y_max = max(y_max, max(totals) if totals else 0)

        # Plot each (implementation, grid) in its own subplot
        for i, key in enumerate(keys_present):
            impl, grid = key
            ax = axes[i]
            phase_data = data[key]

            # Build stack data: list of arrays, one per phase
            stack_data = []
            colors = []
            labels = []
            for phase in all_phases:
                values = [phase_data.get(phase, {}).get(proc, 0.0) for proc in all_processes]
                stack_data.append(values)
                colors.append(PHASE_COLORS.get(phase, '#888888'))
                labels.append(PHASE_DISPLAY_NAMES.get(phase, phase))

            if stack_data:
                ax.stackplot(x, stack_data, colors=colors, edgecolor='black', linewidth=0.5)

            ax.set_ylabel('Runtime (ms)')
            ax.set_title(f'HnS {impl} ({grid})')
            ax.set_ylim(0, y_max * 1.1)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Set x-axis label only on bottom subplot
        axes[-1].set_xlabel('Process Rank')

        # Create a single legend for phases (on the first subplot)
        phase_handles = [Patch(facecolor=PHASE_COLORS.get(phase, '#888888'), edgecolor='black',
                               label=PHASE_DISPLAY_NAMES.get(phase, phase))
                        for phase in all_phases]
        axes[0].legend(handles=phase_handles, title='Phase', loc='upper right')

        title = f'Per-Process Runtime Breakdown: {matrix} ({ngpus} GPUs)'
        if title_suffix:
            title += f' {title_suffix}'
        fig.suptitle(title, fontsize=12)

        plot_subdir = os.path.join(output_dir, 'runtime-per-process-stack', matrix)
        os.makedirs(plot_subdir, exist_ok=True)
        filename = f'{ngpus}gpus{filename_suffix}.png'
        output_path = os.path.join(plot_subdir, filename)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved {output_path}")


def plot_runtime_per_process_stack(
    hns_results: list[HnsResult],
    trilinos_results: list[TrilinosResult],
    output_dir: str,
) -> None:
    """Vertically stacked subplots showing per-phase runtime breakdown per process for each HnS implementation.

    Generates separate plots for regular HnS and accumthread implementations.
    """
    # Regular HnS results
    regular_results = [r for r in hns_results if not r.is_accumthread]
    impl_order = [IMPL_NONE, IMPL_PERMUTE, IMPL_WORKSTEALING]
    _plot_runtime_per_process_stack_impl(regular_results, output_dir, impl_order)

    # Accumthread results
    accumthread_results = [r for r in hns_results if r.is_accumthread]
    accumthread_impl_order = [IMPL_ACCUMTHREAD_NONE, IMPL_ACCUMTHREAD_PERMUTE]
    _plot_runtime_per_process_stack_impl(
        accumthread_results, output_dir, accumthread_impl_order,
        filename_suffix='_accumthread', title_suffix='(Accumthread)'
    )


def get_available_plots():
    """Return a dictionary of available plot types and their functions."""
    return {
        'runtime-comparison': plot_runtime_comparison,
        'runtime-breakdown-per-process': plot_runtime_breakdown_per_process,
        'runtime-breakdown-overall': plot_runtime_breakdown_overall,
        'runtime-per-process-stack': plot_runtime_per_process_stack,
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
