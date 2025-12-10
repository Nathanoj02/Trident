#!/usr/bin/env python3
"""
Matrix Sparsity Pattern Visualizer
Reads MatrixMarket files and generates sparsity pattern images like SuiteSparse
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import mmread
import argparse
import os
from pathlib import Path


def read_matrix_market(filepath):
    """Read a MatrixMarket file and return the sparse matrix."""
    try:
        matrix = mmread(filepath)
        return matrix
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None


def plot_sparsity_pattern(matrix, output_path=None, title=None, dpi=150, 
                          marker_size=1, color='black', background='white'):
    """
    Plot the sparsity pattern of a sparse matrix.
    
    Parameters:
    -----------
    matrix : scipy sparse matrix
        The sparse matrix to visualize
    output_path : str, optional
        Path to save the image. If None, displays the plot.
    title : str, optional
        Title for the plot. If None, generates from matrix dimensions.
    dpi : int
        Resolution of the output image
    marker_size : float
        Size of the markers representing non-zeros
    color : str
        Color of the non-zero markers
    background : str
        Background color of the plot
    """
    # Convert to COO format for easy plotting
    if not hasattr(matrix, 'tocoo'):
        matrix = matrix.tocoo()
    else:
        matrix = matrix.tocoo()
    
    nrows, ncols = matrix.shape
    nnz = matrix.nnz
    
    # Create figure with appropriate size
    fig_width = 10
    fig_height = 10 * (nrows / ncols) if ncols > 0 else 10
    fig_height = max(6, min(fig_height, 15))  # Clamp between 6 and 15
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    
    # Plot non-zero entries
    # Note: We flip the y-axis to match matrix indexing (top-left is (0,0))
    ax.scatter(matrix.col, matrix.row, s=marker_size, c=color, marker='s', 
               linewidths=0, alpha=0.8)
    
    # Set labels and title
    if title is None:
        density = (nnz / (nrows * ncols) * 100) if (nrows * ncols) > 0 else 0
        title = f'Sparsity Pattern: {nrows}×{ncols}, nnz={nnz:,} ({density:.2f}%)'
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Column Index', fontsize=10)
    ax.set_ylabel('Row Index', fontsize=10)
    
    # Invert y-axis so (0,0) is at top-left like matrix notation
    ax.invert_yaxis()
    
    # Set axis limits with small padding
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)
    
    # Add grid for better readability (optional, only for small matrices)
    if nrows <= 100 and ncols <= 100:
        ax.grid(True, alpha=0.2, linewidth=0.5)
    
    # Set background color
    ax.set_facecolor(background)
    fig.patch.set_facecolor('white')
    
    # Format tick labels for large matrices
    ax.ticklabel_format(style='plain', axis='both')
    
    # Make layout tight
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Sparsity pattern saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_matrix_info(matrix):
    """Generate statistics about the matrix."""
    nrows, ncols = matrix.shape
    nnz = matrix.nnz
    density = (nnz / (nrows * ncols) * 100) if (nrows * ncols) > 0 else 0
    
    # Convert to CSR for row statistics
    csr = matrix.tocsr()
    nonzeros_per_row = np.diff(csr.indptr)
    
    info = {
        'dimensions': f'{nrows} × {ncols}',
        'nonzeros': f'{nnz:,}',
        'density': f'{density:.4f}%',
        'avg_nnz_per_row': f'{np.mean(nonzeros_per_row):.2f}',
        'max_nnz_per_row': f'{np.max(nonzeros_per_row)}',
        'min_nnz_per_row': f'{np.min(nonzeros_per_row)}',
        'symmetric': 'Unknown'  # Would need to check explicitly
    }
    
    return info


def print_matrix_info(info):
    """Print matrix information in a formatted way."""
    print("\n" + "="*50)
    print("Matrix Information")
    print("="*50)
    for key, value in info.items():
        key_formatted = key.replace('_', ' ').title()
        print(f"{key_formatted:.<30} {value}")
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize sparsity pattern of MatrixMarket files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s matrix.mtx
  %(prog)s matrix.mtx -o output.png
  %(prog)s matrix.mtx -o output.png --dpi 300 --size 2
  %(prog)s matrix.mtx --color blue --background lightgray
        """
    )
    
    parser.add_argument('input', type=str,
                       help='Path to MatrixMarket (.mtx) file')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output image path (if not specified, displays the plot)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for output image (default: 150)')
    parser.add_argument('--size', type=float, default=1.0,
                       help='Marker size for non-zeros (default: 1.0)')
    parser.add_argument('--color', type=str, default='black',
                       help='Color for non-zero markers (default: black)')
    parser.add_argument('--background', type=str, default='white',
                       help='Background color (default: white)')
    parser.add_argument('--title', type=str, default=None,
                       help='Custom title for the plot')
    parser.add_argument('--info', action='store_true',
                       help='Print matrix information')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' not found")
        return 1
    
    # Read matrix
    print(f"Reading matrix from {args.input}...")
    matrix = read_matrix_market(args.input)
    
    if matrix is None:
        return 1
    
    print(f"Matrix loaded: {matrix.shape[0]}×{matrix.shape[1]} with {matrix.nnz:,} non-zeros")
    
    # Print info if requested
    if args.info:
        info = generate_matrix_info(matrix)
        print_matrix_info(info)
    
    # Generate output path if not specified
    output_path = args.output
    if output_path is None and args.output is None:
        # Will display instead of save
        output_path = None
    elif output_path is None:
        # Auto-generate output filename
        input_path = Path(args.input)
        output_path = input_path.with_suffix('.png')
    
    # Plot sparsity pattern
    print("Generating sparsity pattern visualization...")
    plot_sparsity_pattern(
        matrix,
        output_path=output_path,
        title=args.title,
        dpi=args.dpi,
        marker_size=args.size,
        color=args.color,
        background=args.background
    )
    
    return 0


if __name__ == '__main__':
    exit(main())