## Compile generator

```bash
make
# or
g++ -std=c++11 -O3 -o multigrid_solver_matrix_generator multigrid_solver_matrix_generator.cpp
```

## Generator Usage Examples

```bash
# Generate a 2D 5-point stencil (128×128)
./multigrid_solver_matrix_generator 2 5pt 128 128 laplace_2d

# Generate prolongation operator (64 coarse → 127 fine points in 2D)
./multigrid_solver_matrix_generator 2 prolongation 64 64 prolong_2d

# Generate restriction operator (opposite direction)
./multigrid_solver_matrix_generator 2 restriction 64 64 restrict_2d

# Generate 3D 27-point stencil (64³)
./multigrid_solver_matrix_generator 3 27pt 64 64 64 stencil_3d_27pt
```

## SpGEMM Operations (examples)

1. $A \times P$: `./multigrid_solver_matrix_generator 2 5pt 127 127` and `./multigrid_solver_matrix_generator 2 prolongation 64 64`
2. $R \times A$: `./multigrid_solver_matrix_generator 2 restriction 64 64` and `./multigrid_solver_matrix_generator 2 5pt 127 127`
3. $R \times A \times P$ (Galerkin): Combine all three

## Visualize Sparsity Patterns

```bash
python sparsity_patters_vizualizer.py <MTX file path>
```