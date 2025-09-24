# Hierarchical and Sparsity-Aware Sparse Matrix-Matrix Multiplication

# Notes

cmake on Perlmutter:
```
(CPU partition) cmake -B build -S . -DCMAKE_CXX_COMPILER=$(which g++) -DMPI_CXX_COMPILER=$(which mpicxx) -DKokkos_DIR=/global/homes/l/lpichett/kokkos-4.7.00/install/lib64/cmake/Kokkos/ (put your kokkos install path)
(GPU partition) cmake -B build -S . -DCMAKE_CXX_COMPILER=$(which CC) -DMPI_CXX_COMPILER=$(which CC) -DKokkos_DIR=/global/homes/l/lpichett/kokkos-4.7.00/install/lib64/cmake/Kokkos/ (put your kokkos install path)
```

## Modules

<!-- - **Baldo**: `GCC/12.3.0 CUDA/12.3.2 OpenMpi/4.1.5-CUDA-12.3.2 OpenBLAS/0.3.23-GCC-12.3.0` -->
- **Perlmutter**: `cmake/3.30.2 cudatoolkit/12.9 craype-hugepages2G cray-pmi/6.1.15`

# Baselines

Installation paths are customizable in [`scripts/variables.sh`](scripts/variables.sh).

## Trilinos

To build `Trilinos` as a shared library, run:
```bash
scripts/trilinos_install.sh
```

Notes:
- Currently the script builds for GPU architecture `AMPERE80`

