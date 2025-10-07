# Hierarchical and Sparsity-Aware Sparse Matrix-Matrix Multiplication

## Modules

- **Perlmutter**: `cmake/3.30.2 cudatoolkit/12.9 craype-accel-nvidia80 craype-hugepages2G cray-pmi/6.1.15`
- **Leonardo**: `cmake/3.27.9 gcc/12.2.0 cuda/12.2 openmpi/4.1.6--gcc--12.2.0-cuda-12.2 openblas/0.3.26--gcc--12.2.0`

# HnS-SpGEMM

First, install Kokkos:
```bash
scripts/kokkos_install.sh
```

# Baselines

Installation paths are customizable in [`scripts/variables.sh`](scripts/variables.sh).

## Trilinos

To build `Trilinos` as a shared library, run:
```bash
scripts/trilinos_install.sh
```

Notes:
- Currently the script builds for GPU architecture `AMPERE80`

MCL PUT + CSC