# Hierarchical and Sparsity-Aware Sparse Matrix-Matrix Multiplication

# Notes:

cmake on Perlmutter:
    (CPU partition) cmake -B build -S . -DCMAKE_CXX_COMPILER=$(which g++) -DMPI_CXX_COMPILER=$(which mpicxx) -DKokkos_DIR=/global/homes/l/lpichett/kokkos-4.7.00/install/lib64/cmake/Kokkos/ (put your kokkos install path)
    (GPU partition) cmake -B build -S . -DCMAKE_CXX_COMPILER=$(which CC) -DMPI_CXX_COMPILER=$(which CC) -DKokkos_DIR=/global/homes/l/lpichett/kokkos-4.7.00/install/lib64/cmake/Kokkos/ (put your kokkos install path)
