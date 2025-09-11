# Hierarchical and Sparsity-Aware Sparse Matrix-Matrix Multiplication

# Notes:

cmake on Perlmutter:
cmake -B build -S . -DCMAKE_CXX_COMPILER=$(which g++) -DMPI_CXX_COMPILER=$(which mpicxx) -DKokkos_DIR=/global/homes/l/lpichett/kokkos-4.7.00/install/lib64/cmake/Kokkos/ (put your kokkos install path)
