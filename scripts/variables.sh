export Kokkos_VERSION=4.7.01
export Kokkos_DOWNLOAD_URL=https://github.com/kokkos/kokkos/releases/download/${Kokkos_VERSION}
export Kokkos_PREFIX=~/.local/lib/kokkos
export Kokkos_DIR="${Kokkos_PREFIX}/lib64/cmake/Kokkos"
export KokkosKernels_DIR="${Kokkos_PREFIX}/lib64/cmake/KokkosKernels"

export Trilinos_PREFIX=~/.local/lib/trilinos
export Trilinos_DIR="${Trilinos_PREFIX}/lib64/cmake/Trilinos"

export NVCC_WRAPPER_DEFAULT_COMPILER=CC
export NVCC_WRAPPER="${PWD}/bin/nvcc_wrapper"

export CMAKE_CXX_COMPILER=$(which g++)
export MPI_CXX_COMPILER=$(which mpicxx)