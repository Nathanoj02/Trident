export Kokkos_PREFIX="${HOME}/.local/lib/trilinos/"
export Kokkos_DIR="${Kokkos_PREFIX}/lib64/cmake/Kokkos"
export KokkosKernels_DIR="${Kokkos_PREFIX}/lib64/cmake/KokkosKernels"

export Trilinos_PREFIX="${HOME}/.local/lib/trilinos"
export Trilinos_DIR="${Trilinos_PREFIX}/lib64/cmake/Trilinos"


export CMAKE_C_COMPILER=gcc
export CMAKE_CXX_COMPILER=g++
export MPI_CXX_COMPILER=mpicxx
export CUDA_COMPILER=nvcc

export Kokkos_VERSION=4.7.01
export Kokkos_DOWNLOAD_URL="https://github.com/kokkos/kokkos/releases/download/${Kokkos_VERSION}"

export CC=$CMAKE_C_COMPILER
export CXX=$CMAKE_CXX_COMPILER

export NVCC_WRAPPER_DEFAULT_COMPILER=$CMAKE_CXX_COMPILER
export NVCC_WRAPPER="${Kokkos_PREFIX}/bin/nvcc_wrapper"

#export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/global/homes/l/lpichett/HnS-SpGEMM/kokkos/build:/global/homes/l/lpichett/HnS-SpGEMM/kokkos-kernels/build

export COMBBLAS_DIR="${HOME}/CombBLAS-GPU/CombBLAS/install"
