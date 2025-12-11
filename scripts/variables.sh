export Kokkos_PREFIX="${HOME}/.local/lib/trilinos/"
export Kokkos_DIR="${Kokkos_PREFIX}/lib64/cmake/Kokkos"
export KokkosKernels_DIR="${Kokkos_PREFIX}/lib64/cmake/KokkosKernels"

export Trilinos_PREFIX="${HOME}/.local/lib/trilinos"
export Trilinos_DIR="${Trilinos_PREFIX}/lib64/cmake/Trilinos"

export CMAKE_C_COMPILER=cc
export CMAKE_CXX_COMPILER=CC
export MPI_CXX_COMPILER=CC
export CUDA_COMPILER=CC

export CC=$CMAKE_C_COMPILER
export CXX=$CMAKE_CXX_COMPILER

export NVCC_WRAPPER_DEFAULT_COMPILER=$CMAKE_CXX_COMPILER
export NVCC_WRAPPER="${Kokkos_PREFIX}/bin/nvcc_wrapper"

#export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/global/homes/l/lpichett/HnS-SpGEMM/kokkos/build:/global/homes/l/lpichett/HnS-SpGEMM/kokkos-kernels/build

#export COMBBLAS_DIR="${HOME}/Hns-SpGEMM/CombBLAS/CombBLAS/install"
