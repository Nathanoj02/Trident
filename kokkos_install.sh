#!/usr/bin/bash

export KOKKOS_PREFIX=~/local
export KK_PREFIX=~/local

cd kokkos
# (Recommended) checkout a release tag that matches KokkosKernels later, e.g.:
# git checkout 4.4.00   # example – use the tag you want

# Use kokkos' nvcc_wrapper as the C++ compiler
export NVCC_WRAPPER=$PWD/bin/nvcc_wrapper

cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX="$KOKKOS_PREFIX" \
  -DCMAKE_CXX_COMPILER="$NVCC_WRAPPER" \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF \
  -DKokkos_ENABLE_TESTS=OFF 

cmake --build build -j8
cmake --install build


cd ../kokkos-kernels
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX="$KK_PREFIX" \
  -DCMAKE_CXX_COMPILER="$KOKKOS_PREFIX/bin/nvcc_wrapper" \
  -DKokkos_ROOT="$KOKKOS_PREFIX" \
  -DKokkosKernels_ENABLE_TESTS=OFF \
  -DKokkosKernels_ENABLE_EXAMPLES=OFF \
  -DKokkosKernels_ENABLE_CUDA=ON \
  -DKokkosKernels_INST_DOUBLE=ON \
  -DKokkosKernels_INST_LAYOUTLEFT=ON \
  -DKokkosKernels_INST_EXECSPACE_CUDA=ON \
  -DKokkosKernels_INST_MEMSPACE_CUDAUVMSPACE=OFF \
  -DKokkosKernels_ENABLE_TPL_CUBLAS=OFF \
  -DKokkosKernels_ENABLE_TPL_CUSOLVER=OFF \
  -DKokkosKernels_ENABLE_TPL_CUSPARSE=OFF
\
  # ETI: only build the combos you use
  -DKokkosKernels_INST_DOUBLE=ON \
  -DKokkosKernels_INST_FLOAT=ON \
  -DKokkosKernels_INST_COMPLEX_DOUBLE=OFF \
  -DKokkosKernels_INST_COMPLEX_FLOAT=OFF \
  -DKokkosKernels_INST_ORDINAL_INT=ON \
  -DKokkosKernels_INST_ORDINAL_INT64_T=OFF \
  -DKokkosKernels_INST_OFFSET_SIZE_T=OFF \
  -DKokkosKernels_INST_OFFSET_INT=ON \
  -DKokkosKernels_INST_LAYOUTLEFT=ON \
  -DKokkosKernels_INST_LAYOUTRIGHT=OFF \
  -DKokkosKernels_INST_EXECSPACE_CUDA=ON \
  -DKokkosKernels_INST_EXECSPACE_OPENMP=OFF \
  -DKokkosKernels_INST_EXECSPACE_THREADS=OFF \
  -DKokkosKernels_INST_MEMSPACE_CUDASPACE=ON \
  -DKokkosKernels_INST_MEMSPACE_CUDAUVMSPACE=OFF

cmake --build build -j8
cmake --install build

