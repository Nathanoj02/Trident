#!/usr/bin/bash

set -e

source scripts/variables.sh

if [[ ! -d kokkos ]]; then
  wget ${Kokkos_DOWNLOAD_URL}/kokkos-${Kokkos_VERSION}.tar.gz
  tar -xzvf kokkos-${Kokkos_VERSION}.tar.gz
  mv kokkos-${Kokkos_VERSION} kokkos
  rm kokkos-${Kokkos_VERSION}.tar.gz
fi
cd kokkos

cmake -S . -B build \
  -DCMAKE_C_COMPILER="$CMAKE_C_COMPILER" \
  -DCMAKE_CXX_COMPILER="$CMAKE_CXX_COMPILER" \
  -DCMAKE_CUDA_COMPILER="${CUDA_COMPILER}" \
  -DCMAKE_INSTALL_PREFIX="$Kokkos_PREFIX" \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ARCH_AMPERE80=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF \
  -DKokkos_ENABLE_TESTS=OFF 

cmake --build build -j8
cmake --install build


cd ..
if [[ ! -d kokkos-kernels ]]; then
  wget https://github.com/kokkos/kokkos-kernels/releases/download/4.7.01/kokkos-kernels-4.7.01.tar.gz
  tar -xzvf kokkos-kernels-4.7.01.tar.gz
  mv kokkos-kernels-4.7.01 kokkos-kernels
  rm kokkos-kernels-4.7.01.tar.gz
fi
cd kokkos-kernels

cmake -S . -B build \
  -DCMAKE_C_COMPILER="$CMAKE_C_COMPILER" \
  -DCMAKE_CXX_COMPILER="$CMAKE_CXX_COMPILER" \
  -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
  -DCMAKE_INSTALL_PREFIX="$Kokkos_PREFIX" \
  -DKokkos_ROOT="$Kokkos_PREFIX" \
  -DKokkosKernels_ENABLE_TESTS=OFF \
  -DKokkosKernels_ENABLE_EXAMPLES=OFF \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkosKernels_ENABLE_CUDA=ON \
  -DKokkosKernels_INST_DOUBLE=OFF \
  -DKokkosKernels_INST_FLOAT=ON \
  -DKokkosKernels_INST_LAYOUTLEFT=ON \
  -DKokkosKernels_INST_EXECSPACE_CUDA=ON \
  -DKokkosKernels_INST_MEMSPACE_CUDAUVMSPACE=OFF \
  -DKokkosKernels_ENABLE_TPL_CUBLAS=OFF \
  -DKokkosKernels_ENABLE_TPL_CUSOLVER=OFF \
  -DKokkosKernels_ENABLE_TPL_CUSPARSE=OFF

cmake --build build -j8
cmake --install build
