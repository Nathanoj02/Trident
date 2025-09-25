#!/usr/bin/bash

source scripts/variables.sh

cmake -S . -B build \
    -DCCUTILS_ENABLE_CUDA=ON \
    -DCMAKE_CXX_COMPILER="$KOKKOS_PREFIX/bin/nvcc_wrapper" \
    -DKokkos_ROOT="$KOKKOS_PREFIX" \
    -DKokkosKernels_ROOT="$KK_PREFIX"
