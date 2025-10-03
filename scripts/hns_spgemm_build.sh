#!/usr/bin/bash

set -e

source scripts/variables.sh

cmake -S . -B build_hns \
    -DCMAKE_CXX_COMPILER="${NVCC_WRAPPER}" \
    -DKokkos_DIR="${Kokkos_DIR}" \
    -DKokkosKernels_DIR="${KokkosKernels_DIR}"

cd build_hns
make -j16 run_spgemm
