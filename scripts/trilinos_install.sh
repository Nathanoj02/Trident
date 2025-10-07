#!/usr/bin/bash

set -e

source scripts/variables.sh

[[ -d Trilinos ]] || git clone https://github.com/trilinos/Trilinos.git
cd Trilinos

cmake -B build \
  -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX="${Trilinos_PREFIX}" \
  -D CMAKE_C_COMPILER="${CMAKE_C_COMPILER}" \
  -D CMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}" \
  -D CMAKE_CUDA_COMPILER="${CUDA_COMPILER}" \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
  -D Kokkos_ARCH_AMPERE80=ON \
  -D Trilinos_SET_GROUP_AND_PERMISSIONS_ON_INSTALL_BASE_DIR="${Trilinos_PREFIX}" \
  -D Trilinos_ENABLE_INSTALL_CMAKE_CONFIG_FILES=ON \
  -D Trilinos_ENABLE_Kokkos=ON \
  -D Trilinos_ENABLE_Fortran=OFF \
  -D Trilinos_ENABLE_TESTS=OFF \
  -D Trilinos_ENABLE_Tpetra=ON \
  -D Trilinos_ENABLE_Teuchos=ON \
  -D Trilinos_ENABLE_FLOAT=ON \
  -D Trilinos_ENABLE_DOUBLE=OFF \
  -D Trilinos_ENABLE_COMPLEX=OFF \
  -D TPL_ENABLE_MPI=ON \
  -D TPL_ENABLE_CUDA=ON \
  -D KOKKOSKERNELS_ENABLE_TPL_CUBLAS=OFF \
  -D KOKKOSKERNELS_ENABLE_TPL_CUSOLVER=OFF \
  -D KOKKOSKERNELS_ENABLE_TPL_CUSPARSE=OFF \
  .

cd build
make -j32 install
