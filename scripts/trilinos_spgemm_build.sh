#!/usr/bin/bash

set -e

source scripts/variables.sh

cd comparison

cmake -S . -B build_trilinos \
    -DCMAKE_CXX_COMPILER="$CMAKE_CXX_COMPILER" \
    -DENABLE_TRILINOS=ON

cd build_trilinos
make VERBOSE=1 trilinos_spgemm -j16 
