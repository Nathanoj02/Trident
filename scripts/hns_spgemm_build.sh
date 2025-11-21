#!/usr/bin/bash

set -e

source scripts/variables.sh

cmake -S . -B build \
    -DCMAKE_CXX_COMPILER=CC


cd build
make -j16 run_spgemm
make -j16 trilinos_spgemm
