#!/usr/bin/bash

set -e

#source scripts/variables.sh

cd comparison

if [[ -d "ccutils" ]]; then
  echo "Found symbolic link to ccutils"
else
  ln -s  ccutils
fi


cmake -S . -B build_trilinos \
    -DCMAKE_CXX_COMPILER=CC \
    -DENABLE_TRILINOS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DDMMIO_DIR=../distributed_mmio

cd build_trilinos
make VERBOSE=1 trilinos_spgemm -j16 
