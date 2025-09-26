#!/usr/bin/bash

set -e

source scripts/variables.sh

cd comparison

if [[ -L "ccutils" ]]; then
  echo "Found symbolic link to ccutils"
else
  ln -s  ccutils
fi

cmake -S . -B build_trilinos \
    -DCMAKE_CXX_COMPILER=CC \
    -DENABLE_TRILINOS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE

cd build_trilinos
make VERBOSE=1 trilinos_spgemm -j16 
