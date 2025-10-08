#!/usr/bin/bash

set -e

source scripts/variables.sh

cd comparison

if [[ -d "ccutils" ]]; then
  echo "Found symbolic link to ccutils"
else
  ln -s  ccutils
fi

cmake -S . -B build_trilinos \
  -DCMAKE_BUILD_TYPE=RELEASE 

cd build_trilinos
make trilinos_mcl -j16
make trilinos_mcl_dbg -j16
