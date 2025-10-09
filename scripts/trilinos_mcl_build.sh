#!/usr/bin/bash

set -e

source scripts/variables.sh

cd comparison

cp -r ../ccutils .

cmake -S . -B build_trilinos \
  -DCMAKE_BUILD_TYPE=RELEASE 

cd build_trilinos
make trilinos_mcl -j16
make trilinos_mcl_dbg -j16
