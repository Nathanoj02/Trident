#!/usr/bin/bash

DEST_PATH="/global/cfs/cdirs/m4646/hns_spgemm_matrices_pico/restrictors"
mkdir -p "$DEST_PATH"

# Build the MTX
cd distributed_mmio
cmake -B build -DDMMIO_TOOLS=ON
cd build
make mtx_to_bmtx
CONVERTER="$(pwd)/mtx_to_bmtx"

cd ../../amg_matrices
make

# HV15R
./multigrid_solver_matrix_generator 1 restriction 2017169 # These are the matrix rows
mv restrict_1D_2017169_n2017169_nnz6051505.mtx "${DEST_PATH}/HV15R_restriction.mtx"
./multigrid_solver_matrix_generator 1 restriction 2017169 -t
mv restrict_1D_2017169_n4034337_nnz6051505_T.mtx "${DEST_PATH}/HV15R_restriction_T.mtx"

# nlpkkt160
./multigrid_solver_matrix_generator 1 restriction 8345600
mv restrict_1D_8345600_n8345600_nnz25036798.mtx "${DEST_PATH}/nlpkkt160_restriction.mtx"
./multigrid_solver_matrix_generator 1 restriction 8345600 -t
mv restrict_1D_8345600_n16691199_nnz25036798_T.mtx "${DEST_PATH}/nlpkkt160_restriction_T.mtx"

# uk-2002
./multigrid_solver_matrix_generator 1 restriction 18520486
mv restrict_1D_18520486_n18520486_nnz55561456.mtx "${DEST_PATH}/uk-2002_restriction.mtx"
./multigrid_solver_matrix_generator 1 restriction 18520486 -t
mv restrict_1D_18520486_n37040971_nnz55561456_T.mtx "${DEST_PATH}/uk-2002_restriction_T.mtx"

cd $DEST_PATH
for file in *.mtx; do
  "$CONVERTER" "$file"
done