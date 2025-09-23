#pragma once
#include "common.h"
#include "KokkosWrap.hpp"




namespace SpaComm
{

template <typename IT, typename VT>
using KWrapDMat = typename KokkosWrap::DistribuitedMatrix<IT, IT, VT>;

template <typename IT, typename VT>
struct SpaCommHandler
{

    SpaCommHandler(KWrapDMat<IT, VT>& dist_A, KWrapDMat<IT, VT>& dist_B)
    {

        grid = dist_A->partitioning->grid;
        assert(grid->row_size == grid->col_size);
        int grid_dim = grid->row_size;
        A_nzc.resize(grid_dim);
        B_nzr.resize(grid_dim);

        


    }


    ~SpaCommHandler()
    {
        std::for_each(A_nzc.begin(), A_nzc.end() [](IT * d_ptr){CUDA_FREE_SAFE(d_ptr);});
        std::for_each(B_nzc.begin(), B_nzc.end() [](IT * d_ptr){CUDA_FREE_SAFE(d_ptr);});
    }

    ProcessGrid * grid;

    // Each entry of this vector is a device buffer that stores the indices of the nonzero columns of A to be sent to each node
    // A_nzc[n] -> indices of columns of A to be sent to node n in my node row
    std::vector<IT *> A_nzc;

    // B_nzr[n] -> indices of rows of B to be sent to node n in my node row
    std::vector<IT *> B_nzr;




};


}
