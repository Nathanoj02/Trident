#pragma once 
#include "common.h"
#include "tile_holder.cuh"
#include "cusparse_helpers.cuh"
#include "KokkosWrap.hpp"

#include <ccutils/cuda/cuda_timers.h>

template <typename IT>
__global__ void filter_colinds(IT * d_colinds, const IT ncols, const IT nnz, bool * d_nnz_colinds)
{
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= nnz) return;
    
    IT colidx = d_colinds[tid];
    d_nnz_colinds[colidx] = 1;
}


template <typename IT, typename VT>
struct Hns1DHandler
{
    int num_nodes;
    bool ** row_filters;
    DistCusparseCSX<IT, VT> * A;
    DistCusparseCSX<IT, VT> * B;
    MPI_Comm node_comm;
    static const int node_size = 4;
    dmmio::ProcessGrid * grid;


    Hns1DHandler(DistCusparseCSX<IT, VT> * A, DistCusparseCSX<IT, VT> * B):
        A(A), B(B), grid(A->partitioning->grid)
    {
        assert(grid->row_size == 1 && grid->col_size == 1);

        int np = grid->global_size;
        num_nodes = np / node_size;
        row_filters = (bool **)malloc(sizeof(bool *) * num_nodes);
        IT B_rows = B->getLocalNrows();

        for (int i=0; i<num_nodes; i++)
        {
            CUDA_CHECK(cudaMalloc(&(row_filters[i]), sizeof(bool) * B_rows));
        }


        MPI_Comm node_comm;
        int color = grid->global_rank / node_size;
        int key = grid->global_rank % node_size;
        MPI_Comm_split(MPI_COMM_WORLD, color, key, &node_comm);

    }


    void setup_row_filters()
    {
        mmio::CSX<IT, VT> * csx = A->csx->mat;
        IT A_cols = A->getLocalNcols();
        IT A_rows = A->getLocalNrows();
        IT B_rows = B->getLocalNrows();
        IT A_nnz = A->getLocalNnz();

        bool * d_nnz_cols;
        CUDA_CHECK(cudaMalloc(&d_nnz_cols, sizeof(bool) * (A->getLocalNcols())));
        CUDA_CHECK(cudaMemset(d_nnz_cols, 0, sizeof(bool) * A->getLocalNcols()));

        uint64_t nthreads = 256;
        uint64_t nblocks = std::ceil( (double)(A_cols) / (double)nthreads );
        filter_colinds<<<nblocks, nthreads>>>(csx->col_idx, A_cols, A_nnz, d_nnz_cols);
        CUDA_CHECK(cudaDeviceSynchronize());

        //MPI_Reduce_scatter_block(MPI_IN_PLACE, d_nnz_cols, A_cols, MPI_C_BOOL, MPI_LOR, node_comm);
        MPI_Allreduce(MPI_IN_PLACE, d_nnz_cols, A_cols, MPI_C_BOOL, MPI_LOR, node_comm);

        int p;
        MPI_Comm_rank(node_comm, &p);

        size_t offset = sizeof(bool) * p;
        IT send_count = A->getLocalNrows();

        int node_id = grid->global_rank / node_size;
        for (int i=0; i<num_nodes; i++)
        {
            int dest = ((i + node_id) % num_nodes) + p;
            MPI_Sendrecv(d_nnz_cols + offset, send_count, MPI_C_BOOL, dest, 0,
                         row_filters[i], send_count, MPI_C_BOOL, dest, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        CUDA_FREE_SAFE(d_nnz_cols);

    }




    ~Hns1DHandler()
    {
        for (int i=0; i<num_nodes; i++)
        {
            CUDA_FREE_SAFE(row_filters[i]);
        }
        free(row_filters);
    }
};





template <typename IT, typename VT>
void hns_spgemm_1d(DistCusparseCSX<IT, VT> * A, DistCusparseCSX<IT, VT> * B);

