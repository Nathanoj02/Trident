#pragma once
#include "common.h"
#include "kokkos_helpers.cuh"


template <typename IT, typename VT>
struct TileWindow
{


    TileWindow(VT * d_vals, IT * d_colinds, IT * d_rowptrs,
               const IT nnz, const IT nrows, MPI_Comm comm):
        d_vals(d_vals), d_colinds(d_colinds), d_rowptrs(d_rowptrs),
        nnz(nnz), nrows(nrows), comm(comm)
    {

        MPI_Win_create(d_vals, sizeof(VT) * nnz, sizeof(VT), MPI_INFO_NULL, comm, &vals_win);
        MPI_Win_create(d_colinds, sizeof(IT) * nnz, sizeof(IT), MPI_INFO_NULL, comm, &colinds_win);
        MPI_Win_create(d_rowptrs, sizeof(IT) * (nrows + 1), sizeof(IT), MPI_INFO_NULL, comm, &rowptrs_win);

        MPI_Win_lock_all(MPI_MODE_NOCHECK, vals_win);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, colinds_win);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, rowptrs_win);

    }


    void get_tile(mmio::CSR<IT, VT> * csr, const IT nnz, const int target, MPI_Comm comm)
    {

        csr->nnz = nnz;


        MPI_Get(csr->row_ptr, nrows+1, MPIType<IT>(), target, 0, nrows+1, MPIType<IT>(), rowptrs_win);
        MPI_Get(csr->val, nnz, MPIType<VT>(), target, 0, nnz, MPIType<VT>(), vals_win);
        MPI_Get(csr->col_idx, nnz, MPIType<IT>(), target, 0, nnz, MPIType<VT>(), colinds_win);

        MPI_Win_flush(target, vals_win);
        MPI_Win_flush(target, colinds_win);
        MPI_Win_flush(target, rowptrs_win);
    }




    int node_allgather(const IT nrows, const IT ncols, const IT nnz, dmmio::ProcessGrid * grid,
                       mmio::CSR<IT, VT> * csr,
                       VT **d_node_vals, IT **d_node_colinds, IT **d_node_rowptrs)
    {

        const int node_size   = grid->node_size;
        const int total_nrows = node_size * nrows;

        // Convert rowtprs to nnz per row
        rowptrs_to_rownnz(csr->row_ptr, nrows);


        // Get nnz per tile
        std::vector<int> node_nnz(node_size);
        MPI_Allgather(&nnz, 1, MPI_INT, node_nnz.data(), 1, MPI_INT, grid->node_comm);


        // Recv buffer setup
        std::vector<int> displs(node_size);
        int total_nnz = (IT)std::reduce(node_nnz.begin(), node_nnz.end(), 0);
        std::exclusive_scan(node_nnz.begin(), node_nnz.end(), displs.begin(), 0);


        CUDA_CHECK(cudaMalloc(d_node_vals, sizeof(VT) * total_nnz));
        CUDA_CHECK(cudaMalloc(d_node_colinds, sizeof(IT) * total_nnz));
        CUDA_CHECK(cudaMalloc(d_node_rowptrs, sizeof(IT) * (total_nrows + 1)));
        CUDA_CHECK(cudaMemset(*d_node_rowptrs, 0, sizeof(IT)));
        

        // Allgatherv each buffer

        // Values
        MPI_Allgatherv(csr->val, nnz, MPIType<VT>(), 
                       *d_node_vals, node_nnz.data(), displs.data(),
                       MPIType<VT>(), grid->node_comm);
         
        // Colinds
        MPI_Allgatherv(csr->col_idx, nnz, MPIType<IT>(),
                       *d_node_colinds, node_nnz.data(), displs.data(),
                       MPIType<IT>(), grid->node_comm);
        
        // Rowptrs
        MPI_Allgather(csr->row_ptr + 1, nrows, MPIType<IT>(),         
                      (*d_node_rowptrs) + 1, nrows, MPIType<IT>(),  
                      grid->node_comm);


        // Convert rownnz to rowptrs
        rownnz_to_rowptrs(*d_node_rowptrs, total_nrows);

        return(total_nnz);
    }


    mmio::CSX<IT, VT> * form_mmiocsx(const IT nrows, const IT ncols, const IT nnz, mmio::MajorDim layout,
                                     VT * d_vals, IT * d_inds, IT * d_ptrs)
    {
        mmio::CSX<IT, VT> *csx = (mmio::CSX<IT, VT>*)malloc(sizeof(mmio::CSX<IT, VT>));
        csx->majordim = layout;
        csx->nrows    = nrows;
        csx->ncols    = ncols;
        csx->nnz      = nnz;

        csx->ptr_vec = d_ptrs;
        csx->idx_vec = d_inds;
        csx->val     = d_vals;
        return(csx);
    }


    mmio::CSX<IT, VT> * node_allgather_mmiocsr(mmio::CSR<IT, VT> * csr, const IT nrows, const IT ncols, const IT nnz, dmmio::ProcessGrid * grid) 
    {
        const int total_nrows = (grid->node_size) * nrows;

        VT * d_node_vals;
        IT * d_node_colinds, * d_node_rowptrs;
        int total_nnz = node_allgather(nrows, ncols, nnz, grid, csr, &d_node_vals, &d_node_colinds, &d_node_rowptrs);

        // Done
        return form_mmiocsx(total_nrows, ncols, total_nnz, mmio::MajorDim::ROWS,
                            d_node_vals, d_node_colinds, d_node_rowptrs);
    }


    template <typename T>
    struct DiffOp2
    {
        DiffOp2(){}
        __host__ __device__ __forceinline__
        T operator()(const T& lhs, const T& rhs)
        {
            return lhs - rhs;
        }
    };


    void rownnz_to_rowptrs(IT * d_rowptrs, const IT nrows)
    {
        void * d_tmp = NULL;
        size_t tmp_size = 0;
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_tmp, tmp_size, d_rowptrs+1, nrows));
        CUDA_CHECK(cudaMalloc(&d_tmp, tmp_size));
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_tmp, tmp_size, d_rowptrs+1, nrows));
        CUDA_FREE_SAFE(d_tmp);
        CUDA_CHECK(cudaDeviceSynchronize());
    }


    void rowptrs_to_rownnz(IT * d_rowptrs, const IT nrows)
    {
        void * d_tmp = NULL;
        size_t tmp_size = 0;
        CUDA_CHECK(cub::DeviceAdjacentDifference::SubtractLeft(d_tmp, tmp_size, d_rowptrs, nrows+1, DiffOp2<IT>{}));
        CUDA_CHECK(cudaMalloc(&d_tmp, tmp_size));
        CUDA_CHECK(cub::DeviceAdjacentDifference::SubtractLeft(d_tmp, tmp_size, d_rowptrs, nrows+1, DiffOp2<IT>{}));
        CUDA_FREE_SAFE(d_tmp);
        CUDA_CHECK(cudaDeviceSynchronize());
    }



    ~TileWindow()
    {
        MPI_Win_unlock_all(vals_win);
        MPI_Win_unlock_all(colinds_win);
        MPI_Win_unlock_all(rowptrs_win);


        MPI_Win_free(&vals_win);
        MPI_Win_free(&colinds_win);
        MPI_Win_free(&rowptrs_win);
    }


    IT nnz, nrows;
    VT * d_vals; // I do not own these
    IT * d_colinds, * d_rowptrs; // Or these 

    MPI_Win vals_win, colinds_win, rowptrs_win;

    MPI_Comm comm;
};
