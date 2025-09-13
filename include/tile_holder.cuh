#pragma once
#include "common.h"
#include "kokkos_helpers.cuh"


template <typename IT, typename VT>
struct TileHolder
{

    using LocalCSR = KokkosTypes<IT, VT>::CrsMatrix;

    TileHolder(const IT nrows, const IT nnz_size, MPI_Comm _comm)
    {
        comm = _comm;
        flag = new IT(-1);

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        CUDA_CHECK(cudaMalloc(&d_rowptrs_buf, sizeof(IT) * (nrows + 1)));
        CUDA_CHECK(cudaMalloc(&d_vals_buf, sizeof(VT) * nnz_size));
        CUDA_CHECK(cudaMalloc(&d_colinds_buf, sizeof(IT) * nnz_size));

        MPI_Win_create(d_rowptrs_buf, sizeof(IT) * (nrows + 1), sizeof(IT), MPI_INFO_NULL, comm, &d_rowptrs_win);
        MPI_Win_create(d_colinds_buf, sizeof(IT) * (nnz_size), sizeof(IT), MPI_INFO_NULL, comm, &d_colinds_win);
        MPI_Win_create(d_vals_buf, sizeof(VT) * (nnz_size), sizeof(VT), MPI_INFO_NULL, comm, &d_vals_win);
        MPI_Win_create((void*)flag, sizeof(IT), sizeof(IT), MPI_INFO_NULL, comm, &flag_win);

        MPI_Win_lock_all(0, flag_win);

    }



    void put_tile(VT * d_vals, IT * d_colinds, IT * d_rowptrs, const IT nnz, const IT nrows, const int target)
    {
        // MPI_MODE_NOCHECK should be okay because only one request is out at any given time 
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, MPI_MODE_NOCHECK, d_rowptrs_win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, MPI_MODE_NOCHECK, d_colinds_win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, MPI_MODE_NOCHECK, d_vals_win);


        MPI_Put(d_vals, nnz, MPIType<VT>(), target, 0, nnz, MPIType<VT>(), d_vals_win);
        MPI_Put(d_colinds, nnz, MPIType<IT>(), target, 0, nnz, MPIType<IT>(), d_colinds_win);
        MPI_Put(d_rowptrs, nrows + 1, MPIType<IT>(), target, 0, nrows + 1, MPIType<IT>(), d_rowptrs_win);


        MPI_Win_flush(target, d_rowptrs_win);
        MPI_Win_flush(target, d_colinds_win);
        MPI_Win_flush(target, d_vals_win);

        MPI_Win_unlock(target, d_rowptrs_win);
        MPI_Win_unlock(target, d_colinds_win);
        MPI_Win_unlock(target, d_vals_win);


        // Notify target of completion
        MPI_Accumulate(&nnz, 1, MPIType<IT>(), target, 0, 1, MPIType<IT>(), MPI_REPLACE, flag_win);
        MPI_Win_flush(target, flag_win); //TODO: Do I need this?


    }


    IT wait()
    {
        IT nnz;
        do {
            MPI_Win_flush_all(flag_win); //TODO: Get rid of this?
            MPI_Win_sync(flag_win);
            nnz = *flag;
            std::cout<<"Rank: "<<world_rank<<","<<nnz<<std::endl;
            sleep(1);
        } while (nnz == -1);
        *flag = -1;

        MPI_Win_sync(flag_win);
        return nnz;
    }



    LocalCSR * form_tile(const IT nrows, const IT ncols, const IT nnz)
    {
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, MPI_MODE_NOCHECK, d_rowptrs_win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, MPI_MODE_NOCHECK, d_colinds_win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, MPI_MODE_NOCHECK, d_vals_win);
        MPI_Win_sync(d_vals_win);
        MPI_Win_sync(d_colinds_win);
        MPI_Win_sync(d_rowptrs_win);
        MPI_Win_unlock(rank, d_rowptrs_win);
        MPI_Win_unlock(rank, d_colinds_win);
        MPI_Win_unlock(rank, d_vals_win);
        return csr_to_kokkos_crs(nrows, ncols, nnz,
                                 d_vals_buf, d_colinds_buf, d_rowptrs_buf);
    }


    LocalCSR * form_tile(const IT nrows, const IT ncols, const IT nnz,
                         VT * d_vals, IT * d_colinds, IT * d_rowptrs)
    {
        return csr_to_kokkos_crs(nrows, ncols, nnz,
                                 d_vals, d_colinds, d_rowptrs);
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


    LocalCSR * node_allgather_tiles(const IT nrows, const IT ncols, const IT nnz, dmmio::ProcessGrid * grid)
    {

        const int node_size = grid->node_size;
        const int total_nrows = node_size * nrows;

        // Convert rowtprs to nnz per row
        rowptrs_to_rownnz(d_rowptrs_buf, nrows);


        // Get nnz per tile
        std::vector<int> node_nnz(node_size);
        MPI_Allgather(&nnz, 1, MPI_INT, node_nnz.data(), 1, MPI_INT, grid->node_comm);


        // Recv buffer setup
        std::vector<int> displs(node_size);
        int total_nnz = (IT)std::reduce(node_nnz.begin(), node_nnz.end(), 0);
        std::exclusive_scan(node_nnz.begin(), node_nnz.end(), displs.begin(), 0);

        VT * d_node_vals;
        IT * d_node_colinds, * d_node_rowptrs;

        CUDA_CHECK(cudaMalloc(&d_node_vals, sizeof(VT) * total_nnz));
        CUDA_CHECK(cudaMalloc(&d_node_colinds, sizeof(IT) * total_nnz));
        CUDA_CHECK(cudaMalloc(&d_node_rowptrs, sizeof(IT) * (total_nrows + 1)));
        

        // Allgatherv each buffer

        // Values
        MPI_Allgatherv(d_vals_buf, nnz, MPIType<VT>(), 
                       d_node_vals, displs.data(), node_nnz.data(), 
                       MPIType<VT>(), grid->node_comm);
         
        // Colinds
        MPI_Allgatherv(d_colinds_buf, nnz, MPIType<IT>(), 
                       d_node_colinds, displs.data(), node_nnz.data(), 
                       MPIType<IT>(), grid->node_comm);
        
        // Rowptrs
        MPI_Allgather(d_rowptrs_buf + 1, nrows, MPIType<IT>(),
                      d_node_rowptrs + 1, nrows, MPIType<IT>(),
                      grid->node_comm);


        // Convert rownnz to rowptrs
        rownnz_to_rowptrs(d_node_rowptrs, total_nrows);


        // Done
        return form_tile(total_nrows, ncols, total_nnz,
                         d_node_vals, d_node_colinds, d_node_rowptrs);

    }



    ~TileHolder()
    {
        MPI_Win_unlock_all(flag_win);
        MPI_Win_free(&d_rowptrs_win);
        MPI_Win_free(&d_colinds_win);
        MPI_Win_free(&d_vals_win);
        MPI_Win_free(&flag_win);
        CUDA_FREE_SAFE(d_rowptrs_buf);
        CUDA_FREE_SAFE(d_vals_buf);
        CUDA_FREE_SAFE(d_colinds_buf);
    }


    MPI_Win d_vals_win, d_colinds_win, d_rowptrs_win;
    VT * d_vals_buf;
    IT * d_colinds_buf, * d_rowptrs_buf;

    volatile IT * flag;
    MPI_Win flag_win;

    MPI_Comm comm;
    int rank;
    int world_rank;

};
