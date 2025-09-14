#pragma once
#include "common.h"
#include "kokkos_helpers.cuh"


template <typename IT, typename VT>
struct TileHolder
{

    TileHolder(const IT nrows, const IT nnz_size, MPI_Comm _comm)
    {
        comm = _comm;
        flag = new IT(0);

        MPI_Comm_rank(comm, &rank);

        CUDA_CHECK(cudaMalloc(&d_rowptrs_buf, sizeof(IT) * (nrows + 1)));
        CUDA_CHECK(cudaMalloc(&d_vals_buf, sizeof(VT) * nnz_size));
        CUDA_CHECK(cudaMalloc(&d_colinds_buf, sizeof(IT) * nnz_size));

        MPI_Win_create(d_rowptrs_buf, sizeof(IT) * (nrows + 1), sizeof(IT), MPI_INFO_NULL, comm, &d_rowptrs_win);
        MPI_Win_create(d_colinds_buf, sizeof(IT) * (nnz_size), sizeof(IT), MPI_INFO_NULL, comm, &d_colinds_win);
        MPI_Win_create(d_vals_buf, sizeof(VT) * (nnz_size), sizeof(VT), MPI_INFO_NULL, comm, &d_vals_win);
        MPI_Win_create((void*)flag, sizeof(IT), sizeof(IT), MPI_INFO_NULL, comm, &flag_win);

    }



    void put_tile(VT * d_vals, IT * d_colinds, IT * d_rowptrs, const IT nnz, const IT nrows, const int target)
    {
        // MPI_MODE_NOCHECK should be okay because only one request is out at any given time 
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, MPI_MODE_NOCHECK, d_rowptrs_win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, MPI_MODE_NOCHECK, d_colinds_win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, MPI_MODE_NOCHECK, d_vals_win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, MPI_MODE_NOCHECK, flag_win);


        MPI_Put(d_vals, nnz, MPIType<VT>(), target, 0, nnz, MPIType<VT>(), d_vals_win);
        MPI_Put(d_colinds, nnz, MPIType<IT>(), target, 0, nnz, MPIType<IT>(), d_colinds_win);
        MPI_Put(d_rowptrs, nrows + 1, MPIType<IT>(), target, 0, nrows + 1, MPIType<IT>(), d_rowptrs_win);


        MPI_Win_flush(target, d_rowptrs_win);
        MPI_Win_flush(target, d_colinds_win);
        MPI_Win_flush(target, d_vals_win);

        // Notify target of completion
        MPI_Accumulate(&nnz, 1, MPIType<IT>(), target, 0, 1, MPIType<IT>(), MPI_REPLACE, flag_win);
        MPI_Win_flush(target, flag_win); //TODO: Do I need this?

        MPI_Win_unlock(target, d_rowptrs_win);
        MPI_Win_unlock(target, d_colinds_win);
        MPI_Win_unlock(target, d_vals_win);
        MPI_Win_unlock(target, flag_win);

    }


    IT wait()
    {
        IT nnz;
        do {
            MPI_Win_sync(flag_win);
            nnz = *flag;
        } while (nnz == -1);
        *flag = -1;
        MPI_Win_sync(flag_win);
        return nnz;
    }

    //TODO: create kokkos CRS instance from local pointers
    //make sure to call MPI_Win_sync first

    using KokkosCRS = typename KokkosTypes<IT, VT>::CrsMatrix;
    KokkosCRS* form_tile(const IT nrows, const IT ncols, const IT nnz)
    {
        MPI_Win_sync(d_vals_win);
        MPI_Win_sync(d_colinds_win);
        MPI_Win_sync(d_rowptrs_win);
        return csr_to_kokkos_crs(nrows, ncols, nnz,
                                 d_vals_buf, d_colinds_buf, d_rowptrs_buf);
    }



    ~TileHolder()
    {
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

};
