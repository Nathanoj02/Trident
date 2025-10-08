#pragma once
#include "common.h"
#include "utils.cuh"
#include "kokkos_helpers.cuh"

#define DEBUG_HOLDER 0

//#define PTR_CHECK

#ifdef PTR_CHECK
extern int here_iteration;
#endif

template <typename IT, typename VT>
struct TileHolder
{

    using LocalCSR = typename KokkosTypes<IT, VT>::CrsMatrix;

    TileHolder(const IT _ptr_size, const IT nnz_size, MPI_Comm _comm)
    {
        comm = _comm;
        flag = new IT(-1);
        ptr_size = _ptr_size;

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        CUDA_CHECK(cudaMalloc(&d_ptrs_buf, sizeof(IT) * (_ptr_size + 1)));
        CUDA_CHECK(cudaMalloc(&d_vals_buf, sizeof(VT) * nnz_size));
        CUDA_CHECK(cudaMalloc(&d_inds_buf, sizeof(IT) * nnz_size));

        MPI_Win_create(d_ptrs_buf, sizeof(IT) * (_ptr_size + 1), sizeof(IT), MPI_INFO_NULL, comm, &d_ptrs_win);
        MPI_Win_create(d_inds_buf, sizeof(IT) * (nnz_size), sizeof(IT), MPI_INFO_NULL, comm, &d_inds_win);
        MPI_Win_create(d_vals_buf, sizeof(VT) * (nnz_size), sizeof(VT), MPI_INFO_NULL, comm, &d_vals_win);
        MPI_Win_create((void*)flag, sizeof(IT), sizeof(IT), MPI_INFO_NULL, comm, &flag_win);

        MPI_Win_lock_all(MPI_MODE_NOCHECK, d_ptrs_win);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, d_inds_win);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, d_vals_win);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, flag_win);

    }


    TileHolder(){}



    void put_tile(VT * d_vals, IT * d_inds, IT * d_ptrs, const IT nnz, const IT ptr_size, const int target)
    {

#ifdef PTR_CHECK
        CHECK_PTR(d_vals, here_iteration)
        CHECK_PTR(d_inds, here_iteration)
        CHECK_PTR(d_ptrs, here_iteration)
        here_iteration++;
#endif

        // MPI_Put complains about an invalid datatype if I pass it MPIType<VT>()
        MPI_Put(d_vals, nnz, MPI_FLOAT, target, 0, nnz, MPI_FLOAT, d_vals_win);
        MPI_Put(d_inds, nnz, MPI_INT32_T, target, 0, nnz, MPI_INT32_T, d_inds_win);
        MPI_Put(d_ptrs, ptr_size + 1, MPI_INT32_T, target, 0, ptr_size + 1, MPI_INT32_T, d_ptrs_win);


        MPI_Win_flush(target, d_ptrs_win);
        MPI_Win_flush(target, d_inds_win);
        MPI_Win_flush(target, d_vals_win);

        // Notify target of completion
        MPI_Accumulate(&nnz, 1, MPI_INT32_T, target, 0, 1, MPI_INT32_T, MPI_REPLACE, flag_win);
        MPI_Win_flush(target, flag_win); //TODO: Do I need this?
        //MPI_Win_flush_all(flag_win); //TODO: Do I need this?

    }


    IT wait(int src)
    {
        IT nnz;
        do 
        {
            MPI_Win_flush_all(flag_win); //TODO: Get rid of this? I don't understand why it needs to be here, but it seems necessary to update flag
            MPI_Win_sync(flag_win);
            nnz = *flag;
#if DEBUG_HOLDER
            std::cout<<"Rank: "<<world_rank<<","<<nnz<<std::endl;
            sleep(1);
#endif
        } while (nnz == -1);
        *flag = -1;

        MPI_Win_sync(flag_win);
        return nnz;
    }




    void sync_buffers()
    {
        MPI_Win_sync(d_vals_win);
        MPI_Win_sync(d_inds_win);
        MPI_Win_sync(d_ptrs_win);
    }


    LocalCSR * form_tile(const IT nrows, const IT ncols, const IT nnz)
    {
        sync_buffers();
        return csr_to_kokkos_crs(nrows, ncols, nnz,
                                 d_vals_buf, d_inds_buf, d_ptrs_buf);
    }

    LocalCSR * form_tile(const IT nrows, const IT ncols, const IT nnz,
                         VT * d_vals, IT * d_colinds, IT * d_rowptrs)
    {
        return csr_to_kokkos_crs(nrows, ncols, nnz,
                                 d_vals, d_colinds, d_rowptrs);
    }

    mmio::CSX<IT, VT> * form_mmiocsx(const IT nrows, const IT ncols, const IT nnz, mmio::MajorDim layout,
                                     VT * d_vals, IT * d_inds, IT * d_ptrs)
    {
        mmio::CSX<IT, VT> *csx = (mmio::CSX<IT, VT>*)malloc(sizeof(mmio::CSX<IT, VT>));
        csx->majordim = layout;
        csx->nrows    = nrows;
        csx->ncols    = ncols;
        csx->nnz      = nnz;

        sync_buffers();
        csx->ptr_vec = d_ptrs;
        csx->idx_vec = d_inds;
        csx->val     = d_vals;
        return(csx);
    }

    mmio::CSX<IT, VT> * form_mmiocsx(const IT nrows, const IT ncols, const IT nnz, mmio::MajorDim layout)
    {
        return(form_mmiocsx(nrows, ncols, nnz, layout, d_vals_buf, d_inds_buf, d_ptrs_buf));
    }


    int node_allgather(const IT nrows, const IT ncols, const IT nnz, dmmio::ProcessGrid * grid,
                       VT **d_node_vals, IT **d_node_colinds, IT **d_node_rowptrs, CsxBuffers<IT,VT>* buffers=nullptr)
    {

        sync_buffers();

        const int node_size   = grid->node_size;
        const int total_nrows = node_size * nrows;

        // Get nnz per tile
        std::vector<int> node_nnz(node_size);
        MPI_Allgather(&nnz, 1, MPI_INT, node_nnz.data(), 1, MPI_INT, grid->node_comm);


        // Recv buffer setup
        std::vector<int> displs(node_size);
        int total_nnz = (IT)std::reduce(node_nnz.begin(), node_nnz.end(), 0);
        std::exclusive_scan(node_nnz.begin(), node_nnz.end(), displs.begin(), 0);

        // Buffer set-up
        if (buffers == nullptr) {
            CUDA_CHECK(cudaMalloc(d_node_vals,    sizeof(VT) * total_nnz));
            CUDA_CHECK(cudaMalloc(d_node_colinds, sizeof(IT) * total_nnz));
            CUDA_CHECK(cudaMalloc(d_node_rowptrs, sizeof(IT) * (total_nrows + 1)));
        } else {
            buffers->ensure(total_nnz, total_nrows + 1);
            *d_node_vals    = buffers->d_node_vals;
            *d_node_colinds = buffers->d_node_colinds;
            *d_node_rowptrs = buffers->d_node_rowptrs;
        }
        CUDA_CHECK(cudaMemset(*d_node_rowptrs, 0, sizeof(IT)));

        // Manage the case of singleton with a direct D2D copy
        if (node_size == 1) {
            CUDA_CHECK(cudaMemcpy(*d_node_vals,    d_vals_buf,       nnz * sizeof(VT), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(*d_node_colinds, d_inds_buf,       nnz * sizeof(IT), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(*d_node_rowptrs, d_ptrs_buf, (nrows+1) * sizeof(IT), cudaMemcpyDeviceToDevice));
            return(nnz);
        }

        // Convert rowtprs to nnz per row
        rowptrs_to_rownnz(d_ptrs_buf, nrows, 0, &(buffers->tmp_buffer));


        // Allgatherv each buffer

#ifndef P2P_ALLGATHERV
        // Values
        MPI_Allgatherv(d_vals_buf, nnz, MPIType<VT>(),
                       *d_node_vals, node_nnz.data(), displs.data(),
                       MPIType<VT>(), grid->node_comm);

        // Colinds
        MPI_Allgatherv(d_inds_buf, nnz, MPIType<IT>(),
                       *d_node_colinds, node_nnz.data(), displs.data(),
                       MPIType<IT>(), grid->node_comm);
#else
        int tag;
        MPI_Status  *statuses = (MPI_Status*) malloc(2*(grid->node_size) * sizeof(MPI_Status));
        MPI_Request *requests = (MPI_Request*)malloc(2*(grid->node_size) * sizeof(MPI_Request));
        for (int i=0; i<grid->node_size; i++) {
            if (i!=(grid->node_rank)) {
                tag = (grid->node_rank) * (grid->node_size) + i;
                MPI_Isend(d_vals_buf, nnz, MPIType<VT>(), i, tag, grid->node_comm, &(requests[i]));

    #ifdef DEBUG_P2P_ALLGATHERV
                fprintf(stdout, "[%d, %d] sent to %d with tag %d\n", grid->global_rank, grid->node_rank, i, tag); fflush(stdout);
    #endif

                tag = i * (grid->node_size) + (grid->node_rank);
                MPI_Irecv(d_node_vals + displs[i], node_nnz[i], MPIType<VT>(), i, tag, grid->node_comm, &(requests[grid->node_size + i]));

    #ifdef DEBUG_P2P_ALLGATHERV
                fprintf(stdout, "[%d, %d] receved from %d with tag %d\n", grid->global_rank, grid->node_rank, i, tag); fflush(stdout);
    #endif
            } else {
                // Local communication performed with a D2D copy
                CUDA_CHECK(cudaMemcpy(d_node_vals + displs[i], d_vals_buf, nnz*sizeof(VT), cudaMemcpyDeviceToDevice));
            }
        }
        MPI_Waitall(2*(grid->node_size), requests, statuses);
        MPI_STATUS_CHECK(2*(grid->node_size), statuses, grid->node_comm)

        for (int i=0; i<grid->node_size; i++) {
            if (i!=(grid->node_rank)) {
                tag = (grid->node_rank) * (grid->node_size) + i;
                MPI_Isend(d_inds_buf, nnz, MPIType<IT>(), i, tag, grid->node_comm, &(requests[i]));

    #ifdef DEBUG_P2P_ALLGATHERV
                fprintf(stdout, "[%d, %d] sent to %d with tag %d\n", grid->global_rank, grid->node_rank, i, tag); fflush(stdout);
    #endif

                tag = i * (grid->node_size) + (grid->node_rank);
                MPI_Irecv(d_node_colinds + displs[i], node_nnz[i], MPIType<IT>(), i, tag, grid->node_comm, &(requests[grid->node_size + i]));

    #ifdef DEBUG_P2P_ALLGATHERV
                fprintf(stdout, "[%d, %d] receved from %d with tag %d\n", grid->global_rank, grid->node_rank, i, tag); fflush(stdout);
    #endif
            } else {
                // Local communication performed with a D2D copy
                CUDA_CHECK(cudaMemcpy(d_node_vals + displs[i], d_vals_buf, nnz*sizeof(VT), cudaMemcpyDeviceToDevice));
            }
        }
        MPI_Waitall(2*(grid->node_size), requests, statuses);
        MPI_STATUS_CHECK(2*(grid->node_size), statuses, grid->node_comm)
        free(requests);
        free(statuses);
#endif
        // Rowptrs
        MPI_Allgather(d_ptrs_buf + 1, nrows, MPIType<IT>(),
                      (*d_node_rowptrs) + 1, nrows, MPIType<IT>(),
                      grid->node_comm);


        // Convert rownnz to rowptrs
        rownnz_to_rowptrs(*d_node_rowptrs, total_nrows, 0, &(buffers->tmp_buffer));

        return(total_nnz);
    }

    LocalCSR * node_allgather_tiles(const IT nrows, const IT ncols, const IT nnz, dmmio::ProcessGrid * grid, CsxBuffers<IT,VT>* buffers=nullptr)
    {
        const int total_nrows = (grid->node_size) * nrows;

        VT * d_node_vals;
        IT * d_node_colinds, * d_node_rowptrs;
        int total_nnz = node_allgather(nrows, ncols, nnz, grid, &d_node_vals, &d_node_colinds, &d_node_rowptrs, buffers);

#ifdef PTR_CHECK
        CHECK_PTR(d_node_vals, here_iteration)
        CHECK_PTR(d_node_colinds, here_iteration)
        CHECK_PTR(d_node_rowptrs, here_iteration)
	here_iteration++;
#endif
        // Done
        return form_tile(total_nrows, ncols, total_nnz,
                         d_node_vals, d_node_colinds, d_node_rowptrs);
    }

    mmio::CSX<IT, VT> * node_allgather_mmiocsx(const IT nrows, const IT ncols, const IT nnz, dmmio::ProcessGrid * grid, CsxBuffers<IT,VT>* buffers=nullptr)
    {
        const int total_nrows = (grid->node_size) * nrows;

        VT * d_node_vals;
        IT * d_node_colinds, * d_node_rowptrs;
        int total_nnz = node_allgather(nrows, ncols, nnz, grid, &d_node_vals, &d_node_colinds, &d_node_rowptrs, buffers);

#ifdef PTR_CHECK
        CHECK_PTR(d_node_vals, here_iteration)
        CHECK_PTR(d_node_colinds, here_iteration)
        CHECK_PTR(d_node_rowptrs, here_iteration)
        here_iteration++;
#endif
        // Done
        return form_mmiocsx(total_nrows, ncols, total_nnz, mmio::MajorDim::ROWS,
                            d_node_vals, d_node_colinds, d_node_rowptrs);
    }


    ~TileHolder()
    {
        MPI_Win_unlock_all(d_ptrs_win);
        MPI_Win_unlock_all(d_vals_win);
        MPI_Win_unlock_all(d_inds_win);
        MPI_Win_unlock_all(flag_win);
        MPI_Win_free(&d_ptrs_win);
        MPI_Win_free(&d_inds_win);
        MPI_Win_free(&d_vals_win);
        MPI_Win_free(&flag_win);
        CUDA_FREE_SAFE(d_ptrs_buf);
        CUDA_FREE_SAFE(d_vals_buf);
        CUDA_FREE_SAFE(d_inds_buf);
    }


    MPI_Win d_vals_win, d_inds_win, d_ptrs_win;
    VT * d_vals_buf;
    IT * d_inds_buf, * d_ptrs_buf;

    volatile IT * flag;
    MPI_Win flag_win;

    IT ptr_size;

    MPI_Comm comm;
    int rank;
    int world_rank;

};
