#pragma once
#include "common.h"
#include "utils.cuh"
#include "cusparse_helpers.cuh"
#include "KokkosWrap.hpp"

#define DEBUG_HOLDER 0

using namespace KokkosWrap;


template <typename IT, typename VT>
struct TileHolder
{

    using LocalMatrix = KokkosWrap::LocalMatrix<IT,IT,VT>;

    TileHolder(const size_t _buf_size, const IT _ptr_size, const IT nnz_size, MPI_Comm _comm, const bool _window=false, MPI_Comm _nodecomm=MPI_COMM_NULL)
    {
        static_assert(sizeof(IT) == sizeof(VT));

        d_buf_size = _buf_size;
        comm = _comm;
        max_nnz  = nnz_size;
        ptr_size = _ptr_size;
        window = _window;

        current_nnz = new uint64_t(0);

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        CUDA_CHECK(cudaMalloc(&d_buf, d_buf_size));
        CUDA_CHECK(cudaMemset(d_buf, 0, d_buf_size));

        if (window)
        {
            MPI_Win_create(d_buf, d_buf_size, sizeof(char), MPI_INFO_NULL, comm, &buf_win);
            MPI_Win_create(current_nnz, sizeof(uint64_t), sizeof(uint64_t), MPI_INFO_NULL, comm, &current_nnz_win);
        }

#ifdef NCCL_ALLGATHERV
        ncclUniqueId id;

        int mpi_rank, mpi_size;
        MPI_Comm_rank(_nodecomm, &mpi_rank);
        MPI_Comm_size(_nodecomm, &mpi_size);

        // Step 1: unique ID created by rank 0
        if (mpi_rank == 0) {
            ncclGetUniqueId(&id);
        }

        // Step 2: send ID to all others using MPI
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, _nodecomm);

        // MPI_ALL_PRINT(
        //     for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
        //         fprintf(fp, "%02x", static_cast<unsigned char>(id.internal[i]));
        //     }
        // )

        // Step 3: create NCCL communicator
        ncclCommInitRank(&ncclNodecomm, mpi_size, id, mpi_rank);
#else
        ncclNodecomm = nullptr;
#endif

    }


    TileHolder(){}


    void set_csx_ptrs()
    {
        set_csx_ptrs(*current_nnz);
    }


    void set_csx_ptrs(const IT nnz)
    {
        assert(sizeof(IT) == sizeof(VT));
        size_t offset = 0;
        d_vals_buf= (VT*)d_buf;
        offset += sizeof(VT) * nnz;
        d_inds_buf= (IT*)(d_buf + offset);
        offset += sizeof(IT) * nnz;
        d_ptrs_buf= (IT*)(d_buf + offset);
    }


    float send_tile_contig(char * d_sendbuf, const IT sendbuf_size, const IT nnz, const int target, cudaStream_t stream = 0, int tag = 0)
    {

#ifdef NVTX_PROFILING
        int nvtx_color;
        char comunication_str[20], nvtx_char;
        if (tag == 1) 
        {
            nvtx_color = 3;
            nvtx_char  = 'B';
        } 
        else 
        {
            nvtx_color = 4;
            nvtx_char  = 'A';
        }
        sprintf(comunication_str, "Send time %c", nvtx_char);
#endif

        float time = 0.0;
        CPU_TIMER_DEF(tmp_timer)

#ifdef NVTX_PROFILING
        NVTX_PUSH_RANGE_CUDA(comunication_str,nvtx_color,stream);
#endif

        CPU_TIMER_START(tmp_timer)

        MPI_Request size_req, main_req;
        IT payload[2] = {nnz, sendbuf_size};
        MPI_Isend(payload, 2, MPIType<IT>(), target, 0, comm, &size_req);

        if (nnz > 0)
        {
            MPI_Isend(d_sendbuf, sendbuf_size, MPI_CHAR, target, 1, comm, &main_req);
            MPI_Wait(&main_req, MPI_STATUS_IGNORE);
        }

        CPU_TIMER_STOP(tmp_timer)

#ifdef NVTX_PROFILING
        NVTX_POP_RANGE;
#endif
        return(__timer_vals_tmp_timer.back());
    }


    IT recv_tile_contig(int src, MPI_Request * recv_req) 
    {
        IT sizes[2];
        MPI_Request size_req;
        MPI_Irecv(sizes, 2, MPIType<IT>(), src, 0, comm, &size_req);
        MPI_Wait(&size_req, MPI_STATUS_IGNORE);

        set_csx_ptrs(sizes[0]);

        if (sizes[0] > 0)
        {
            MPI_Irecv(d_buf, sizes[1], MPI_CHAR, src, 1, comm, recv_req);
        }
        else
        {
            // Have to do this so rowptrs array is valid for a nnz=0 matrix
            CHECK_CUDA(cudaMemsetAsync(d_ptrs_buf, 0, sizeof(IT) * (ptr_size + 1), cudaStreamPerThread));
            CUDA_SYNC(cudaStreamPerThread);
        }

        return(sizes[0]);
    }


    IT copy_device_local_csx(mmio::CSX<IT,VT> *input, cudaStream_t stream = 0) 
    {

        set_csx_ptrs(input->nnz);

#ifdef NVTX_PROFILING
        NVTX_PUSH_RANGE("Fetching local data",2);
#endif

        CUDA_CHECK(cudaMemcpyAsync(d_vals_buf, input->val,     (input->nnz) * sizeof(VT), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_inds_buf, input->idx_vec, (input->nnz) * sizeof(IT), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ptrs_buf, input->ptr_vec, (ptr_size+1) * sizeof(IT), cudaMemcpyDeviceToDevice, stream));

#ifdef NVTX_PROFILING
        NVTX_POP_RANGE;
#endif
        return(input->nnz);
    }


    mmio::CSX<IT, VT> * form_mmiocsx(const IT nrows, const IT ncols, const IT nnz, mmio::MajorDim layout,
                                     VT * d_vals, IT * d_inds, IT * d_ptrs)
    {
        mmio::CSX<IT, VT> *csx = (mmio::CSX<IT, VT>*)malloc(sizeof(mmio::CSX<IT, VT>));
        csx->majordim = layout;
        csx->nrows    = nrows;
        csx->ncols    = ncols;
        csx->nnz      = nnz;
        csx->contig = false;
        csx->buf_size = 0;

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

        assert(buffers != nullptr);
        cudaStream_t * stream = buffers->stream;
        assert(stream != nullptr);

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
        if (buffers == nullptr) 
        {
            CUDA_CHECK(cudaMalloc(d_node_vals,    sizeof(VT) * total_nnz));
            CUDA_CHECK(cudaMalloc(d_node_colinds, sizeof(IT) * total_nnz));
            CUDA_CHECK(cudaMalloc(d_node_rowptrs, sizeof(IT) * (total_nrows + 1)));
        } 
        else 
        {
            buffers->ensure(total_nnz, total_nrows + 1, ncols);
            *d_node_vals    = buffers->d_node_vals;
            *d_node_colinds = buffers->d_node_colinds;
            *d_node_rowptrs = buffers->d_node_rowptrs;
        }


        // Manage the case of singleton with a direct D2D copy
        if (node_size == 1) 
        {
            CUDA_CHECK(cudaMemcpyAsync(*d_node_vals,    d_vals_buf,       nnz * sizeof(VT), cudaMemcpyDeviceToDevice, *stream));
            CUDA_CHECK(cudaMemcpyAsync(*d_node_colinds, d_inds_buf,       nnz * sizeof(IT), cudaMemcpyDeviceToDevice, *stream));
            CUDA_CHECK(cudaMemcpyAsync(*d_node_rowptrs, d_ptrs_buf, (nrows+1) * sizeof(IT), cudaMemcpyDeviceToDevice, *stream));
            CUDA_SYNC(*stream);
            return(nnz);
        }

        // Convert rowtprs to nnz per row
        rowptrs_to_rownnz(d_ptrs_buf, nrows, *stream, &(buffers->tmp_buffers[0]));


        // Allgatherv each buffer
        // Values
#ifndef MPI_ALLGATHERV_OFF
        MPI_Allgatherv(d_vals_buf, nnz, MPIType<VT>(),
                       *d_node_vals, node_nnz.data(), displs.data(),
                       MPIType<VT>(), grid->node_comm);


        // Colinds
        MPI_Allgatherv(d_inds_buf, nnz, MPIType<IT>(),
                       *d_node_colinds, node_nnz.data(), displs.data(),
                       MPIType<IT>(), grid->node_comm);
#else
#ifndef NCCL_ALLGATHERV
        MPI_Request *send_request;
        send_request = (MPI_Request*)malloc(sizeof(MPI_Request)*node_size*2);

        #pragma unroll
        for (int dest=0; dest<node_size; dest++) {
            MPI_Isend(d_vals_buf, nnz, MPIType<VT>(), dest, 0, grid->node_comm, &(send_request[dest]));
            MPI_Isend(d_inds_buf, nnz, MPIType<IT>(), dest, 1, grid->node_comm, &(send_request[node_size + dest]));
        }

        MPI_Request *recv_request;
        recv_request = (MPI_Request*)malloc(sizeof(MPI_Request)*node_size*2);

        #pragma unroll
        for (int src=0; src<node_size; src++) {
            MPI_Irecv((*d_node_vals) + displs[src], node_nnz[src], MPIType<VT>(), src, 0, grid->node_comm, &(recv_request[src]));
            MPI_Irecv((*d_node_colinds) + displs[src], node_nnz[src], MPIType<IT>(), src, 1, grid->node_comm, &(recv_request[node_size + src]));
        }
        MPI_Waitall(2*node_size, recv_request, MPI_STATUS_IGNORE);
        MPI_Waitall(2*node_size, send_request, MPI_STATUS_IGNORE);
        free(send_request);
        free(recv_request);
#else
        ncclGroupStart();
        for (int dest = 0; dest < node_size; dest++) {
            ncclSend(d_vals_buf, nnz, NCCLType<VT>(), dest, ncclNodecomm, *stream);
            ncclSend(d_inds_buf, nnz, NCCLType<IT>(), dest, ncclNodecomm, *stream);
        }
        for (int src = 0; src < node_size; src++) {
            ncclRecv((*d_node_vals)    + displs[src], node_nnz[src], NCCLType<VT>(), src, ncclNodecomm, *stream);
            ncclRecv((*d_node_colinds) + displs[src], node_nnz[src], NCCLType<IT>(), src, ncclNodecomm, *stream);
        }
        ncclGroupEnd();
        cudaStreamSynchronize(*stream);
#endif
#endif

        // Rowptrs
        MPI_Allgather(d_ptrs_buf + 1, nrows, MPIType<IT>(),
                      (*d_node_rowptrs) + 1, nrows, MPIType<IT>(),
                      grid->node_comm);


        // Convert rownnz to rowptrs
        rownnz_to_rowptrs(*d_node_rowptrs, total_nrows, *stream, &(buffers->tmp_buffers[0]));

        return total_nnz;
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



    void aggregate_remote_tile(LocalMatrix& mat, const int target, CsxBuffers<IT, VT> * landing_zone)
    {
        assert(window);

        // To make things easier
        mmio::CSX<IT, VT> * csx = rawptr_get(mat);


        // Get remote tile
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, 0, buf_win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, 0, current_nnz_win);


        // NNZ in remote tile
        uint64_t remote_nnz;
        MPI_Get(&remote_nnz, 1, MPI_UINT64_T, target, 0, 1, MPI_UINT64_T, current_nnz_win);
        MPI_Win_flush_local(target, current_nnz_win);


        // TODO: memory pool?
        // Simple, just put the product onto the tile window without aggregation
        if (remote_nnz == 0)
        {

            if (csx->nnz >= max_nnz)
            {
                std::cerr << RED << csx->nnz << ">" << max_nnz << RESET << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            uint64_t accum_nnz = (uint64_t)csx->nnz;
            MPI_Accumulate(&(accum_nnz), 1, MPI_UINT64_T, target, 0, 1, MPI_UINT64_T, MPI_REPLACE, current_nnz_win);
            put_tile(csx, target);

        }
        else
        {
            // Aggregate locally 
            landing_zone->ensure(remote_nnz, csx->nrows + 1, csx->ncols);
            get_tile(landing_zone, remote_nnz, csx->nrows, target);
            mmio::CSX<IT, VT> * landing_zone_csx = landing_zone->to_mmio_csx();


            LocalMatrix remote_C(landing_zone_csx);
            LocalMatrix::spadd(mat, remote_C);


            mmio::CSX<IT,VT> * agg_csx = rawptr_get(remote_C);

            if (agg_csx->nnz >= max_nnz)
            {
                std::cerr << agg_csx->nnz << ">" << max_nnz << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }


            // Put aggregated remote tile back
            uint64_t agg_nnz = agg_csx->nnz;
            MPI_Accumulate(&(agg_nnz), 1, MPI_UINT64_T, target, 0, 1, MPI_UINT64_T, MPI_REPLACE, current_nnz_win);
            put_tile(agg_csx, target);
        }

        // Cleanup
        MPI_Win_unlock(target, current_nnz_win);
        MPI_Win_unlock(target, buf_win);
    }



    void get_tile(CsxBuffers<IT,VT> * landing_zone, const uint64_t remote_nnz, const IT remote_nrows, const int target)
    {
        assert(window);

        MPI_Get(landing_zone->d_node_vals, remote_nnz * sizeof(VT) / sizeof(int), MPI_INT, target, 0, 
                remote_nnz * sizeof(VT) / sizeof(int), MPI_INT, buf_win);
        MPI_Get(landing_zone->d_node_colinds, remote_nnz * sizeof(IT) / sizeof(int), MPI_INT, target, remote_nnz * sizeof(VT), 
                remote_nnz * sizeof(IT) / sizeof(int), MPI_INT, buf_win);
        MPI_Get(landing_zone->d_node_rowptrs, (remote_nrows + 1) * sizeof(IT) / sizeof(int), MPI_INT, target, remote_nnz * sizeof(VT) + remote_nnz * sizeof(IT), 
                (remote_nrows + 1) * sizeof(IT) / sizeof(int), MPI_INT, buf_win);
        MPI_Win_flush_local(target, buf_win);
    }



    void put_tile(mmio::CSX<IT, VT> * csx, const int target)
    {
        assert(window);
        MPI_Put(csx->val, csx->nnz * sizeof(VT) / sizeof(int), MPI_INT, target, 0, 
                csx->nnz * sizeof(VT) / sizeof(int), MPI_INT, buf_win);
        MPI_Put(csx->idx_vec, csx->nnz * sizeof(IT) / sizeof(int), MPI_INT, target, csx->nnz * sizeof(VT), 
                csx->nnz * sizeof(IT) / sizeof(int), MPI_INT, buf_win);
        MPI_Put(csx->ptr_vec, (csx->nrows + 1) * sizeof(IT) / sizeof(int), MPI_INT, target, csx->nnz * sizeof(VT) + csx->nnz * sizeof(IT), 
                (csx->nrows + 1) * sizeof(IT) / sizeof(int), MPI_INT, buf_win);
        MPI_Win_flush_local(target, buf_win);
    }



    ~TileHolder()
    {
        if (window)
        {
            MPI_Win_free(&buf_win);
            MPI_Win_free(&current_nnz_win);
        }
        CUDA_FREE_SAFE(d_buf);
        delete current_nnz;
    }


    VT * d_vals_buf;
    IT * d_inds_buf, * d_ptrs_buf;
    char * d_buf;

    IT ptr_size;
    IT max_nnz;
    uint64_t * current_nnz;
    size_t d_buf_size;

    MPI_Comm comm;
    ncclComm_t ncclNodecomm;
    int rank;
    int world_rank;

    bool window;
    MPI_Win buf_win;
    MPI_Win current_nnz_win;

};
