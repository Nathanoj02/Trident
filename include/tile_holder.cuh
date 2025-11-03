#pragma once
#include "common.h"
#include "utils.cuh"

#define DEBUG_HOLDER 0


template <typename IT, typename VT>
struct TileHolder
{

    TileHolder(const IT _buf_size, const IT _ptr_size, const IT nnz_size, MPI_Comm _comm)
    {
        d_buf_size = _buf_size;
        comm = _comm;
        max_nnz  = nnz_size;
        ptr_size = _ptr_size;

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        CUDA_CHECK(cudaMalloc(&d_buf, d_buf_size));
        CUDA_CHECK(cudaMemset(d_buf, 0, d_buf_size));

    }


    TileHolder(){}


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

        MPI_Request size_req, main_req;
        IT payload[2] = {nnz, sendbuf_size};
        MPI_Isend(payload, 2, MPIType<IT>(), target, 0, comm, &size_req);

        if (nnz > 0)
        {
            MPI_Isend(d_sendbuf, sendbuf_size, MPI_CHAR, target, 1, comm, &main_req);
            MPI_Wait(&main_req, MPI_STATUS_IGNORE);
        }

#ifdef NVTX_PROFILING
        NVTX_POP_RANGE;
#endif
        return(time);
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
        CUDA_CHECK(cudaMemset(*d_node_rowptrs, 0, sizeof(IT)));

        // Manage the case of singleton with a direct D2D copy
        if (node_size == 1) 
        {
            CUDA_CHECK(cudaMemcpy(*d_node_vals,    d_vals_buf,       nnz * sizeof(VT), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(*d_node_colinds, d_inds_buf,       nnz * sizeof(IT), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(*d_node_rowptrs, d_ptrs_buf, (nrows+1) * sizeof(IT), cudaMemcpyDeviceToDevice));
            return(nnz);
        }

        // Convert rowtprs to nnz per row
        rowptrs_to_rownnz(d_ptrs_buf, nrows, 0, &(buffers->tmp_buffers[0]));


        // Allgatherv each buffer

        // Values
        MPI_Allgatherv(d_vals_buf, nnz, MPIType<VT>(),
                       *d_node_vals, node_nnz.data(), displs.data(),
                       MPIType<VT>(), grid->node_comm);

        // Colinds
        MPI_Allgatherv(d_inds_buf, nnz, MPIType<IT>(),
                       *d_node_colinds, node_nnz.data(), displs.data(),
                       MPIType<IT>(), grid->node_comm);
        // Rowptrs
        MPI_Allgather(d_ptrs_buf + 1, nrows, MPIType<IT>(),
                      (*d_node_rowptrs) + 1, nrows, MPIType<IT>(),
                      grid->node_comm);


        // Convert rownnz to rowptrs
        rownnz_to_rowptrs(*d_node_rowptrs, total_nrows, 0, &(buffers->tmp_buffers[0]));

        return(total_nnz);
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
        CUDA_FREE_SAFE(d_buf);
    }


    VT * d_vals_buf;
    IT * d_inds_buf, * d_ptrs_buf;
    char * d_buf;

    IT ptr_size;
    IT max_nnz;
    IT d_buf_size;

    MPI_Comm comm;
    int rank;
    int world_rank;

};
