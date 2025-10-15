#include "hns_spgemm.cuh"

MPIDataTypeCache mpidtc; //fix linker error

//#ifdef PTR_CHECK
int here_iteration = 0;
//#endif

// #define DEBUG_THREAD_COMPRESSION

// Barrier for sync afther thread allocs
#include <condition_variable>
SimpleBarrier alloc_sync_point(3);
SimpleBarrier free_sync_point(3);

template <typename IT, typename VT>
inline uint64_t compute_message_size(int nnz, int ptr_size) {
        return( (nnz * sizeof(VT)) + ((nnz + ptr_size) * sizeof(IT)) );
}

template <typename IT, typename VT>
void comm_thread_loop_csx(MessageQueue<int>& queue, TileHolder<IT, VT>& holder, const mmio::CSX<IT, VT> * csx, const Implementation impl, SpaComm::SpaCommHandler<IT,VT>* spacomm, int dev_id, int comm_rank, std::mutex& mpi_mutex, int tag=0)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cudaStream_t stream;
    CUDA_CHECK(cudaSetDevice(dev_id)); // To be sure each thread on the same process is assigned to the same GPU
    CUDA_CHECK(cudaStreamCreate(&stream));

    SpaComm::SpaCommBuffers<IT,VT> *compression_buffers = new SpaComm::SpaCommBuffers<IT,VT>(csx);
    /*
    if (spacomm != nullptr)
        holder.warmup(impl, compression_buffers->compressed_values, compression_buffers->compressed_indices, compression_buffers->compressed_pointers);
    else
        holder.warmup(impl, csx->val, csx->idx_vec, csx->ptr_vec);
    */

    float internode_comm;
    CUDA_TIMER_DEF(compression_time)
#ifdef NVTX_PROFILING
    int nvtx_color;
    char compression_str[20], comunication_str[20], nvtx_char;
    if (tag == 1) {
            nvtx_color = 3;
            nvtx_char  = 'B';
    } else {
            nvtx_color = 4;
            nvtx_char  = 'A';
    }
    sprintf(compression_str,  "Compression %c", nvtx_char);
#endif

    IT ptrsize = (csx->majordim == mmio::MajorDim::ROWS) ? (csx->nrows) : (csx->ncols) ;
    alloc_sync_point.arrive_and_wait();

    while (true)
    {

#ifdef NVTX_PROFILING
        NVTX_PUSH_RANGE_CUDA("Waiting_for_work",6,stream);
#endif

        // Wait until someone tells me to send them a tile
        int target = queue.wait();

#ifdef NVTX_PROFILING
        NVTX_POP_RANGE;
#endif

        if (target == comm_rank) continue; // NOTE: self communication managed by main tread

        // Only way this should be able to happen is if I've satisfied all requests, so I can return at this point
        if (target == -2)
        {
            free_sync_point.arrive_and_wait();
            compression_buffers->explicitFree();
            delete compression_buffers;
            return;
        }

#ifdef NVTX_PROFILING
        NVTX_PUSH_RANGE_CUDA(compression_str,nvtx_color,stream);
#endif

        mmio::CSX<IT, VT> *compressed = nullptr;
        if (spacomm != nullptr) {
                CUDA_TIMER_START(compression_time, stream)
                compressed = spacomm->Compress(csx, target, compression_buffers, stream);
                CUDA_TIMER_STOP(compression_time)
                CUDA_CHECK(cudaStreamSynchronize(stream));
        }

#ifdef NVTX_PROFILING
        NVTX_POP_RANGE;
#endif

        // Put tile on remote process
        //   NOTE: timer and NVTX ranges inside the function
        const mmio::CSX<IT, VT> *tosend = (spacomm != nullptr) ? compressed : csx ;
        if (impl == Implementation::PUT) {
                internode_comm = holder.put_tile(tosend->val, tosend->idx_vec, tosend->ptr_vec, tosend->nnz, ptrsize, target, mpi_mutex, stream, tag);
        } else {
                internode_comm = holder.send_tile(tosend->val, tosend->idx_vec, tosend->ptr_vec, tosend->nnz, ptrsize, target, mpi_mutex, stream, tag);
        }


        char tmpstr[20];
        char desc = (tag == 0) ? 'A' : 'B' ; // I suppose I use tag 0 for A (left operand) and tag 1 for B (right operand)
        sprintf(tmpstr, "[p %d, t %d, m %c]", rank, target, desc);
        printf("<%s>[%s] %lf ms, %lf ms, %lu B, %lu B\n", tmpstr, "internode_comm(comp+comm+size)",
               (spacomm != nullptr) ? (__timer_vals_compression_time.back()) : 0.0, internode_comm,
               compute_message_size<IT,VT>(csx->nnz,    ptrsize+1),
               compute_message_size<IT,VT>(tosend->nnz, ptrsize+1)
        );

#if DEBUG_MAIN
        fprintf(stdout, "Rank %d -- Servicing request from rank %d -- %d/%d requests serviced\n", rank, target, queue.serviced, queue.size);
        FLUSH_WAIT(1.0);
#endif

    }
}

template <typename IT, typename VT>
mmio::CSX<IT, VT>* hns_spgemm_main(KWrapDMat<IT, VT>& kwd_A, KWrapDMat<IT, VT>& kwd_B, const Implementation impl, ThreadPool& pool,
                                   SpaComm::SpaCommHandler<IT, VT> *spcomm, bool skipspgemm)
{
    // ------------------ Test compression on an independent buffer ------------------
#ifdef NVTX_PROFILING
    NVTX_PUSH_RANGE("Copy for bug fix",2);
#endif

    mmio::CSX<IT,VT> *bku_B = (mmio::CSX<IT,VT>*)malloc(sizeof(mmio::CSX<IT,VT>));
    {
        bku_B->majordim = kwd_B.mmio_csx->majordim;
        bku_B->nnz      = kwd_B.mmio_csx->nnz;
        bku_B->nrows    = kwd_B.mmio_csx->nrows;
        bku_B->ncols    = kwd_B.mmio_csx->ncols;

        VT *new_val;
        IT *new_row, *new_idx;
        CUDA_CHECK(cudaMalloc(&new_row, sizeof(IT)*(bku_B->nrows +1)));
        CUDA_CHECK(cudaMalloc(&new_idx, sizeof(IT)*(bku_B->nnz)));
        CUDA_CHECK(cudaMalloc(&new_val, sizeof(VT)*(bku_B->nnz)));
        CUDA_CHECK(cudaMemcpy(new_row, kwd_B.mmio_csx->ptr_vec, sizeof(IT)*(bku_B->nrows +1), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(new_idx, kwd_B.mmio_csx->idx_vec, sizeof(IT)*(bku_B->nnz),      cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(new_val, kwd_B.mmio_csx->val,     sizeof(VT)*(bku_B->nnz),      cudaMemcpyDeviceToDevice));
        bku_B->val      = new_val;
        bku_B->ptr_vec  = new_row;
        bku_B->idx_vec  = new_idx;
    }

    mmio::CSX<IT,VT> *bku_A = (mmio::CSX<IT,VT>*)malloc(sizeof(mmio::CSX<IT,VT>));
    {
        bku_A->majordim = kwd_A.mmio_csx->majordim;
        bku_A->nnz      = kwd_A.mmio_csx->nnz;
        bku_A->nrows    = kwd_A.mmio_csx->nrows;
        bku_A->ncols    = kwd_A.mmio_csx->ncols;

        VT *new_val;
        IT *new_row, *new_idx, ptr_size = (bku_A->majordim == mmio::MajorDim::ROWS ) ? (bku_A->nrows +1) : (bku_A->ncols +1) ;
        CUDA_CHECK(cudaMalloc(&new_row, sizeof(IT)*(ptr_size)));
        CUDA_CHECK(cudaMalloc(&new_idx, sizeof(IT)*(bku_A->nnz)));
        CUDA_CHECK(cudaMalloc(&new_val, sizeof(VT)*(bku_A->nnz)));
        CUDA_CHECK(cudaMemcpy(new_row, kwd_A.mmio_csx->ptr_vec, sizeof(IT)*(ptr_size),   cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(new_idx, kwd_A.mmio_csx->idx_vec, sizeof(IT)*(bku_A->nnz), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(new_val, kwd_A.mmio_csx->val,     sizeof(VT)*(bku_A->nnz), cudaMemcpyDeviceToDevice));
        bku_A->val      = new_val;
        bku_A->ptr_vec  = new_row;
        bku_A->idx_vec  = new_idx;
    }

#ifdef NVTX_PROFILING
    NVTX_POP_RANGE;
#endif
    // -------------------------------------------------------------------------------

    // Process grid info
    dmmio::ProcessGrid * grid = kwd_A.partitioning->grid;
    int node_size        = grid->node_size;                     // NOTE: every grid must have the same node size!!
    int common_grid_size = kwd_A.partitioning->grid->row_size; // This must be equal to kwd_B->...->col_size


    // Number of iterations (i.e. number of tile to fetch to complete the global SpGEMM)
    const int n_iters = common_grid_size;


    // Are we using Acsc_flag?
    bool Acsc_flag = (kwd_A.mmio_csx->majordim == mmio::MajorDim::COLS);


    // Indices of tiles to fetch in the first iteration from each communicator
    int colAtoGet = (grid->row_rank + grid->col_rank) % common_grid_size; // Stragger left
    int rowBtoGet = (grid->col_rank + grid->row_rank) % common_grid_size; // Stragger down


    // Message queue setup -- these will contain indices of the processes that request tiles of A and tiles of B
    MessageQueue<int> A_queue(n_iters, grid->row_comm);
    MessageQueue<int> B_queue(n_iters, grid->col_comm);

    // CHECK_PTRVEC(kwd_B.mmio_csx->ptr_vec, kwd_B.mmio_csx->nrows+1)

    // Get max nnz for A and B tiles (to allocate recv buffers once)
    uint64_t A_max_nnz = (uint64_t)(kwd_A.getLocalNnz()); // have to cast, since MPI_MAX won't work on MPIType<IT>()
    MPI_Allreduce(MPI_IN_PLACE, &A_max_nnz, 1, MPI_UINT64_T, MPI_MAX, grid->row_comm);

    uint64_t B_max_nnz = (uint64_t)(kwd_B.getLocalNnz());
    MPI_Allreduce(MPI_IN_PLACE, &B_max_nnz, 1, MPI_UINT64_T, MPI_MAX, grid->col_comm);

    // CHECK_PTRVEC(kwd_B.mmio_csx->ptr_vec, kwd_B.mmio_csx->nrows+1)

#ifdef NVTX_PROFILING
    NVTX_PUSH_RANGE("Alloc holders & buffers",2);
#endif

    // Tile holders for A and B -- these are buffers that remote processes will write tiles to
    TileHolder<IT, VT> A_holder(kwd_A.getLocalPtrvecsize(), (IT)A_max_nnz*1.5, grid->row_comm);
    TileHolder<IT, VT> B_holder(kwd_B.getLocalPtrvecsize(), (IT)B_max_nnz*1.5, grid->col_comm);
    CsxBuffers<IT,VT> *gather_buffs = new CsxBuffers<IT,VT>(B_max_nnz*1.5, kwd_B.getLocalNrows()*node_size +1);
    CsxBuffers<IT,VT> *conversion_buffs = new CsxBuffers<IT,VT>(A_max_nnz*1.5, kwd_A.getLocalNcols()+1);

#ifdef NVTX_PROFILING
    NVTX_POP_RANGE;
#endif

    // CHECK_PTRVEC(kwd_B.mmio_csx->ptr_vec, kwd_B.mmio_csx->nrows+1)

    // Cuda stream for main thread
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // cusparse handle
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));


    // Local C tile to accumulate the result (during each iter: C += A*B)
    KokkosWrap::LocalMatrix<int32_t, int32_t, float> C_local;

#if DEBUG_MAIN
    MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Launching threads\n"));
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    int dev_id; // To be sure each thread on the same process is assigned to the same GPU
    CUDA_CHECK(cudaGetDevice(&dev_id));

    // Launch comm threads
    std::mutex mpi_mutex;
    auto A_comm_thread = pool.enqueue(comm_thread_loop_csx<IT, VT>,
                            std::ref(A_queue), std::ref(A_holder), bku_A, impl, spcomm, dev_id, grid->row_rank, std::ref(mpi_mutex), 0);
    auto B_comm_thread = pool.enqueue(comm_thread_loop_csx<IT, VT>,
                            std::ref(B_queue), std::ref(B_holder), bku_B, impl, spcomm, dev_id, grid->col_rank, std::ref(mpi_mutex), 1);

#if DEBUG_MAIN
    MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Beginning main loop\n"));
    MPI_Barrier(MPI_COMM_WORLD);
#endif


    // Row and column rank 
    int* row_rank = new int(grid->row_rank);
    int* col_rank = new int(grid->col_rank);


#ifdef DETAILED_TIMERS
    CPU_TIMER_DEF(wait_for_input)
    CUDA_TIMER_DEF(intranode_comm)
    CUDA_TIMER_DEF(comp_time)
    CUDA_TIMER_DEF(A_conversion)
#endif

    CPU_TIMER_DEF(spgemm);


    // Main loop
    alloc_sync_point.arrive_and_wait();
    for (int iter = 0; iter < n_iters; iter++)
    {
        CPU_TIMER_START(spgemm);
        if (grid->global_rank == 0)
        {
            std::cout<<"Iteration "<<iter<<std::endl;
        }

#if DEBUG_MAIN
        int A_owner_global = colAtoGet*node_size + *col_rank * (node_size * grid->row_size) + grid->node_rank;
        int B_owner_global = rowBtoGet*(node_size*grid->row_size);
        fprintf(stdout, "Iteration %d -- Rank %d asking for tile of A from %d and tile of B from %d\n", iter, grid->global_rank,
                A_owner_global,
                B_owner_global);
        FLUSH_WAIT(1.0);
#endif

        // CHECK_PTRVEC(kwd_B.mmio_csx->ptr_vec, kwd_B.mmio_csx->nrows+1)

        // ----- Just for test -----

        // fflush(stdout); fflush(stderr);
/*
        cudaStream_t stream;
        CUDA_TIMER_DEF(compression_time)
        CUDA_CHECK(cudaStreamCreate(&stream));
        mmio::CSX<IT, VT> *csx = bku_B, *csxtosend = nullptr;
        // CHECK_PTRVEC(csx->ptr_vec, csx->nrows+1)
        int rank = kwd_B.partitioning->grid->global_rank, target=colAtoGet;
        char desc = 'B';
        char tmpstr2[20];
        sprintf(tmpstr2, "[p %d, t %d, m %c]", rank, target, desc);
        CUDA_TIMER_START(compression_time, stream)
        csxtosend = spcomm->Compress(csx, target, stream);
        CUDA_TIMER_STOP(compression_time)
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // CHECK_PTRVEC(csxtosend->ptr_vec, csxtosend->nrows+1)

        double comp_rate = ((double)csxtosend->nnz) / ((double) csx->nnz);
        fprintf(stdout, "<%s> Comp-rate: %d,%d,%lf\n", tmpstr2, csx->nnz, csxtosend->nnz, comp_rate);
        ccutils_timers::print_last_time(__timer_vals_compression_time, "compression_time", tmpstr2);
        CSX_destroy_device(&csxtosend);

        desc = 'A';
        csx = bku_A;
        sprintf(tmpstr2, "[p %d, t %d, m %c]", rank, target, desc);
        CUDA_TIMER_START(compression_time, stream)
        csxtosend = spcomm->Compress(csx, target, stream);
        CUDA_TIMER_STOP(compression_time)
        CUDA_CHECK(cudaStreamSynchronize(stream));

        comp_rate = ((double)csxtosend->nnz) / ((double) csx->nnz);
        fprintf(stdout, "<%s> Comp-rate: %d,%d,%lf\n", tmpstr2, csx->nnz, csxtosend->nnz, comp_rate);
        ccutils_timers::print_last_time(__timer_vals_compression_time, "compression_time", tmpstr2);
        CSX_destroy_device(&csxtosend);

        // CUDA_CHECK(cudaDeviceSynchronize());
        // fflush(stdout); fflush(stderr);
        // MPI_Barrier(MPI_COMM_WORLD);
*/
        // --------------------------

        // CHECK_PTRVEC(kwd_B.mmio_csx->ptr_vec, kwd_B.mmio_csx->nrows+1)

        // Tell target I'm ready for tiles of A and B
        A_queue.notify(row_rank, colAtoGet, iter);
        B_queue.notify(col_rank, rowBtoGet, iter);

        // CHECK_PTRVEC(kwd_B.mmio_csx->ptr_vec, kwd_B.mmio_csx->nrows+1)

        // ----- Just for test -----

        // fflush(stdout); fflush(stderr);
        //
        //
        // cudaStream_t stream;
        // CUDA_TIMER_DEF(compression_time)
        // CUDA_CHECK(cudaStreamCreate(&stream));
        // mmio::CSX<IT, VT> *csx = kwd_B.mmio_csx, *csxtosend = nullptr;
        // CHECK_PTRVEC(csx->ptr_vec, csx->nrows+1)
        // int rank = kwd_B.partitioning->grid->global_rank, target=colAtoGet;
        // char desc = (csx->majordim == mmio::MajorDim::ROWS) ? 'B' : 'A' ;
        // char tmpstr2[20];
        // sprintf(tmpstr2, "[p %d, t %d, m %c]", rank, target, desc);
        // CUDA_TIMER_START(compression_time, stream)
        // csxtosend = spcomm->Compress(csx, target, stream);
        // CUDA_TIMER_STOP(compression_time)
        // CUDA_CHECK(cudaStreamSynchronize(stream));
        // CHECK_PTRVEC(csxtosend->ptr_vec, csxtosend->nrows+1)
        //
        // double comp_rate = ((double)csxtosend->nnz) / ((double) csx->nnz);
        // fprintf(stdout, "<%s> Comp-rate: %d,%d,%lf\n", tmpstr2, csx->nnz, csxtosend->nnz, comp_rate);
        // ccutils_timers::print_last_time(__timer_vals_compression_time, "compression_time", tmpstr2);
        // CSX_destroy_device(&csxtosend);
        // CUDA_CHECK(cudaDeviceSynchronize());
        // fflush(stdout); fflush(stderr);
        // MPI_Barrier(MPI_COMM_WORLD);

        // --------------------------

// NOTE:  NVTX ranges are inside '.wait' and '.copy_device_local_csx'
#ifdef DETAILED_TIMERS
        CPU_TIMER_START(wait_for_input)
#endif

        // Wait until I've been sent A and B or copy from the local data
        //   NOTE: here I manage the self communication case
        IT A_tile_nnz, B_tile_nnz;
        if (colAtoGet != grid->row_rank) {
            if (impl == Implementation::PUT) {
                A_tile_nnz = A_holder.wait(colAtoGet);
            } else {
                A_tile_nnz = A_holder.receve_tile(colAtoGet, std::ref(mpi_mutex));
            }
        } else {
            A_tile_nnz = A_holder.copy_device_local_csx(kwd_A.mmio_csx, stream);
        }

        if (rowBtoGet != grid->col_rank) {
            if (impl == Implementation::PUT) {
                B_tile_nnz = B_holder.wait(rowBtoGet);
            } else {
                B_tile_nnz = B_holder.receve_tile(rowBtoGet, std::ref(mpi_mutex));
            }
        } else {
            B_tile_nnz = B_holder.copy_device_local_csx(kwd_B.mmio_csx, stream);
        }

#ifdef DETAILED_TIMERS
        CPU_TIMER_STOP(wait_for_input)
#endif

#if DEBUG_MAIN
        fprintf(stdout, "Rank %d received tiles for iteration %d\n", grid->global_rank, iter);
        FLUSH_WAIT(1.0);
#endif

#ifdef VERBOSE
        fflush(stdout);
        fprintf(stdout, "rank %d: expected A (%dx%d) * expected B (%dx%d)\n", grid->global_rank,
                                                            kwd_A.getLocalNrows(), kwd_A.getLocalNcols(),
                                                            kwd_B.partitioning->group_rows, kwd_B.partitioning->group_cols);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef NVTX_PROFILING
        NVTX_PUSH_RANGE("A_csc2csr_conversion",1);
#endif

        /* TODO check: I must to be carefull here since A can be both a CSR or CSC and all the parameeters
         *  I am considering 'kwd_A->getLocalNrows()' refers to the local owned tiles, not the receved ones.
         */
#ifdef DETAILED_TIMERS
        CUDA_TIMER_START_DEFAULT(A_conversion)
#endif

        KokkosWrap::LocalMatrix<int32_t, int32_t, float> A_remote(handle, A_holder.form_mmiocsx(kwd_A.mmio_csx->nrows, kwd_A.mmio_csx->ncols, A_tile_nnz, kwd_A.mmio_csx->majordim), conversion_buffs);

#ifdef DETAILED_TIMERS
        CUDA_TIMER_STOP(A_conversion)
#endif

#ifdef NVTX_PROFILING
        NVTX_POP_RANGE;
        NVTX_PUSH_RANGE("B_allghater",1);
#endif

#ifdef DETAILED_TIMERS
        CUDA_TIMER_START_DEFAULT(intranode_comm)
#endif

        KokkosWrap::LocalMatrix<int32_t, int32_t, float> B_node(handle, B_holder.node_allgather_mmiocsx(kwd_B.mmio_csx->nrows, kwd_B.mmio_csx->ncols, B_tile_nnz, grid, gather_buffs));

#ifdef DETAILED_TIMERS
        CUDA_TIMER_STOP(intranode_comm)
#endif

#ifdef NVTX_PROFILING
        NVTX_POP_RANGE;
#endif

#ifdef VERBOSE
        fprintf(stdout, "rank %d: A_remote (%dx%d) * B_node (%dx%d)\n", grid->global_rank,
                                                            A_remote.storage.numRows(), A_remote.storage.numCols(),
                                                            B_node.storage.numRows(),   B_node.storage.numCols());
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        // Local multiply
#if DEBUG_MAIN
        fprintf(stdout, "Rank %d beginning multiply for iteration %d\n", grid->global_rank, iter);
        print_d_arr((int*)A_remote.storage.graph.entries.data(), A_remote.storage.nnz(), "A_remote colinds");
        print_d_arr((int*)B_node.storage.graph.entries.data(), B_node.storage.nnz(), "B_remote colinds");
        print_d_arr((int*)A_remote.storage.graph.row_map.data(), A_remote.storage.numRows() + 1, "B_remote rowptrs");
        print_d_arr((int*)B_node.storage.graph.row_map.data(), B_node.storage.numRows() + 1, "B_remote rowptrs");
#endif


#ifdef NVTX_PROFILING
        NVTX_PUSH_RANGE("Local SpGEMM",0);
#endif


#ifdef DETAILED_TIMERS
        CUDA_TIMER_START_DEFAULT(comp_time)
#endif

        // This perform C_local += A_remote * B_node
#ifndef SKIP_SPGEMM
        if (!skipspgemm)
                KokkosWrap::LocalMatrix<int32_t, int32_t, float>::sp_mma(A_remote, B_node, C_local);
#endif

#ifdef DETAILED_TIMERS
        CUDA_TIMER_STOP(comp_time)
#endif

#ifdef NVTX_PROFILING
        NVTX_POP_RANGE;
#endif

        // Round shift
        colAtoGet = (colAtoGet + 1) % common_grid_size; // ShiftLeft
        rowBtoGet = (rowBtoGet + 1) % common_grid_size; // ShiftDown


        // Cleanup
        // B_node underlying storage must be manually freed because its views are unmanaged
        if (gather_buffs==nullptr) B_node.freeBuffers();
        if (Acsc_flag && conversion_buffs == nullptr)
        {
            A_remote.freeBuffers(); // Free A_remote if spcomm, since received tile was copied into a separate buffer
        }

#ifdef DETAILED_TIMERS
        char tmpstr[100];
        sprintf(tmpstr, "[process %d]", grid->global_rank);
        TIMER_PRINT_WPREFIX_STR(wait_for_input, tmpstr)
        TIMER_PRINT_WPREFIX_STR(intranode_comm, tmpstr)
        TIMER_PRINT_WPREFIX_STR(comp_time, tmpstr)
        TIMER_PRINT_WPREFIX_STR(A_conversion, tmpstr)
        fflush(stdout);
#endif

#ifdef BULK_SYNC
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

#if DEBUG_MAIN
    fprintf(stdout, "Rank %d joining on communication threads\n", grid->global_rank);
    FLUSH_WAIT(1.0);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    CPU_TIMER_STOP(spgemm);

#if DEBUG_MAIN
    fprintf(stdout, "Main loop complete for rank %d\n", grid->global_rank);
    FLUSH_WAIT(1.0);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    int64_t nnz_global = (int64_t)C_local.storage.nnz();
    MPI_Allreduce(MPI_IN_PLACE, &nnz_global, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (grid->global_rank==0)
    {
        std::cout<<"NNZ C: "<<nnz_global<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);


    CUSPARSE_CHECK(cusparseDestroy(handle));

    if (grid->global_rank==0)
    {
        TIMER_PRINT_LAST(spgemm);
    }


    mmio::CSX<IT, VT> *out = KokkosWrap::rawptr_get(C_local);

    free_sync_point.arrive_and_wait();

    CSX_destroy_device(&bku_B);
    CSX_destroy_device(&bku_A);

    conversion_buffs->explicitFree();
    gather_buffs->explicitFree();
    delete conversion_buffs;
    delete gather_buffs;

    A_comm_thread.get();
    B_comm_thread.get();
    return out;
}

template mmio::CSX<int32_t, float>* hns_spgemm_main(KWrapDMat<int32_t, float>& kwd_A, KWrapDMat<int32_t, float>& kwd_B, const Implementation impl, ThreadPool& pool, SpaComm::SpaCommHandler<int32_t, float> *spcomm, bool skipspgemm);
