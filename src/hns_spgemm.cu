#include "hns_spgemm.cuh"

MPIDataTypeCache mpidtc; //fix linker error

int here_iteration = 0;


// Barrier for sync afther thread allocs
#include <condition_variable>
SimpleBarrier alloc_sync_point(3);
SimpleBarrier free_sync_point(3);

template <typename IT, typename VT>
inline uint64_t compute_message_size(int nnz, int ptr_size) {
        return( (nnz * sizeof(VT)) + ((nnz + ptr_size) * sizeof(IT)) );
}

template <typename IT, typename VT>
void comm_thread_loop_csx(MessageQueue<int>& queue, TileHolder<IT, VT>& holder, mmio::CSX<IT, VT> * csx, 
                          const Implementation impl, SpaComm::SpaCommHandler<IT,VT>* spacomm, int dev_id, 
                          int comm_rank, std::mutex& mpi_mutex, int tag=0)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cudaStream_t stream;
    CUDA_CHECK(cudaSetDevice(dev_id)); // To be sure each thread on the same process is assigned to the same GPU
    CUDA_CHECK(cudaStreamCreate(&stream));

    SpaComm::SpaCommBuffers<IT,VT> *compression_buffers = new SpaComm::SpaCommBuffers<IT,VT>(csx);


    float internode_comm;
    CUDA_TIMER_DEF(compression_time)
#ifdef NVTX_PROFILING
    int nvtx_color;
    char compression_str[20], comunication_str[20], nvtx_char;
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
        if (spacomm != nullptr) 
        {
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
        assert(tosend->contig && "Tosend must be contiguous");
        internode_comm = holder.send_tile_contig(tosend->buf, tosend->buf_size, tosend->nnz, target, stream, tag);


#ifdef DETAILED_TIMERS
        char tmpstr[20];
        char desc = (tag == 0) ? 'A' : 'B' ; // I suppose I use tag 0 for A (left operand) and tag 1 for B (right operand)
        sprintf(tmpstr, "[p %d, t %d, m %c]", rank, target, desc);
        printf("<%s>[%s] %lf ms, %lf ms, %lu B, %lu B\n", tmpstr, "internode_comm(comp+comm+size)",
               (spacomm != nullptr) ? (__timer_vals_compression_time.back()) : 0.0, internode_comm,
               compute_message_size<IT,VT>(csx->nnz,    ptrsize+1),
               compute_message_size<IT,VT>(tosend->nnz, ptrsize+1)
        );
#endif

#if DEBUG_MAIN
        fprintf(stdout, "Rank %d -- Servicing request from rank %d -- %d/%d requests serviced\n", rank, target, queue.serviced, queue.size);
        FLUSH_WAIT(1000000);
#endif

    }
}


template <typename IT, typename VT>
DistCusparseCSX<IT,VT> * hns_spgemm_main(DistCusparseCSX<IT, VT> * dist_A, DistCusparseCSX<IT, VT> * dist_B, 
                                         const Implementation impl, ThreadPool& pool,
                                         SpaComm::SpaCommHandler<IT, VT> *spcomm, 
                                         bool skipspgemm)
{

    // Process grid info
    dmmio::ProcessGrid * grid = dist_A->partitioning->grid;
    int node_size        = grid->node_size;                     // NOTE: every grid must have the same node size!!
    int common_grid_size = dist_A->partitioning->grid->row_size; // This must be equal to dist_B->...->col_size


    // For forming mpi groups in the gat stuff 
    MPI_Group row_group;
    MPI_Comm_group(grid->row_comm, &row_group);

    MPI_Group col_group;
    MPI_Comm_group(grid->col_comm, &col_group);


    // Number of iterations (i.e. number of tile to fetch to complete the global SpGEMM)
    const int n_iters = common_grid_size;


    // Are we using Acsc_flag?
    bool Acsc_flag = (dist_A->csx->mat->majordim == mmio::MajorDim::COLS);
    mmio::MajorDim majordim = dist_A->csx->mat->majordim; 


    // Indices of tiles to fetch in the first iteration from each communicator
    int colAtoGet = (grid->row_rank + grid->col_rank) % common_grid_size; // Stragger left
    int rowBtoGet = (grid->col_rank + grid->row_rank) % common_grid_size; // Stragger down


    // Message queue setup -- these will contain indices of the processes that request tiles of A and tiles of B
    MessageQueue<int> A_queue(n_iters, grid->row_comm);
    MessageQueue<int> B_queue(n_iters, grid->col_comm);


    // Get max nnz for A and B tiles (to allocate recv buffers once)
    IT A_max_nnz = (dist_A->getLocalNnz()); 
    MPI_Allreduce(MPI_IN_PLACE, &A_max_nnz, 1, MPI_UINT32_T, MPI_MAX, grid->row_comm);


    IT B_max_nnz = (dist_B->getLocalNnz());
    MPI_Allreduce(MPI_IN_PLACE, &B_max_nnz, 1, MPI_UINT32_T, MPI_MAX, grid->col_comm);


#ifdef NVTX_PROFILING
    NVTX_PUSH_RANGE("Alloc holders & buffers",2);
#endif


    // Tile holders for A and B -- these are buffers that remote processes will write tiles to
    size_t A_buf_size = CSX_buf_size<IT, VT>(dist_A->getLocalNrows(), dist_A->getLocalNcols(), A_max_nnz, majordim);
    size_t B_buf_size = CSX_buf_size<IT, VT>(dist_B->getLocalNrows(), dist_B->getLocalNcols(), B_max_nnz, mmio::MajorDim::ROWS);
    TileHolder<IT, VT> A_holder(A_buf_size, dist_A->getLocalPtrvecsize(), (IT)A_max_nnz*1.5, grid->row_comm);
    TileHolder<IT, VT> B_holder(B_buf_size, dist_B->getLocalPtrvecsize(), (IT)B_max_nnz*1.5, grid->col_comm);


    // Temporary allgather buffers
    CsxBuffers<IT,VT> * gather_buffs = new CsxBuffers<IT,VT>(B_max_nnz*1.5, dist_B->getLocalNrows()*node_size + 1, dist_B->getLocalNcols());


    // Temporary csc->csr buffers
    CsxBuffers<IT,VT> * conversion_buffs = new CsxBuffers<IT,VT>(A_max_nnz*1.5, dist_A->getLocalNcols()+1, dist_A->getLocalNrows());


    // Temporary local SpGEMM buffers
    // TODO: Some other size heuristic
    int nbuffers = 6;
    CsxBuffers<IT,VT> * C_prod_buffs = new CsxBuffers<IT,VT>(A_max_nnz*1.5, dist_A->getLocalNrows()+1, dist_A->getLocalNcols(), 6);
    CsxBuffers<IT,VT> * C_local_buffs = new CsxBuffers<IT,VT>(A_max_nnz*1.5, dist_A->getLocalNrows()+1, dist_A->getLocalNcols());
    CsxBuffers<IT,VT> * C_accum_buffs = new CsxBuffers<IT,VT>(A_max_nnz*1.5, dist_A->getLocalNrows()+1, dist_A->getLocalNcols());


    // Make CusparseCSX Objects
    CusparseCSX<IT, VT> * C_prod = new CusparseCSX<IT,VT>(C_prod_buffs);
    CusparseCSX<IT, VT> * C_local = new CusparseCSX<IT,VT>(C_local_buffs);
    CusparseCSX<IT, VT> * C_accum = new CusparseCSX<IT,VT>(C_accum_buffs);

#ifdef NVTX_PROFILING
    NVTX_POP_RANGE;
#endif


    // Cuda stream for main thread
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));


    // Create cusparse handle
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));


    // Set cusparse stream
    CUSPARSE_CHECK(cusparseSetStream(handle, stream));



    // Local partitions of A and B
    CSX<IT, VT> * A_loc = dist_A->csx->mat;
    CSX<IT, VT> * B_loc = dist_B->csx->mat;


    int dev_id; // To be sure each thread on the same process is assigned to the same GPU
    CUDA_CHECK(cudaGetDevice(&dev_id));


    // Launch comm threads
    std::mutex mpi_mutex;
    auto A_comm_thread = pool.enqueue(comm_thread_loop_csx<IT, VT>,
                            std::ref(A_queue), std::ref(A_holder), A_loc, 
                            impl, spcomm, dev_id, 
                            grid->row_rank, std::ref(mpi_mutex), 0);
    auto B_comm_thread = pool.enqueue(comm_thread_loop_csx<IT, VT>,
                            std::ref(B_queue), std::ref(B_holder), B_loc, 
                            impl, spcomm, dev_id, 
                            grid->col_rank, std::ref(mpi_mutex), 1);


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

    // For deciding whether or not to accumulate
    bool done_one_spgemm = false;

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
        fflush(stdout);
        FLUSH_WAIT(1000000);
#endif


#ifdef NVTX_PROFILING
        NVTX_PUSH_RANGE("NotifyReqTilesA",1);
#endif

        // Tell target I'm ready for tiles of A and B
        A_queue.notify(row_rank, colAtoGet, iter);

#ifdef NVTX_PROFILING
        NVTX_POP_RANGE;
        NVTX_PUSH_RANGE("NotifyReqTilesB",1);
#endif

        B_queue.notify(col_rank, rowBtoGet, iter);

#ifdef NVTX_PROFILING
        NVTX_POP_RANGE;
#endif


#ifdef NVTX_PROFILING
        NVTX_PUSH_RANGE("RecvInputTiles",1);
#endif

#ifdef DETAILED_TIMERS
        CPU_TIMER_START(wait_for_input)
#endif

        // Wait until I've been sent A and B or copy from the local data
        IT A_tile_nnz, B_tile_nnz;
        MPI_Request reqs[2];

        if (colAtoGet != grid->row_rank) 
        {
            A_tile_nnz = A_holder.recv_tile_contig(colAtoGet, &reqs[0]);
        } 
        else 
        {
            A_tile_nnz = A_holder.copy_device_local_csx(dist_A->csx->mat, stream);
        }

        if (rowBtoGet != grid->col_rank) 
        {
            B_tile_nnz = B_holder.recv_tile_contig(rowBtoGet, &reqs[1]);
        } 
        else 
        {
            B_tile_nnz = B_holder.copy_device_local_csx(dist_B->csx->mat, stream);
        }

        if (colAtoGet != grid->row_rank && A_tile_nnz > 0)
        {
            MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
        }

        if (rowBtoGet != grid->col_rank && B_tile_nnz > 0)
        {
            MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
        }

        CUDA_SYNC(stream);

#ifdef DETAILED_TIMERS
        CPU_TIMER_STOP(wait_for_input)
#endif

#ifdef NVTX_PROFILING
        NVTX_POP_RANGE;
#endif

#if DEBUG_MAIN
        fprintf(stdout, "Iteration -- %d: Rank %d received tiles for iteration %d\n", iter, grid->global_rank, iter);
        fflush(stdout);
        FLUSH_WAIT(1000000);
#endif

#ifdef VERBOSE
        fflush(stdout);
        fprintf(stdout, "rank %d: expected A (%dx%d) * expected B (%dx%d)\n", grid->global_rank,
                                                            dist_A->getLocalNrows(), dist_A->getLocalNcols(),
                                                            dist_B->partitioning->group_rows, dist_B->partitioning->group_cols);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef NVTX_PROFILING
        NVTX_PUSH_RANGE("A_csc2csr_conversion",1);
#endif

#ifdef DETAILED_TIMERS
        CUDA_TIMER_START_DEFAULT(A_conversion)
#endif


        CusparseCSX<IT, VT> * A_remote = new CusparseCSX<IT,VT>(handle, 
                                                                A_holder.form_mmiocsx(dist_A->csx->nrows(), 
                                                                                      dist_A->csx->ncols(), 
                                                                                      A_tile_nnz, 
                                                                                      dist_A->csx->mat->majordim), 
                                                                conversion_buffs);
        CUDA_SYNC(stream);

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

        CusparseCSX<IT, VT> * B_node = new CusparseCSX<IT, VT>(B_holder.node_allgather_mmiocsx(dist_B->csx->nrows(), dist_B->csx->ncols(), B_tile_nnz, grid, gather_buffs));
        CUDA_SYNC(stream);

#ifdef DETAILED_TIMERS
        CUDA_TIMER_STOP(intranode_comm)
#endif

#ifdef NVTX_PROFILING
        NVTX_POP_RANGE;
#endif

        // Local multiply
#if DEBUG_MAIN
        fprintf(stdout, "Rank %d beginning multiply for iteration %d\n", grid->global_rank, iter);
        //print_d_arr((int*)A_remote.storage.graph.entries.data(), A_remote.storage.nnz(), "A_remote colinds");
        //print_d_arr((int*)B_node.storage.graph.entries.data(), B_node.storage.nnz(), "B_remote colinds");
        //print_d_arr((int*)A_remote.storage.graph.row_map.data(), A_remote.storage.numRows() + 1, "B_remote rowptrs");
        //print_d_arr((int*)B_node.storage.graph.row_map.data(), B_node.storage.numRows() + 1, "B_remote rowptrs");
#endif


#ifdef NVTX_PROFILING
        NVTX_PUSH_RANGE("Local SpGEMM",0);
#endif


#ifdef DETAILED_TIMERS
        CUDA_TIMER_START_DEFAULT(comp_time)
#endif

        // This perform C_local += A_remote * B_node
        if (!skipspgemm && A_remote->nnz() > 0 && B_node->nnz() > 0)
        {
            int did_spgemm = cusparse_spmma<IT, VT>(handle, A_remote, B_node, &C_prod, &C_accum, &C_local, done_one_spgemm, &stream);
            done_one_spgemm = did_spgemm || done_one_spgemm;
        }
        CUDA_SYNC(stream);

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
        //B_node->explicit_free();
        if (Acsc_flag && conversion_buffs == nullptr)
        {
            A_remote->explicit_free();
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
    FLUSH_WAIT(1000000);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    CPU_TIMER_STOP(spgemm);

#if DEBUG_MAIN
    fprintf(stdout, "Main loop complete for rank %d\n", grid->global_rank);
    FLUSH_WAIT(1000000);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    int64_t nnz_global = (int64_t)C_local->nnz();
    MPI_Allreduce(MPI_IN_PLACE, &nnz_global, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (grid->global_rank==0)
    {
        std::cout<<"NNZ C: "<<nnz_global<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);


    CUDA_CHECK(cudaStreamDestroy(stream));
    CUSPARSE_CHECK(cusparseDestroy(handle));

    if (grid->global_rank==0)
    {
        TIMER_PRINT_LAST(spgemm);
    }


    free_sync_point.arrive_and_wait();

    //CSX_destroy_device(&bku_B);
    //CSX_destroy_device(&bku_A);

    //conversion_buffs->explicitFree();
    //gather_buffs->explicitFree();
    delete conversion_buffs;
    delete gather_buffs;
    delete C_prod_buffs;
    delete C_accum_buffs;

    A_comm_thread.get();
    B_comm_thread.get();
    return new DistCusparseCSX<IT,VT>(C_local, dist_A->partitioning);
}

template DistCusparseCSX<int32_t,float> *  hns_spgemm_main(DistCusparseCSX<int32_t, float> * dist_A, DistCusparseCSX<int32_t, float> * dist_B, const Implementation impl, ThreadPool& pool, SpaComm::SpaCommHandler<int32_t, float> *spcomm, bool skipspgemm=false);
