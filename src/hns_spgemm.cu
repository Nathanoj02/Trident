#include "hns_spgemm.cuh"

MPIDataTypeCache mpidtc; //fix linker error

//#ifdef PTR_CHECK
int here_iteration = 0;
//#endif

template <typename IT, typename VT>
void comm_thread_loop_csx(MessageQueue<int>& queue, TileHolder<IT, VT>& holder, mmio::CSX<IT, VT> * csx)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    IT ptrsize = (csx->majordim == mmio::ROWS) ? (csx->nrows) : (csx->ncols) ;
    while (true)
    {
        // Wait until someone tells me to send them a tile
        int target = queue.wait();

        // Only way this should be able to happen is if I've satisfied all requests, so I can return at this point
        if (target == -2)
        {
            return;
        }

        // Put tile on remote process
        holder.put_tile(csx->val, csx->idx_vec, csx->ptr_vec, csx->nnz, ptrsize, target);

#if DEBUG_MAIN
        fprintf(stdout, "Rank %d -- Servicing request from rank %d -- %d/%d requests serviced\n", rank, target, queue.serviced, queue.size);
        FLUSH_WAIT(1000000);
#endif

    }


}

template <typename IT, typename VT>
mmio::CSX<IT, VT>* hns_spgemm_main(KWrapDMat<IT, VT>& kwd_A, KWrapDMat<IT, VT>& kwd_B)
{
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


    // Get max nnz for A and B tiles (to allocate recv buffers once)
    uint64_t A_max_nnz = (uint64_t)(kwd_A.getLocalNnz()); // have to cast, since MPI_MAX won't work on MPIType<IT>()
    MPI_Allreduce(MPI_IN_PLACE, &A_max_nnz, 1, MPI_UINT64_T, MPI_MAX, grid->row_comm);

    uint64_t B_max_nnz = (uint64_t)(kwd_B.getLocalNnz());
    MPI_Allreduce(MPI_IN_PLACE, &B_max_nnz, 1, MPI_UINT64_T, MPI_MAX, grid->col_comm);


    // Tile holders for A and B -- these are buffers that remote processes will write tiles to
    TileHolder<IT, VT> A_holder(kwd_A.getLocalPtrvecsize(), (IT)A_max_nnz*1.5, grid->row_comm);
    TileHolder<IT, VT> B_holder(kwd_B.getLocalPtrvecsize(), (IT)B_max_nnz*1.5, grid->col_comm);


    // cusparse handle
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));


    // Local C tile to accumulate the result (during each iter: C += A*B)
    KokkosWrap::LocalMatrix<int32_t, int32_t, float> C_local;

#if DEBUG_MAIN
    MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Launching threads\n"));
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Launch comm threads
    std::thread A_comm_thread(comm_thread_loop_csx<IT, VT>,
                            std::ref(A_queue), std::ref(A_holder), kwd_A.mmio_csx);
    std::thread B_comm_thread(comm_thread_loop_csx<IT, VT>,
                            std::ref(B_queue), std::ref(B_holder), kwd_B.mmio_csx);

#if DEBUG_MAIN
    MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Beginning main loop\n"));
    MPI_Barrier(MPI_COMM_WORLD);
#endif


    // Row and column rank 
    int* row_rank = new int(grid->row_rank);
    int* col_rank = new int(grid->col_rank);


#ifdef DETAILED_TIMERS
    CUDA_TIMER_DEF(internode_comm)
    CUDA_TIMER_DEF(intranode_comm)
    CUDA_TIMER_DEF(comp_time)
    CUDA_TIMER_DEF(A_conversion)
#endif

    CPU_TIMER_DEF(spgemm);


    // Main loop
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
        FLUSH_WAIT(1000000);
#endif

        // Tell target I'm ready for tiles of A and B
        A_queue.notify(row_rank, colAtoGet, iter);
        B_queue.notify(col_rank, rowBtoGet, iter);


#ifdef DETAILED_TIMERS
        CUDA_TIMER_START_DEFAULT(internode_comm)
#endif

        // Wait until I've been sent A and B
        IT A_tile_nnz = A_holder.wait(colAtoGet);
        IT B_tile_nnz = B_holder.wait(rowBtoGet);


#ifdef DETAILED_TIMERS
        CUDA_TIMER_STOP(internode_comm)
#endif

#if DEBUG_MAIN
        fprintf(stdout, "Rank %d received tiles for iteration %d\n", grid->global_rank, iter);
        FLUSH_WAIT(1000000);
#endif

#ifdef VERBOSE
        fflush(stdout);
        fprintf(stdout, "rank %d: expected A (%dx%d) * expected B (%dx%d)\n", grid->global_rank,
                                                            kwd_A.getLocalNrows(), kwd_A.getLocalNcols(),
                                                            kwd_B.partitioning->group_rows, kwd_B.partitioning->group_cols);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        /* TODO check: I must to be carefull here since A can be both a CSR or CSC and all the parameeters
         *  I am considering 'kwd_A->getLocalNrows()' refers to the local owned tiles, not the receved ones.
         */
#ifdef DETAILED_TIMERS
        CUDA_TIMER_START_DEFAULT(A_conversion)
#endif

        KokkosWrap::LocalMatrix<int32_t, int32_t, float> A_remote(handle, A_holder.form_mmiocsx(kwd_A.mmio_csx->nrows, kwd_A.mmio_csx->ncols, A_tile_nnz, kwd_A.mmio_csx->majordim));

#ifdef DETAILED_TIMERS
        CUDA_TIMER_STOP(A_conversion)
#endif

#ifdef DETAILED_TIMERS
        CUDA_TIMER_START_DEFAULT(intranode_comm)
#endif

        KokkosWrap::LocalMatrix<int32_t, int32_t, float> B_node(handle, B_holder.node_allgather_mmiocsx(kwd_B.mmio_csx->nrows, kwd_B.mmio_csx->ncols, B_tile_nnz, grid));

#ifdef DETAILED_TIMERS
        CUDA_TIMER_STOP(intranode_comm)
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


#ifdef DETAILED_TIMERS
        CUDA_TIMER_START_DEFAULT(comp_time)
#endif

        // This perform C_local += A_remote * B_node
        KokkosWrap::LocalMatrix<int32_t, int32_t, float>::sp_mma(A_remote, B_node, C_local);

#ifdef DETAILED_TIMERS
        CUDA_TIMER_STOP(comp_time)
#endif

        // Round shift
        colAtoGet = (colAtoGet + 1) % common_grid_size; // ShiftLeft
        rowBtoGet = (rowBtoGet + 1) % common_grid_size; // ShiftDown


        // Cleanup
        // B_node underlying storage must be manually freed because its views are unmanaged
        B_node.freeBuffers();
        if (Acsc_flag)
        {
            A_remote.freeBuffers(); // Free A_remote if spcomm, since received tile was copied into a separate buffer
        }

#ifdef BULK_SYNC
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

#ifdef DETAILED_TIMERS
    char tmpstr[100];
    sprintf(tmpstr, "[process %d]", grid->global_rank);
    TIMER_PRINT_WPREFIX_STR(internode_comm, tmpstr)
    TIMER_PRINT_WPREFIX_STR(intranode_comm, tmpstr)
    TIMER_PRINT_WPREFIX_STR(comp_time, tmpstr)
    TIMER_PRINT_WPREFIX_STR(A_conversion, tmpstr)
    fflush(stdout);
#endif

#if DEBUG_MAIN
    fprintf(stdout, "Rank %d joining on communication threads\n", grid->global_rank);
    FLUSH_WAIT(1000000);
#endif

    A_comm_thread.join();
    B_comm_thread.join();

    MPI_Barrier(MPI_COMM_WORLD);
    CPU_TIMER_STOP(spgemm);

#if DEBUG_MAIN
    fprintf(stdout, "Main loop complete for rank %d\n", grid->global_rank);
    FLUSH_WAIT(1000000);
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


    mmio::CSX<IT, VT> *out = C_local.get_csx(); // KokkosWrap::rawptr_get(C_local);
    return out;
}

template mmio::CSX<int32_t, float>* hns_spgemm_main(KWrapDMat<int32_t, float>& kwd_A, KWrapDMat<int32_t, float>& kwd_B);
