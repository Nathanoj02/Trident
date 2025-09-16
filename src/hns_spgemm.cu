#include "hns_spgemm.cuh"

MPIDataTypeCache mpidtc; //fix linker error

template <typename IT, typename VT>
void comm_thread_loop2(MessageQueue<int>& A_queue, TileHolder<IT, VT>& A_holder, typename KokkosTypes<IT, VT>::CrsMatrix * A_csr, 
                       MessageQueue<int>& B_queue, TileHolder<IT, VT>& B_holder, typename KokkosTypes<IT, VT>::CrsMatrix * B_csr)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    while (true)
    {

        // Notify remote processes of any tile needs
        A_queue.poll_notify();
        B_queue.poll_notify();


        // See if someone tells me to send them a tile
        int Atarget = A_queue.poll();


        // Only way this should be able to happen is if I've satisfied all requests, so I can return at this point
        if (Atarget >= 0)
        {
            // Put tile on remote process
            A_holder.put_tile(A_csr->values.data(), A_csr->graph.entries.data(), (int32_t*)A_csr->graph.row_map.data(), 
                              A_csr->nnz(), A_csr->numRows(), Atarget);
        }


        // See if someone tells me to send them a tile
        int Btarget = B_queue.poll();


        // Only way this should be able to happen is if I've satisfied all requests, so I can return at this point
        if (Btarget >= 0)
        {
            // Put tile on remote process
            B_holder.put_tile(B_csr->values.data(), B_csr->graph.entries.data(), (int32_t*)B_csr->graph.row_map.data(), 
                              B_csr->nnz(), B_csr->numRows(), Btarget);
        }


        // Return 
        if (A_queue.done() && B_queue.done())
        {
            return;
        }

    }

}

template <typename IT, typename VT>
void comm_thread_loop(MessageQueue<int>& queue, TileHolder<IT, VT>& holder, typename KokkosTypes<IT, VT>::CrsMatrix * csr)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
        holder.put_tile(csr->values.data(), csr->graph.entries.data(), (int32_t*)csr->graph.row_map.data(), csr->nnz(), csr->numRows(), target);
        queue.serviced++;
#if DEBUG_MAIN
        fprintf(stdout, "Rank %d -- Servicing request from rank %d -- %d/%d requests serviced\n", rank, target, queue.serviced, queue.size);
        FLUSH_WAIT(1.0);
#endif
    }

}


template <typename IT, typename VT>
DistCSR<IT, VT> * hns_spgemm_main(DistCSR<IT, VT> * dist_A, DistCSR<IT, VT> * dist_B) 
{
    // Type aliases
    using LocalCSR = DistCSR<IT, VT>::LocalCSR;


    // Matrix bookeeping
    IT A_local_nnz = dist_A->csr->nnz();
    IT A_local_nrows = dist_A->csr->numRows();
    IT A_local_ncols = dist_A->csr->numCols();

    IT B_local_nnz = dist_B->csr->nnz();
    IT B_local_nrows = dist_B->csr->numRows();
    IT B_local_ncols = dist_B->csr->numCols();


    // Process grid info
    dmmio::ProcessGrid * grid = dist_A->partitioning->grid;
    // TODO: check both grids are the same 
    dmmio::utils::ProcessGrid_graph(grid, stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    int node_size = grid->node_size; // NOTE: every grid must have the same node size!!
    int common_grid_size = dist_A->partitioning->grid->row_size; // == dcoo_B->partitioning->grid->col_size
                                                

    // Indices of tiles to fetch from each communicator
    int colAtoGet = (grid->row_rank + grid->col_rank) % common_grid_size; // Stragger left
    int rowBtoGet = (grid->col_rank + grid->row_rank) % common_grid_size; // Stragger down


    // Number of iterations 
    const int n_iters = dist_A->partitioning->grid->row_size; // This must be equal to dcoo_B->partitioning->grid->col_size


    // Message queue setup -- these will contain indices of the processes that request tiles of A and tiles of B
    MessageQueue<int> A_queue(n_iters, grid->row_comm); 
    MessageQueue<int> B_queue(n_iters, grid->col_comm); 


    // Get max nnz for A and B tiles 
    uint64_t A_max_nnz = (uint64_t)dist_A->csr->nnz(); // have to cast, since MPI_MAX won't work on MPIType<IT>()
    MPI_Allreduce(MPI_IN_PLACE, &A_max_nnz, 1, MPI_UINT64_T, MPI_MAX, grid->row_comm);

    uint64_t B_max_nnz = (uint64_t)dist_B->csr->nnz();
    MPI_Allreduce(MPI_IN_PLACE, &B_max_nnz, 1, MPI_UINT64_T, MPI_MAX, grid->col_comm);


    // Tile holders for A and B -- these are buffers that remote processes will write tiles to
    TileHolder<IT, VT> A_holder(dist_A->partitioning->local_rows, (IT)A_max_nnz*1.5, grid->row_comm);
    TileHolder<IT, VT> B_holder(dist_B->partitioning->local_rows, (IT)B_max_nnz*1.5, grid->col_comm);


#ifdef SPCOMM
    //TODO: Transpose my local tile of A if spcomm
#endif


    // Local C
    LocalCSR * C_local = new LocalCSR();

    MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Launching threads\n"));
    MPI_Barrier(MPI_COMM_WORLD);

    // Launch comm threads
    std::thread comm_thread(comm_thread_loop2<IT, VT>, 
                            std::ref(A_queue), std::ref(A_holder), dist_A->csr,
                            std::ref(B_queue), std::ref(B_holder), dist_B->csr); 

    MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Beginning main loop\n"));
    MPI_Barrier(MPI_COMM_WORLD);

    int* row_rank = new int(grid->row_rank);
    int* col_rank = new int(grid->col_rank);

    for (int iter = 0; iter < n_iters; iter++)
    {


#if DEBUG_MAIN
        int A_owner_global = colAtoGet*node_size + *col_rank * (node_size * grid->row_size) + grid->node_rank;
        int B_owner_global = rowBtoGet*(node_size*grid->row_size);
        fprintf(stdout, "Iteration %d -- Rank %d asking for tile of A from %d and tile of B from %d\n", iter, grid->global_rank, 
                A_owner_global, 
                B_owner_global);
        FLUSH_WAIT(1.0);
#endif

        // Tell target I'm ready for tiles of A and B
        A_queue.local_notify(row_rank, colAtoGet, iter);
        B_queue.local_notify(col_rank, rowBtoGet, iter);


        // Wait until I've been sent A and B
        IT A_tile_nnz = A_holder.wait(colAtoGet);
        IT B_tile_nnz = B_holder.wait(rowBtoGet);

#if DEBUG_MAIN
        fprintf(stdout, "Rank %d received tiles for iteration %d\n", grid->global_rank, iter);
        FLUSH_WAIT(1.0);
#endif


        // Create Kokkos CRS instances for local multiplication
        LocalCSR * A_remote = A_holder.form_tile(A_local_nrows, A_local_ncols, A_tile_nnz);


        // Allgatherv of B
        LocalCSR * B_node = B_holder.node_allgather_tiles(B_local_nrows, B_local_ncols, B_tile_nnz, grid);


        // Local multiply
#if DEBUG_MAIN
        fprintf(stdout, "Rank %d beginning multiply for iteration %d\n", grid->global_rank, iter);
        FLUSH_WAIT(1.0);
        print_d_arr((int*)A_remote->graph.row_map.data(), A_remote->numRows()+1, "A remote rowptrs: ");
        print_d_arr(A_remote->graph.entries.data(), A_remote->nnz(), "A remote colinds: ");

        FLUSH_WAIT(1.0);
        print_d_arr((int*)B_node->graph.row_map.data(), B_node->numRows()+1, "B remote rowptrs: ");
        print_d_arr(B_node->graph.entries.data(), B_node->nnz(), "B remote colinds: ");
#endif

        //kokkos_spgemm<IT, VT>(*A_remote, *B_node, *C_local); 


        // Round shift
        colAtoGet = (colAtoGet + 1) % common_grid_size; // ShiftLeft
        rowBtoGet = (rowBtoGet + 1) % common_grid_size; // ShiftDown
                                                        

        // Cleanup
        delete A_remote;
        // Do not free underlying storage of A_remote, since it's the same as is used for A_holder

        // B_node underlying storage must be manually freed because its views are unmanaged
        CUDA_FREE_SAFE(B_node->values.data());
        CUDA_FREE_SAFE(B_node->graph.entries.data());
        CUDA_FREE_SAFE((void*)B_node->graph.row_map.data());
        delete B_node;
#ifdef BULK_SYNC
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

    A_queue.tell_done_notifying();
    B_queue.tell_done_notifying();

#if DEBUG_MAIN
    fprintf(stdout, "Rank %d joining on communication threads\n", grid->global_rank);
    FLUSH_WAIT(1.0);
#endif

    comm_thread.join();

#if DEBUG_MAIN
    MPI_Barrier(MPI_COMM_WORLD);
    //print_d_arr((int*)C_local->graph.entries.data(), C_local->numRows()+1, "C_local rowptrs: ");
    //print_d_arr(C_local->values.data(), C_local->nnz(), "C_local values: ");
    FLUSH_WAIT(1.0);
    MPI_Barrier(MPI_COMM_WORLD);
#endif


#if DEBUG_MAIN
    fprintf(stdout, "Main loop complete for rank %d\n", grid->global_rank);
    FLUSH_WAIT(1.0);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    int64_t nnz_global = (int64_t)C_local->nnz();
    MPI_Allreduce(MPI_IN_PLACE, &nnz_global, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

    if (grid->global_rank==0)
    {
        std::cout<<"NNZ C: "<<nnz_global<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);


    return new DistCSR<IT, VT>{C_local, dist_A->partitioning};
}

template DistCSR<int32_t, float> * hns_spgemm_main(DistCSR<int32_t, float> * dist_A, DistCSR<int32_t, float> * dist_B) ;
