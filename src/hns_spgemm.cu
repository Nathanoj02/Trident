#include "hns_spgemm.cuh"

MPIDataTypeCache mpidtc; //fix linker error


template <typename IT, typename VT>
void comm_thread_loop(MessageQueue<int>& queue, TileHolder<IT, VT>& holder, typename KokkosTypes<IT, VT>::CrsMatrix * csr)
{

    while (true)
    {
        // Wait until someone tells me to send them a tile
        int target = queue.wait();

        // Only way this should be able to happen is if I've satisfied all requests, so I can return at this point
        if (target == -1)
        {
            return;
        }

        // Put tile on remote process
        holder.put_tile(csr->values.data(), csr->graph.entries.data(), (int32_t*)csr->graph.row_map.data(), csr->nnz(), csr->numRows(), target);
    }

}


template <typename IT, typename VT>
DistCSR<IT, VT> * hns_spgemm_main(DistCSR<IT, VT> * dist_A, DistCSR<IT, VT> * dist_B) 
{
    // Type aliases
    using LocalCSR = typename DistCSR<IT, VT>::LocalCSR;

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
    IT A_max_nnz = dist_A->csr->nnz();
    MPI_Allreduce(MPI_IN_PLACE, &A_max_nnz, 1, MPIType<IT>(), MPI_MAX, grid->row_comm);

    IT B_max_nnz = dist_B->csr->nnz();
    MPI_Allreduce(MPI_IN_PLACE, &B_max_nnz, 1, MPIType<IT>(), MPI_MAX, grid->col_comm);


    // Tile holders for A and B -- these are buffers that remote processes will write tiles to
    TileHolder<IT, VT> A_holder(dist_A->partitioning->local_rows, (IT)A_max_nnz*1.5, grid->row_comm);
    TileHolder<IT, VT> B_holder(dist_B->partitioning->local_rows, (IT)B_max_nnz*1.5, grid->col_comm);


    //TODO: Transpose my local tile of A


    // Launch comm threads
    std::thread A_comm_thread(comm_thread_loop<IT, VT>, std::ref(A_queue), std::ref(A_holder), dist_A->csr); 
    std::thread B_comm_thread(comm_thread_loop<IT, VT>, std::ref(B_queue), std::ref(B_holder), dist_B->csr); 

    for (int iter = 0; iter < n_iters; iter++)
    {

        // Tell target I'm ready for tiles of A and B
        A_queue.notify(&(grid->row_rank), colAtoGet, iter);
        B_queue.notify(&(grid->col_rank), rowBtoGet, iter);


        // Wait until I've been sent A and B
        IT A_tile_nnz = A_holder.wait();
        IT B_tile_nnz = B_holder.wait();


        // Create Kokkos CRS instances for local multiplication
        LocalCSR * A_remote = A_holder.form_tile(A_local_nrows, A_local_ncols, A_tile_nnz);
        LocalCSR * B_remote = B_holder.form_tile(B_local_nrows, B_local_ncols, B_tile_nnz);


        // Allgatherv of B
        LocalCSR * B_node;


        // Local multiply


        // Round shift
        colAtoGet = (colAtoGet + 1) % common_grid_size; // ShiftLeft
        rowBtoGet = (rowBtoGet + 1) % common_grid_size; // ShiftDown
                                                        //

        // Free B_node


        // Cleanup
        delete A_remote;
        delete B_remote;
        MPI_Barrier(MPI_COMM_WORLD);
    }


    // Both should have already returned by now, but join in case
    A_comm_thread.join();
    B_comm_thread.join();

    MPI_Barrier(MPI_COMM_WORLD);

}

template DistCSR<int32_t, float> * hns_spgemm_main(DistCSR<int32_t, float> * dist_A, DistCSR<int32_t, float> * dist_B) ;
