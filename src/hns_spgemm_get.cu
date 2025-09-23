#include "hns_spgemm_get.cuh"
#include "tile_window.cuh"



template <typename IT, typename VT>
mmio::CSX<IT, VT> * hns_spgemm_get(KWrapDMat<IT, VT>& kwd_A, KWrapDMat<IT, VT>& kwd_B)
{

    // Process grid info
    auto grid = kwd_A.partitioning->grid;
    const int n_iters = grid->row_size;


    // Indices of tiles to fetch in the first iteration from each communicator
    int colAtoGet = (grid->row_rank + grid->col_rank) % n_iters; // Stragger left
    int rowBtoGet = (grid->col_rank + grid->row_rank) % n_iters; // Stragger down


    // Get max nnz for A and B tiles (to allocate recv buffers once)
    uint64_t A_max_nnz = (uint64_t)(kwd_A.getLocalNnz()); // have to cast, since MPI_MAX won't work on MPIType<IT>()
    MPI_Allreduce(MPI_IN_PLACE, &A_max_nnz, 1, MPI_UINT64_T, MPI_MAX, grid->row_comm);

    uint64_t B_max_nnz = (uint64_t)(kwd_B.getLocalNnz());
    MPI_Allreduce(MPI_IN_PLACE, &B_max_nnz, 1, MPI_UINT64_T, MPI_MAX, grid->col_comm);


    // Local C tile to accumulate the result (during each iter: C += A*B)
    KWrapLMat<IT, VT> KC_local;

    
    // Tiles to hold remote chunks of the inputs
    mmio::CSR<IT, VT> A_remote, B_remote;
    A_remote.nrows = kwd_A.getLocalNrows();
    A_remote.ncols = kwd_A.getLocalNcols();

    CHECK_CUDA(cudaMalloc(&A_remote.val, sizeof(VT) * A_max_nnz));
    CHECK_CUDA(cudaMalloc(&A_remote.col_idx, sizeof(IT) * A_max_nnz));
    CHECK_CUDA(cudaMalloc(&A_remote.row_ptr, sizeof(IT) * (A_remote.nrows+1)));


    B_remote.nrows = kwd_B.getLocalNrows();
    B_remote.ncols = kwd_B.getLocalNcols();

    CHECK_CUDA(cudaMalloc(&B_remote.val, sizeof(VT) * B_max_nnz));
    CHECK_CUDA(cudaMalloc(&B_remote.col_idx, sizeof(IT) * B_max_nnz));
    CHECK_CUDA(cudaMalloc(&B_remote.row_ptr, sizeof(IT) * (B_remote.nrows+1)));


    // Tile windows
    TileWindow<IT, VT> A_win(kwd_A.mmio_csx->val, kwd_A.mmio_csx->idx_vec, kwd_A.mmio_csx->ptr_vec, 
                             A_remote.nrows, A_max_nnz, grid->row_comm);
    TileWindow<IT, VT> B_win(kwd_B.mmio_csx->val, kwd_B.mmio_csx->idx_vec, kwd_B.mmio_csx->ptr_vec, 
                             B_remote.nrows, B_max_nnz, grid->row_comm);


    // Row and column rank 
    int row_rank = grid->row_rank;
    int col_rank = grid->col_rank;


    // Precompute nnz to be fetched each iteration
    std::vector<IT> A_tile_nnz(n_iters, 0);
    std::vector<IT> B_tile_nnz(n_iters, 0);

    A_tile_nnz[row_rank] = kwd_A.getLocalNnz();
    B_tile_nnz[col_rank] = kwd_B.getLocalNnz();

    MPI_Allreduce(MPI_IN_PLACE, A_tile_nnz.data(), n_iters, MPIType<IT>(), MPI_SUM, grid->row_comm);
    MPI_Allreduce(MPI_IN_PLACE, B_tile_nnz.data(), n_iters, MPIType<IT>(), MPI_SUM, grid->col_comm);


    // Main loop
    for (int iter=0; iter<n_iters; iter++)
    {
        if (grid->global_rank == 0)
        {
            std::cout<<"Iteration "<<iter<<std::endl;
        }


        // Get remote tiles
        A_win.get_tile(&A_remote, A_tile_nnz[colAtoGet], colAtoGet, grid->row_comm);
        B_win.get_tile(&B_remote, B_tile_nnz[rowBtoGet], rowBtoGet, grid->col_comm);


        // Convert to kokkos crs
        KWrapLMat<IT, VT> KA_remote(&A_remote);


        // Allgather B
        KWrapLMat<IT, VT> KB_node(B_win.node_allgather_mmiocsr(&B_remote, B_remote.nrows, B_remote.ncols, B_tile_nnz[rowBtoGet], grid)); 


        // Local multiply-accumulate
        KWrapLMat<IT, VT>::sp_mma(KA_remote, KB_node, KC_local);


        colAtoGet = (colAtoGet + 1) % n_iters; // ShiftLeft
        rowBtoGet = (rowBtoGet + 1) % n_iters; // ShiftDown


        // Cleanup
        KB_node.freeBuffers();
    }


    // Cleanup
    CUDA_FREE_SAFE(A_remote.val);
    CUDA_FREE_SAFE(A_remote.col_idx);
    CUDA_FREE_SAFE(A_remote.row_ptr);

    CUDA_FREE_SAFE(B_remote.val);
    CUDA_FREE_SAFE(B_remote.col_idx);
    CUDA_FREE_SAFE(B_remote.row_ptr);



    mmio::CSX<IT, VT> *out = KokkosWrap::rawptr_get(KC_local);
    return out;

}

template mmio::CSX<int32_t, float>* hns_spgemm_get(KWrapDMat<int32_t, float>& kwd_A, KWrapDMat<int32_t, float>& kwd_B);
