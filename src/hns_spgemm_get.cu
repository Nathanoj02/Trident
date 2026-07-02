#include "hns_spgemm_get.cuh"
#include "tile_window.cuh"
#include <ccutils/cuda/cuda_timers.h>

template <typename IT, typename VT>
mmio::CSX<IT, VT> * hns_spgemm_get(KWrapDMat<IT, VT>& kwd_A, KWrapDMat<IT, VT>& kwd_B)
{
#ifdef DETAILED_TIMERS
    CUDA_TIMER_DEF(internode_comm)
    CUDA_TIMER_DEF(intranode_comm)
    CUDA_TIMER_DEF(comp_time)
#endif

    CPU_TIMER_DEF(spgemm);


    // Asserts
    assert(kwd_A.mmio_csx->majordim == mmio::MajorDim::ROWS);


    // Process grid info
    auto grid = kwd_A.partitioning->grid;
    const int n_iters = grid->row_size;


    // Indices of tiles to fetch in the first iteration from each communicator
    int colAtoGet = (grid->row_rank + grid->col_rank) % n_iters; // Stragger left
    int rowBtoGet = (grid->col_rank + grid->row_rank) % n_iters; // Stragger down


    // Get max nnz for A and B tiles (to allocate recv buffers once)
    uint64_t A_max_nnz = (uint64_t)(kwd_A.mmio_csx->nnz); // have to cast, since MPI_MAX won't work on MPIType<IT>()
    MPI_Allreduce(MPI_IN_PLACE, &A_max_nnz, 1, MPI_UINT64_T, MPI_MAX, grid->row_comm);

    uint64_t B_max_nnz = (uint64_t)(kwd_B.mmio_csx->nnz);
    MPI_Allreduce(MPI_IN_PLACE, &B_max_nnz, 1, MPI_UINT64_T, MPI_MAX, grid->col_comm);
    // This is necessary to prevent non-empty windows from being created using null pointers


    // Local C tile to accumulate the result (during each iter: C += A*B)
    KWrapLMat<IT, VT> KC_local;

    
    // Tiles to hold remote chunks of the inputs
    mmio::CSR<IT, VT> A_remote, B_remote;
    A_remote.nrows = kwd_A.mmio_csx->nrows;
    A_remote.ncols = kwd_A.mmio_csx->ncols;

    CHECK_CUDA(cudaMalloc(&A_remote.val, sizeof(VT) * A_max_nnz));
    CHECK_CUDA(cudaMalloc(&A_remote.col_idx, sizeof(IT) * A_max_nnz));
    CHECK_CUDA(cudaMalloc(&A_remote.row_ptr, sizeof(IT) * (A_remote.nrows+1)));


    B_remote.nrows = kwd_B.mmio_csx->nrows;
    B_remote.ncols = kwd_B.mmio_csx->ncols;

    CHECK_CUDA(cudaMalloc(&B_remote.val, sizeof(VT) * B_max_nnz));
    CHECK_CUDA(cudaMalloc(&B_remote.col_idx, sizeof(IT) * B_max_nnz));
    CHECK_CUDA(cudaMalloc(&B_remote.row_ptr, sizeof(IT) * (B_remote.nrows+1)));

#if DEBUG_GET
    par_print("mmio A ptr: %p, nnzA: %d\n", kwd_A.mmio_csx->val, kwd_A.mmio_csx->nnz);
    par_print("mmio B ptr: %p, nnzB: %d\n", kwd_B.mmio_csx->val, kwd_B.mmio_csx->nnz);
#endif

    // Tile windows
    TileWindow<IT, VT> A_win(kwd_A.mmio_csx->val, kwd_A.mmio_csx->idx_vec, kwd_A.mmio_csx->ptr_vec, 
                             kwd_A.mmio_csx->nnz, kwd_A.mmio_csx->nrows, 
                             grid->row_comm);
    TileWindow<IT, VT> B_win(kwd_B.mmio_csx->val, kwd_B.mmio_csx->idx_vec, kwd_B.mmio_csx->ptr_vec, 
                             kwd_B.mmio_csx->nnz, kwd_B.mmio_csx->nrows, 
                             grid->col_comm);


    // Row and column rank 
    int row_rank = grid->row_rank;
    int col_rank = grid->col_rank;


    // Precompute nnz to be fetched each iteration
    std::vector<int32_t> A_tile_nnz(n_iters, 0);
    std::vector<int32_t> B_tile_nnz(n_iters, 0);

    A_tile_nnz[row_rank] = kwd_A.mmio_csx->nnz;
    B_tile_nnz[col_rank] = kwd_B.mmio_csx->nnz;

    MPI_Allreduce(MPI_IN_PLACE, A_tile_nnz.data(), n_iters, MPI_INT32_T, MPI_SUM, grid->row_comm);
    MPI_Allreduce(MPI_IN_PLACE, B_tile_nnz.data(), n_iters, MPI_INT32_T, MPI_SUM, grid->col_comm);


    // Main loop
    CPU_TIMER_START(spgemm);
    for (int iter=0; iter<n_iters; iter++)
    {

#if DEBUG_GET
        par_print("Iteration %d\b", iter);
#endif


#ifdef DETAILED_TIMERS
        CUDA_TIMER_START_DEFAULT(internode_comm)
#endif
        // Get remote tiles
        A_win.get_tile(&A_remote, A_tile_nnz[colAtoGet], colAtoGet, grid->row_comm);
        B_win.get_tile(&B_remote, B_tile_nnz[rowBtoGet], rowBtoGet, grid->col_comm);
#ifdef DETAILED_TIMERS
        CUDA_TIMER_STOP(internode_comm)
#endif


#if DEBUG_GET
        print_d_arr(A_remote.col_idx, A_remote.nnz, "A_remote colinds: ");
        print_d_arr(B_remote.col_idx, B_remote.nnz, "B_remote colinds: ");
        print_d_arr(A_remote.row_ptr, A_remote.nrows + 1, "A_remote rowptrs: ");
        print_d_arr(B_remote.row_ptr, B_remote.nrows + 1, "B_remote rowptrs: ");
#endif


        // Convert to kokkos crs
        KWrapLMat<IT, VT> KA_remote(&A_remote);


#ifdef DETAILED_TIMERS
        CUDA_TIMER_START_DEFAULT(intranode_comm)
#endif
        // Allgather B
        KWrapLMat<IT, VT> KB_node(B_win.node_allgather_mmiocsr(&B_remote, B_remote.nrows, B_remote.ncols, B_tile_nnz[rowBtoGet], grid)); 
#ifdef DETAILED_TIMERS
        CUDA_TIMER_STOP(intranode_comm)
#endif


#ifdef DETAILED_TIMERS
        CUDA_TIMER_START_DEFAULT(comp_time)
#endif
        // Local multiply-accumulate
        KWrapLMat<IT, VT>::sp_mma(KA_remote, KB_node, KC_local);
#ifdef DETAILED_TIMERS
        CUDA_TIMER_STOP(comp_time)
#endif


        colAtoGet = (colAtoGet + 1) % n_iters; // ShiftLeft
        rowBtoGet = (rowBtoGet + 1) % n_iters; // ShiftDown


        // Cleanup
        KB_node.freeBuffers();

#ifdef DETAILED_TIMERS
    char tmpstr[100];
    sprintf(tmpstr, "[process %d]", grid->global_rank);
    ccutils_timers::print_stats(__timer_vals_internode_comm, "internode_comm", tmpstr);  // TMP FIX
    ccutils_timers::print_stats(__timer_vals_intranode_comm, "intranode_comm", tmpstr);  // TMP FIX
    ccutils_timers::print_stats(__timer_vals_comp_time, "comp_time", tmpstr);  // TMP FIX
    fflush(stdout);
#endif
    }


    // Cleanup
    CUDA_FREE_SAFE(A_remote.val);
    CUDA_FREE_SAFE(A_remote.col_idx);
    CUDA_FREE_SAFE(A_remote.row_ptr);

    CUDA_FREE_SAFE(B_remote.val);
    CUDA_FREE_SAFE(B_remote.col_idx);
    CUDA_FREE_SAFE(B_remote.row_ptr);

    MPI_Barrier(MPI_COMM_WORLD);
    CPU_TIMER_STOP(spgemm);


    mmio::CSX<IT, VT> *out = KC_local.get_csx(); // KokkosWrap::rawptr_get(KC_local);
    int64_t nnz_local = out->nnz;
    MPI_Allreduce(MPI_IN_PLACE, &nnz_local, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#ifdef DETAILED_TIMERS
    if (grid->global_rank==0)
    {
        std::cout<<"NNZ C: "<<nnz_local<<std::endl;
    }


    if (grid->global_rank==0)
    {
        TIMER_PRINT_LAST(spgemm);
    }
#endif

    return out;

}

template mmio::CSX<int32_t, float>* hns_spgemm_get(KWrapDMat<int32_t, float>& kwd_A, KWrapDMat<int32_t, float>& kwd_B);
