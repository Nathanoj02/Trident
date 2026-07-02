#include "sparse_summa.cuh"
#include "getVramData.h"


template <typename IT, typename VT>
DistCusparseCSX<IT, VT> * sparse_summa(DistCusparseCSX<IT, VT> * A, DistCusparseCSX<IT, VT> * B)
{

    dmmio::ProcessGrid * grid = A->partitioning->grid;
    int node_size = grid->node_size;
    assert(node_size==1);
    int niters = grid->row_size;


    IT loc_nnz_A = A->getLocalNnz();
    IT max_nnz_A;
    MPI_Allreduce(&loc_nnz_A, &max_nnz_A, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    IT loc_nnz_B = B->getLocalNnz();
    IT max_nnz_B;
    MPI_Allreduce(&loc_nnz_B, &max_nnz_B, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);


    // Set cusparse stream
    cudaStream_t stream = cudaStreamPerThread;
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    size_t A_buf_size = CSX_buf_size<IT, VT>(A->getLocalNrows(), A->getLocalNcols(), max_nnz_A*1.5, mmio::MajorDim::ROWS);
    size_t B_buf_size = CSX_buf_size<IT, VT>(B->getLocalNrows(), B->getLocalNcols(), max_nnz_B*1.5, mmio::MajorDim::ROWS);

    TileHolder<IT, VT> A_holder(A_buf_size, A->getLocalPtrvecsize(), (IT)max_nnz_A * 1.5, grid->row_comm, grid->node_comm);
    TileHolder<IT, VT> B_holder(B_buf_size, B->getLocalPtrvecsize(), (IT)max_nnz_B * 1.5, grid->col_comm, grid->node_comm);

    CsxBuffers<IT,VT> * C_accum_buffs = new CsxBuffers<IT,VT>(max_nnz_A*1.5, A->getLocalNrows()+1, A->getLocalNcols(), &stream, 1, true);
    CsxBuffers<IT,VT> * C_local_buffs = new CsxBuffers<IT,VT>(max_nnz_A*1.5, A->getLocalNrows()+1, A->getLocalNcols(), &stream, 1, true);

    CusparseCSX<IT, VT> * C_accum = new CusparseCSX<IT,VT>(C_accum_buffs);
    CusparseCSX<IT, VT> * C_local = new CusparseCSX<IT,VT>(C_local_buffs);


    // Local partitions of A and B
    CSX<IT, VT> * A_loc = A->csx->mat;
    CSX<IT, VT> * B_loc = B->csx->mat;


    int dev_id; // To be sure each thread on the same process is assigned to the same GPU
    CUDA_CHECK(cudaGetDevice(&dev_id));


    int row_rank = grid->row_rank;
    int col_rank = grid->col_rank;

    bool did_one_spgemm = false;

    char tmp_name[50];
    sprintf(tmp_name, "Memcnt%d", grid->global_rank);
    MyMemData mymemdata(tmp_name);

    CPU_TIMER_DEF(spgemm);
    CPU_TIMER_DEF(bcast);
    CPU_TIMER_START(spgemm);
    for (int p=0; p<niters; p++)
    {
        char tmp_name[50];
        sprintf(tmp_name, "Iteration%d", p);
        mymemdata.append_measure(tmp_name);

        // Bcast tile of A
        CPU_TIMER_START(bcast);
        IT A_tile_nnz = A_holder.tile_bcast(A_loc, p);
        // Bcast tile of B
        IT B_tile_nnz = B_holder.tile_bcast(B_loc, p);
        CPU_TIMER_STOP(bcast);


        // Form local tiles
        CusparseCSX<IT, VT> * A_remote = new CusparseCSX<IT,VT>(&handle, 
                                                                A_holder.form_mmiocsx(A->csx->nrows(), 
                                                                                      A->csx->ncols(), 
                                                                                      A_tile_nnz, 
                                                                                      A->csx->mat->majordim), 
                                                                nullptr);
        CUDA_SYNC(stream);

        CusparseCSX<IT, VT> * B_remote = new CusparseCSX<IT,VT>(&handle, 
                                                                B_holder.form_mmiocsx(B->csx->nrows(), 
                                                                                      B->csx->ncols(), 
                                                                                      B_tile_nnz, 
                                                                                      B->csx->mat->majordim), 
                                                                nullptr);
        CUDA_SYNC(stream);

        // Local SpGEMM
        if (A_remote->nnz() > 0 && B_remote->nnz() > 0)
        {
            LocalMatrix<IT, IT, VT> A_p(A_remote->mat);
            LocalMatrix<IT, IT, VT> B_p(B_remote->mat);
            sp_mma_hybrid(&handle, A_p, B_p, &C_local, &C_accum, did_one_spgemm);
        }

        CUDA_SYNC(stream);

    }

    // mymemdata.print();
#ifdef DETAILED_TIMERS
    std::string memstr = mymemdata.short_print();
    MPI_ALL_PRINT(fprintf(fp, "%s\n", memstr.c_str()));
    // fprintf(stdout, "%s\n", memstr.c_str());
#endif


    MPI_Barrier(MPI_COMM_WORLD);
    CPU_TIMER_STOP(spgemm);

    int64_t nnz_global = (int64_t)C_local->nnz();
    MPI_Allreduce(MPI_IN_PLACE, &nnz_global, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#ifdef DETAILED_TIMERS
    if (grid->global_rank==0)
    {
        std::cout<<"NNZ C: "<<nnz_global<<std::endl;
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    fflush(stdout);

#ifdef DETAILED_TIMERS
    char tmpstr[100];
    sprintf(tmpstr, "[process %d]", grid->global_rank);
    TIMER_PRINT_WPREFIX_STR(bcast, tmpstr)
    fflush(stdout);

    if (grid->global_rank==0)
    {
        TIMER_PRINT_LAST(spgemm);
    }
    fflush(stdout);
#endif

    delete C_accum;

    C_local->to_mat();

    return new DistCusparseCSX<IT, VT>(C_local, A->partitioning);
}



template DistCusparseCSX<int32_t, float> * sparse_summa(DistCusparseCSX<int32_t, float> * A, DistCusparseCSX<int32_t, float> * B);
