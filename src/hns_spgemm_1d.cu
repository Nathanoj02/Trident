#include "hns_spgemm_1d.cuh"






template <typename IT, typename VT>
void hns_spgemm_1d(DistCusparseCSX<IT, VT> * A, DistCusparseCSX<IT, VT> * B)
{

    dmmio::ProcessGrid * grid = A->partitioning->grid;

    assert(grid->row_size == 1 && grid->col_size == 1);


    int np = grid->global_size;
    const int node_size = 4; //hardcoded for now

    // Set cusparse stream
    cudaStream_t stream = cudaStreamPerThread;
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    int dev_id; // To be sure each thread on the same process is assigned to the same GPU
    CUDA_CHECK(cudaGetDevice(&dev_id));


    // Local partitions of A and B
    CSX<IT, VT> * A_loc = A->csx->mat;
    CSX<IT, VT> * B_loc = B->csx->mat;


    // First, figure out which rows of B need to be sent to each process
    Hns1DHandler<IT, VT> hns_handler(A, B);


    // Do the sends

}





template void hns_spgemm_1d(DistCusparseCSX<int32_t, float> * A, DistCusparseCSX<int32_t, float> * B);
