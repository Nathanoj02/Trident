#pragma once
#include "common.h"
#include "KokkosWrap.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_adjacent_difference.cuh>

#include <cub/cub.cuh>

namespace SpaComm
{

// ----------------------------------------------------------------------------------------------
// Funtions for the sparsity pattern communication (generate, communicate and intersect bitmasks)
// ----------------------------------------------------------------------------------------------

// A general function to compute result vector given a row/col pointer array
// A general function to compute result vector given a row/col pointer array
template<typename IT>
int8_t* gen_bitmask(const IT* ptr_d, int n, int mask_size) {
    // Each entry is a byte that might hold one bit
    thrust::device_vector<int8_t> presult(n);

    // Mark positions: 1 << (i % mask_size) if row is non-empty
    thrust::transform(
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(n),
        presult.begin(),
        [=] __device__(int i) {
            IT a = ptr_d[i];
            IT b = ptr_d[i + 1];
            int8_t c = 0;
            if (a != b) {
                c = 1 << (i % mask_size);
            }
            return c;
        });

    auto op = cub::Sum();
    int initial_value = 0;
    int segment_size  = mask_size;

    // Number of mask bytes needed
    int num_segments  = (n + mask_size - 1) / mask_size;

    // Allocate result array (one byte per segment)
    int8_t *result;
    CUDA_CHECK(cudaMalloc(&result, sizeof(int8_t) * num_segments));

    // Segment offsets
    thrust::host_vector<int> h_offsets(num_segments + 1);
    for (int i = 0; i < num_segments; i++) {
        h_offsets[i] = i * segment_size;
    }
    h_offsets[num_segments] = n;  // last boundary must be "n", not num_segments*segment_size

    thrust::device_vector<int> d_offsets = h_offsets;

    // Temporary storage requirements
    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage, temp_storage_bytes,
        presult.begin(), result,
        num_segments, d_offsets.data().get(), d_offsets.data().get() + 1,
        op, initial_value
    );

    // Allocate temp storage
    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    // Run reduction
    cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage, temp_storage_bytes,
        presult.begin(), result,
        num_segments, d_offsets.data().get(), d_offsets.data().get() + 1,
        op, initial_value
    );

    return result;
}


struct BitwiseAnd {
    __device__ __host__ int operator()(const thrust::tuple<int,int>& t) const {
        return thrust::get<0>(t) & thrust::get<1>(t);
    }
};

template<typename T>
T* intersect_bitmasks(const T* a, const T* b, int n) {
    // Allocate output buffer on device
    T* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(T) * n));

    // Wrap raw pointers with device_ptr so Thrust can use them
    auto a_begin = thrust::device_pointer_cast(a);
    auto b_begin = thrust::device_pointer_cast(b);
    auto out_begin = thrust::device_pointer_cast(d_out);

    // Zip iterators over both inputs
    auto zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(a_begin, b_begin));
    auto zipped_end   = zipped_begin + n;

    // Create transform iterator that applies bitwise AND
    auto transform_iter = thrust::make_transform_iterator(zipped_begin, BitwiseAnd());

    // Copy results into d_out
    thrust::copy(transform_iter, transform_iter + n, out_begin);

    return d_out; // caller must cudaFree()
}

inline void printBit_left2right(int8_t* bitmask, int nwords, FILE* fp=stdout) {
    for (int i=0; i<nwords; i++) {
        for (int j=0; j<MASK_SIZE; j++) {
            fprintf(fp, "%c", (bitmask[i] & (1<<j)) ? '1' : '0');
        }
        if (i!=(nwords-1)) fprintf(fp, "|");
    }
}

// #define DEBUG_SPCOMM

template<typename IT, typename VT>
void spcomm_2D (mmio::CSX<IT,VT> *Acsc, mmio::CSX<IT,VT> *Bcsr, dmmio::ProcessGrid *grid,
                int8_t** col_filters, int8_t** row_filters) {
    ASSERT(Acsc->majordim==mmio::MajorDim::COLS, "A must be a CSC");
    ASSERT(Bcsr->majordim==mmio::MajorDim::ROWS, "B must be a CSR");
    ASSERT(Acsc->ncols == Bcsr->nrows, "In 2D mask cols of A must be equal to rows of B");
    ASSERT(grid->row_size == grid->col_size, "The 2D grid must be a square");
    ASSERT(grid->node_size == 1, "The spcomm 2D do not support node size > 1");
    ASSERT((Acsc->ncols % 8) == 0, "The columns of A must divide the bit in a world of the bitmask (i.e. 8 bit)");
    ASSERT((Bcsr->ncols % 8) == 0, "The rows of B must divide the bit in a world of the bitmask (i.e. 8 bit)");

    int k = Acsc->ncols;
    int mask_size = ((k%MASK_SIZE)==0) ? (k/MASK_SIZE) : ((k/MASK_SIZE)+1) ;
    int8_t *A_map = SpaComm::gen_bitmask(Acsc->ptr_vec, Acsc->ncols, MASK_SIZE);
    int8_t *B_map = SpaComm::gen_bitmask(Bcsr->ptr_vec, Bcsr->nrows, MASK_SIZE);
/*
    // ---------- Ghatering all the required maps ----------
    int8_t *recv_A_maps;
    CUDA_CHECK( cudaMalloc(&recv_A_maps, sizeof(int8_t)*mask_size*(grid->row_size)) );
    MPI_Allgather(A_map, mask_size, MPI_INT8_T, recv_A_maps, mask_size, MPI_INT8_T, grid->row_comm);

    int8_t *recv_B_maps;
    CUDA_CHECK( cudaMalloc(&recv_B_maps, sizeof(int8_t)*mask_size*(grid->col_size)) );
    MPI_Allgather(B_map, mask_size, MPI_INT8_T, recv_B_maps, mask_size, MPI_INT8_T, grid->col_comm);

    // ---------- Performing the mask intersection ----------
    // since mapsizes are equal, we can comput all the intersections togheter
    int8_t *all_intersections = SpaComm::intersect_bitmasks(recv_A_maps, recv_B_maps, mask_size * grid->row_size); // grid->row_size == grid->col_size

    // ---------- Alltoall data sisplacement back ----------
    // *col_filters = (int8_t*)malloc(sizeof(int8_t)*mask_size*(grid->row_size));
    CUDA_CHECK( cudaMalloc(col_filters, sizeof(int8_t)*mask_size*(grid->row_size)) );
    MPI_Alltoall(all_intersections, mask_size, MPI_INT8_T, *col_filters, mask_size, MPI_INT8_T, grid->row_comm);

    // *row_filters = (int8_t*)malloc(sizeof(int8_t)*mask_size*(grid->col_size));
    CUDA_CHECK( cudaMalloc(row_filters, sizeof(int8_t)*mask_size*(grid->col_size)) );
    MPI_Alltoall(all_intersections, mask_size, MPI_INT8_T, *row_filters, mask_size, MPI_INT8_T, grid->col_comm);

#ifdef DEBUG_SPCOMM
    {
        int size = mask_size*(grid->row_size);
        int8_t *h_A_map             = (int8_t*)malloc(sizeof(int8_t)*mask_size);
        int8_t *h_B_map             = (int8_t*)malloc(sizeof(int8_t)*mask_size);
        int8_t *h_recv_A_maps       = (int8_t*)malloc(sizeof(int8_t)*size);
        int8_t *h_recv_B_maps       = (int8_t*)malloc(sizeof(int8_t)*size);
        int8_t *h_all_intersections = (int8_t*)malloc(sizeof(int8_t)*size);
        int8_t *h_col_filters       = (int8_t*)malloc(sizeof(int8_t)*size);
        int8_t *h_row_filters       = (int8_t*)malloc(sizeof(int8_t)*size);
        CUDA_CHECK(cudaMemcpy(h_A_map,             A_map,             sizeof(int8_t)*mask_size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(h_B_map,             B_map,             sizeof(int8_t)*mask_size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(h_recv_A_maps,       recv_A_maps,       sizeof(int8_t)*size,      cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(h_recv_B_maps,       recv_B_maps,       sizeof(int8_t)*size,      cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(h_all_intersections, all_intersections, sizeof(int8_t)*size,      cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(h_col_filters,       *col_filters,      sizeof(int8_t)*size,      cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(h_row_filters,       *row_filters,      sizeof(int8_t)*size,      cudaMemcpyDeviceToDevice));

        MPI_ALL_PRINT(
            fprintf(fp, "C[%d,%d] %20s: ", grid->col_rank, grid->row_rank, "local A_map");
            printBit_left2right(h_A_map, mask_size, fp);
            fprintf(fp, "\nC[%d,%d] %20s: ", grid->col_rank, grid->row_rank, "local B_map");
            printBit_left2right(h_B_map, mask_size, fp);
            fprintf(fp, "\n\n");

            fprintf(fp, "C[%d,%d] %20s: ", grid->col_rank, grid->row_rank, "recv A bitmasks");
            printBit_left2right(h_recv_A_maps, size, fp);
            fprintf(fp, "\nC[%d,%d] %20s: ", grid->col_rank, grid->row_rank, "recv B bitmasks");
            printBit_left2right(h_recv_B_maps, size, fp);
            fprintf(fp, "\n");

            fprintf(fp, "C[%d,%d] %20s: ", grid->col_rank, grid->row_rank, "h_all_intersections");
            printBit_left2right(h_all_intersections, size, fp);
            fprintf(fp, "\n\n");

            fprintf(fp, "C[%d,%d] %20s: ", grid->col_rank, grid->row_rank, "h_col_filters");
            printBit_left2right(h_col_filters, size, fp);
            fprintf(fp, "\nC[%d,%d] %20s: ", grid->col_rank, grid->row_rank, "h_row_filters");
            printBit_left2right(h_row_filters, size, fp);
            fprintf(fp, "\n");
        )

        free(h_all_intersections);
        free(h_recv_A_maps);
        free(h_recv_B_maps);
        free(h_col_filters);
        free(h_row_filters);
        free(h_A_map);
        free(h_B_map);
    }
#endif

    CUDA_CHECK(cudaFree(all_intersections));
    CUDA_CHECK(cudaFree(recv_A_maps));
    CUDA_CHECK(cudaFree(recv_B_maps));
    CUDA_CHECK(cudaFree(A_map));
    CUDA_CHECK(cudaFree(B_map));
    */
}

// ----------------------------------------------------------------------------------------------
//                              Funtions to perform the data compression
// ----------------------------------------------------------------------------------------------

struct KeepFlagByIndex
{
    const int* s;
    int m;
    const int* c;

    __host__ __device__
    KeepFlagByIndex(const int* s_, int m_, const int* c_)
        : s(s_), m(m_), c(c_) {}

    __device__ __forceinline__
    int locate_segment_by_pos(int pos) const {
        int low = 0, high = m;
        while (low < high) {
            int mid = (low + high) >> 1;
            if (s[mid] <= pos) low = mid + 1;
            else high = mid;
        }
        return low - 1;
    }

    __device__ __forceinline__
    int operator()(int pos) const {
        int seg = locate_segment_by_pos(pos);
        if (seg < 0) return 0; // skip if out of range
        int word = seg / MASK_SIZE;
        int bit  = seg % MASK_SIZE;
        return ((c[word] >> bit) & 1) ? 1 : 0;
    }
};

struct MaskedTransform
{
    const int* c; // bitmask

    __host__ __device__
    MaskedTransform(const int* c_) : c(c_) {}

    __device__ __forceinline__
    int operator()(const thrust::tuple<int,int>& t) const {
        int idx = thrust::get<0>(t);
        int val = thrust::get<1>(t);
        int word = idx / MASK_SIZE;
        int bit  = idx % MASK_SIZE;
        return ( (c[word] >> bit) & 1 ) ? val : 0;
    }
};


template<typename VT, typename PT> // Vector type (usually int if idx or float if val) and ptr type
int select_entries(VT* input_vec, int n, PT* ptr_vec, int m, PT* mask, VT **output) {

    auto counting = thrust::make_counting_iterator<int>(0);
    auto flags_it = thrust::make_transform_iterator(
        counting,
        KeepFlagByIndex(ptr_vec, m, mask)
    );

    VT  *d_out;
    int *d_num_selected;
    cudaMalloc(&d_out, sizeof(VT)*n);
    cudaMalloc(&d_num_selected, sizeof(int));

    // Temp storage
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes,
        input_vec, // values
        flags_it,                             // lazy flags
        d_out,
        d_num_selected,
        n
    );

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes,
        input_vec, // values
        flags_it,                             // lazy flags
        d_out,
        d_num_selected,
        n
    );

    // Move num_selected on host
    int num_selected;
    cudaMemcpy(&num_selected, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost);

#ifdef DEBUG
    VT* h_out = (VT*)malloc(sizeof(VT)*num_selected);
    cudaMemcpy(h_out, d_out, sizeof(VT)*num_selected, cudaMemcpyDeviceToHost);

    std::cout << "Selected " << num_selected << " values:\n";
    for (int i=0; i<num_selected; i++) std::cout << h_out[i] << " ";
    std::cout << "\n";
    free(h_out);
#endif

    cudaFree(d_temp_storage);

    *output = d_out;
    return(num_selected);
}

template<typename IT>
IT* select_ptrs(IT* raw_ptr, int m, IT* mask) {

    rowptrs_to_rownnz<IT>(raw_ptr, m-1); // The function require the number of rows/cols, not the real vecsize

    int mask_size = (((m-1)%MASK_SIZE)==0) ? ((m-1)/MASK_SIZE) : (((m-1)/MASK_SIZE)+1) ;
    thrust::device_vector<IT> d_mask(mask, mask + mask_size);
    thrust::device_vector<IT> d_row(raw_ptr, raw_ptr + m);

#ifdef DEBUG
    thrust::host_vector<IT> h_row = d_row;
    std::cout << "Nnz ptr_vec: ";
    for (int x : h_row) std::cout << x << " ";
    std::cout << "\n";
#endif

    auto counting = thrust::make_counting_iterator<int>(0);
    auto zipped = thrust::make_zip_iterator(thrust::make_tuple(counting, d_row.begin()+1));

    thrust::transform(
        zipped, zipped + (m-1),
        d_row.begin()+1,
        MaskedTransform(thrust::raw_pointer_cast(d_mask.data()))
    );

#ifdef DEBUG
    h_row = d_row; // Update the host mirror
    std::cout << "Masked ptr_vec: ";
    for (int x : h_row) std::cout << x << " ";
    std::cout << "\n";
#endif

    raw_ptr = thrust::raw_pointer_cast(d_row.data());
    rownnz_to_rowptrs<int>(raw_ptr, m-1); // The function require the number of rows/cols, not the real vecsize

#ifdef DEBUG
    h_row = d_row; // Update the host mirror
    std::cout << "New ptr_vec: ";
    for (int x : h_row) std::cout << x << " ";
    std::cout << "\n";
#endif

    IT *output;
    cudaMalloc(&output, sizeof(IT)*m);
    cudaMemcpy(output, raw_ptr, m*sizeof(IT), cudaMemcpyDeviceToDevice);
    return(output);
}

template <typename IT, typename VT>
using KWrapDMat = typename KokkosWrap::DistribuitedMatrix<IT, IT, VT>;

template <typename IT, typename VT>
struct SpaCommHandler
{

    SpaCommHandler(KWrapDMat<IT, VT>& dist_A, KWrapDMat<IT, VT>& dist_B)
    {

        grid = dist_A.partitioning->grid;
        assert(grid->row_size == grid->col_size);
        int grid_dim = grid->row_size;

        ASSERT(dist_A.mmio_csx->ncols == dist_B.mmio_csx->nrows, "A cols must be equal to B rows");

        int cdim = dist_A.mmio_csx->ncols;
        mask_len = ((cdim%MASK_SIZE)==0) ? (cdim/MASK_SIZE) : ((cdim/MASK_SIZE)+1) ;
        spcomm_2D(dist_A.mmio_csx, dist_B.mmio_csx, grid, &A_column_filters, &B_rows_filters);

        // ----- debug -----
        /*{
            size_t size = (mask_len)*(dist_A.partitioning->grid->row_size);
            int8_t* hcolmaps = (int8_t*)malloc(sizeof(int8_t)*size);
            CUDA_CHECK(cudaMemcpy(hcolmaps, A_column_filters, size, cudaMemcpyDeviceToHost));
            SpaComm::printBit_left2right(hcolmaps, size, stdout);
            free(hcolmaps);
        }*/
        // -----------------

    }

    mmio::CSX<IT,VT>* Compress (KWrapDMat<IT, VT>& dist_M, int iteration_number) {

        ASSERT(iteration_number < grid->row_comm, "ERROR: provided an invalid iteration number");

        mmio::CSX<IT,VT> *M = dist_M->mmio_csx;
        mmio::MajorDim layout = M->majordim;

        // Set-up parameeters according to A or B operand
        int ptr_size;
        int8_t *mask;
        if (layout == mmio::MajorDim::ROWS) {
            ptr_size = M->nrows+1;
            mask = B_rows_filters + mask_len*iteration_number;
        } else {
            ptr_size = M->ncols+1;
            mask = A_column_filters + mask_len*iteration_number;
        }

        // Compute compressed index vector
        IT* new_idx;
        int num_selected = select_entries<IT, IT>(
                M->idx_vec, M->nnz,
                M->ptr_vec, ptr_size,
                mask,
                &new_idx
        );

        // Compute compressed value vector
        VT* new_val;
        int num_selected_val = select_entries<VT, IT>(
                M->val, M->nnz,
                M->ptr_vec, ptr_size,
                mask,
                &new_val
        );

        // Check
        if (num_selected != num_selected_val) {
            fprintf(stderr, "Error: num_selected_val (%d) differ from num_selected (%d)!\n", num_selected_val, num_selected);
            MPI_Abort(grid->world_comm, __LINE__);
        }

        // Compute compressed pointer vector
        int *new_row = select_ptrs(M->ptr_vec, ptr_size, mask);

        // Wrapping results in the output csx
        mmio::CSX<IT,VT> *output = (mmio::CSX<IT,VT>*)malloc(sizeof(mmio::CSX<IT,VT>));
        output->majordim = M->majordim;
        output->nnz      = num_selected;
        output->nrows    = M->nrows;
        output->ncols    = M->ncols;
        output->val      = new_val;
        output->ptr_vec  = new_row;
        output->idx_vec  = new_idx;

        return(output);
    }


    ~SpaCommHandler()
    {
        CUDA_FREE_SAFE(A_column_filters);
        CUDA_FREE_SAFE(B_rows_filters);
    }

    // Communicator grid
    dmmio::ProcessGrid * grid;

    // These two vectors are device vectors that the process will use to compress local tiles before the communication
    int8_t* A_column_filters; // Filters to be applied to A(i,j) matrix before than send to Process (i,k)
    int8_t* B_rows_filters;   // Filters to be applied to B(i,j) matrix before than send to Process (k,j)

    IT mask_len;  // This is the len of every single mask (i.e. A-B common dimension / MASK_SIZE)

};


}
