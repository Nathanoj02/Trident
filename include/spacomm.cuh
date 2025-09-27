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

// #define DEBUG_SPCOMM
// #define DEBUG_COMPRESSION
// #define DEBUG_PTR_COMPRESS

namespace SpaComm
{

// ----------------------------------------------------------------------------------------------
// Funtions for the sparsity pattern communication (generate, communicate and intersect bitmasks)
// ----------------------------------------------------------------------------------------------

// A general function to compute result vector given a row/col pointer array
// A general function to compute result vector given a row/col pointer array
template<typename IT>
BMASK_TYPE* gen_bitmask(const IT* ptr_d, int n, int mask_size) {
    // Each entry is a byte that might hold one bit
    thrust::device_vector<int> presult(n);

    // Mark positions: 1 << (i % mask_size) if row is non-empty
    thrust::transform(
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(n),
        presult.begin(),
        [=] __device__(int i) {
            IT a = ptr_d[i];
            IT b = ptr_d[i + 1];
            int c = 0;
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
    int *result_int;
    CUDA_CHECK(cudaMalloc(&result_int, sizeof(int) * num_segments));

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
    CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage, temp_storage_bytes,
        presult.begin(), result_int,
        num_segments, d_offsets.data().get(), d_offsets.data().get() + 1,
        op, initial_value
    ));

    // Allocate temp storage
    // thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    // d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Run reduction
    CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(
        d_temp_storage, temp_storage_bytes,
        presult.begin(), result_int,
        num_segments, d_offsets.data().get(), d_offsets.data().get() + 1,
        op, initial_value
    ));

    CUDA_CHECK(cudaFree(d_temp_storage));

    BMASK_TYPE* result_bytes;
    cudaMalloc(&result_bytes, sizeof(BMASK_TYPE) * num_segments);

// --------------------------
// NOTE: I don't know why but the device cast not works, if someone is able to fix it I will offer him/her/it a beer
    // thrust::transform(
    //     result_int, result_int + num_segments,
    //     thrust::device_pointer_cast(result_bytes),
    //     [] __device__ (int v) { return static_cast<BMASK_TYPE>(v); }
    // );
// >>> test bug fix >>>>
    int    *h_result_int   = (int*)   malloc(sizeof(int)    * num_segments);
    BMASK_TYPE *h_result_bytes = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE) * num_segments);
    CUDA_CHECK(cudaMemcpy(h_result_int, result_int, sizeof(int)*num_segments, cudaMemcpyDeviceToHost));
    for (int i=0; i<num_segments; i++) h_result_bytes[i] = static_cast<BMASK_TYPE>(h_result_int[i]);
    CUDA_CHECK(cudaMemcpy(result_bytes, h_result_bytes, sizeof(BMASK_TYPE)*num_segments, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaFree(result_int));
    free(h_result_bytes);
    free(h_result_int);
// --------------------------

    return result_bytes;
}

template<typename IT>
BMASK_TYPE* gen_bytemask(const IT* ptr_d, int n) {
    // Each entry is a byte that might hold one bit
    thrust::device_vector<BMASK_TYPE> presult(n);

    thrust::transform(
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(n),
        presult.begin(),
        [=] __device__(int i) {
            IT a = ptr_d[i];
            IT b = ptr_d[i + 1];
            BMASK_TYPE c = 0;
            if (a != b) {
                c = 1;
            }
            return c;
        });

    BMASK_TYPE *result;
    CUDA_CHECK(cudaMalloc(&result, sizeof(BMASK_TYPE)*n));
    CUDA_CHECK(cudaMemcpy(result, thrust::raw_pointer_cast(presult.data()), sizeof(BMASK_TYPE)*n, cudaMemcpyDeviceToDevice));

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

inline void printBit_left2right(BMASK_TYPE* bitmask, int nwords, FILE* fp=stdout) {
    for (int i=0; i<nwords; i++) {
        for (int j=0; j<MASK_SIZE; j++) {
            fprintf(fp, "%c", (bitmask[i] & (1<<j)) ? '1' : '0');
        }
        if (i!=(nwords-1)) fprintf(fp, "|");
    }
}

struct LogicAnd {
    __device__ __host__ int operator()(const thrust::tuple<int,int>& t) const {
        return thrust::get<0>(t) && thrust::get<1>(t);
    }
};

template<typename T>
T* intersect_bytemasks(const T* a, const T* b, int n) {
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
    auto transform_iter = thrust::make_transform_iterator(zipped_begin, LogicAnd());

    // Copy results into d_out
    thrust::copy(transform_iter, transform_iter + n, out_begin);

    return d_out; // caller must cudaFree()
}

template<typename IT, typename VT>
int spcomm_2D (mmio::CSX<IT,VT> *Acsc, mmio::CSX<IT,VT> *Bcsr, dmmio::ProcessGrid *grid,
                BMASK_TYPE** col_filters, BMASK_TYPE** row_filters) {
    ASSERT(Acsc->majordim==mmio::MajorDim::COLS, "A must be a CSC");
    ASSERT(Bcsr->majordim==mmio::MajorDim::ROWS, "B must be a CSR");
    ASSERT(Acsc->ncols == Bcsr->nrows, "In 2D mask cols of A must be equal to rows of B");
    ASSERT(grid->row_size == grid->col_size, "The 2D grid must be a square");
    ASSERT(grid->node_size == 1, "The spcomm 2D do not support node size > 1");
    ASSERT((Acsc->ncols % 8) == 0, "The columns of A must divide the bit in a world of the bitmask (i.e. 8 bit)");
    ASSERT((Bcsr->ncols % 8) == 0, "The rows of B must divide the bit in a world of the bitmask (i.e. 8 bit)");

    int k = Acsc->ncols;
    int mask_size = ((k%MASK_SIZE)==0) ? (k/MASK_SIZE) : ((k/MASK_SIZE)+1) ;
    BMASK_TYPE *A_map = SpaComm::gen_bitmask(Acsc->ptr_vec, Acsc->ncols, MASK_SIZE);
    BMASK_TYPE *B_map = SpaComm::gen_bitmask(Bcsr->ptr_vec, Bcsr->nrows, MASK_SIZE);

    // ---------- Ghatering all the required maps ----------
    BMASK_TYPE *recv_A_maps;
    CUDA_CHECK( cudaMalloc(&recv_A_maps, sizeof(BMASK_TYPE)*mask_size*(grid->row_size)) );
    MPI_Allgather(A_map, sizeof(BMASK_TYPE)*mask_size, MPI_BYTE, recv_A_maps, sizeof(BMASK_TYPE)*mask_size, MPI_BYTE, grid->row_comm);

    BMASK_TYPE *recv_B_maps;
    CUDA_CHECK( cudaMalloc(&recv_B_maps, sizeof(BMASK_TYPE)*mask_size*(grid->col_size)) );
    MPI_Allgather(B_map, sizeof(BMASK_TYPE)*mask_size, MPI_BYTE, recv_B_maps, sizeof(BMASK_TYPE)*mask_size, MPI_BYTE, grid->col_comm);

    // ---------- Performing the mask intersection ----------
    // since mapsizes are equal, we can comput all the intersections togheter
    BMASK_TYPE *all_intersections = SpaComm::intersect_bitmasks(recv_A_maps, recv_B_maps, mask_size * grid->row_size); // grid->row_size == grid->col_size

    // ---------- Alltoall data sisplacement back ----------
    // *col_filters = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*mask_size*(grid->row_size));
    CUDA_CHECK( cudaMalloc(col_filters, sizeof(BMASK_TYPE)*mask_size*(grid->row_size)) );
    MPI_Alltoall(all_intersections, sizeof(BMASK_TYPE)*mask_size, MPI_BYTE, *col_filters, sizeof(BMASK_TYPE)*mask_size, MPI_BYTE, grid->row_comm);

    // *row_filters = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*mask_size*(grid->col_size));
    CUDA_CHECK( cudaMalloc(row_filters, sizeof(BMASK_TYPE)*mask_size*(grid->col_size)) );
    MPI_Alltoall(all_intersections, sizeof(BMASK_TYPE)*mask_size, MPI_BYTE, *row_filters, sizeof(BMASK_TYPE)*mask_size, MPI_BYTE, grid->col_comm);

#ifdef DEBUG_SPCOMM
    {
        int size = mask_size*(grid->row_size);
        BMASK_TYPE *h_A_map             = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*mask_size);
        BMASK_TYPE *h_B_map             = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*mask_size);
        BMASK_TYPE *h_recv_A_maps       = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*size);
        BMASK_TYPE *h_recv_B_maps       = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*size);
        BMASK_TYPE *h_all_intersections = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*size);
        BMASK_TYPE *h_col_filters       = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*size);
        BMASK_TYPE *h_row_filters       = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*size);
        CUDA_CHECK(cudaMemcpy(h_A_map,             A_map,             sizeof(BMASK_TYPE)*mask_size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(h_B_map,             B_map,             sizeof(BMASK_TYPE)*mask_size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(h_recv_A_maps,       recv_A_maps,       sizeof(BMASK_TYPE)*size,      cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(h_recv_B_maps,       recv_B_maps,       sizeof(BMASK_TYPE)*size,      cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(h_all_intersections, all_intersections, sizeof(BMASK_TYPE)*size,      cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(h_col_filters,       *col_filters,      sizeof(BMASK_TYPE)*size,      cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(h_row_filters,       *row_filters,      sizeof(BMASK_TYPE)*size,      cudaMemcpyDeviceToDevice));

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

    return(mask_size);
}

template<typename IT, typename VT>
void spcomm_2D_bytemask (mmio::CSX<IT,VT> *Acsc, mmio::CSX<IT,VT> *Bcsr, dmmio::ProcessGrid *grid,
                BMASK_TYPE** col_filters, BMASK_TYPE** row_filters) {
    ASSERT(Acsc->majordim==mmio::MajorDim::COLS, "A must be a CSC");
    ASSERT(Bcsr->majordim==mmio::MajorDim::ROWS, "B must be a CSR");
    ASSERT(Acsc->ncols == Bcsr->nrows, "In 2D mask cols of A must be equal to rows of B");
    ASSERT(grid->row_size == grid->col_size, "The 2D grid must be a square");
    ASSERT(grid->node_size == 1, "The spcomm 2D do not support node size > 1");
    ASSERT((Acsc->ncols % 8) == 0, "The columns of A must divide the bit in a world of the bitmask (i.e. 8 bit)");
    ASSERT((Bcsr->ncols % 8) == 0, "The rows of B must divide the bit in a world of the bitmask (i.e. 8 bit)");

    int k = Acsc->ncols;
    int mask_size = k ;
    BMASK_TYPE *A_map = SpaComm::gen_bytemask(Acsc->ptr_vec, Acsc->ncols);
    BMASK_TYPE *B_map = SpaComm::gen_bytemask(Bcsr->ptr_vec, Bcsr->nrows);

    // ---------- Ghatering all the required maps ----------
    BMASK_TYPE *recv_A_maps;
    CUDA_CHECK( cudaMalloc(&recv_A_maps, sizeof(BMASK_TYPE)*mask_size*(grid->row_size)) );
    MPI_Allgather(A_map, sizeof(BMASK_TYPE)*mask_size, MPI_BYTE, recv_A_maps, sizeof(BMASK_TYPE)*mask_size, MPI_BYTE, grid->row_comm);

    BMASK_TYPE *recv_B_maps;
    CUDA_CHECK( cudaMalloc(&recv_B_maps, sizeof(BMASK_TYPE)*mask_size*(grid->col_size)) );
    MPI_Allgather(B_map, sizeof(BMASK_TYPE)*mask_size, MPI_BYTE, recv_B_maps, sizeof(BMASK_TYPE)*mask_size, MPI_BYTE, grid->col_comm);

    // ---------- Performing the mask intersection ----------
    // since mapsizes are equal, we can comput all the intersections togheter
    BMASK_TYPE *all_intersections = SpaComm::intersect_bytemasks(recv_A_maps, recv_B_maps, mask_size * grid->row_size); // grid->row_size == grid->col_size

    // ---------- Alltoall data sisplacement back ----------
    // *col_filters = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*mask_size*(grid->row_size));
    CUDA_CHECK( cudaMalloc(col_filters, sizeof(BMASK_TYPE)*mask_size*(grid->row_size)) );
    MPI_Alltoall(all_intersections, sizeof(BMASK_TYPE)*mask_size, MPI_BYTE, *col_filters, sizeof(BMASK_TYPE)*mask_size, MPI_BYTE, grid->row_comm);

    // *row_filters = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*mask_size*(grid->col_size));
    CUDA_CHECK( cudaMalloc(row_filters, sizeof(BMASK_TYPE)*mask_size*(grid->col_size)) );
    MPI_Alltoall(all_intersections, sizeof(BMASK_TYPE)*mask_size, MPI_BYTE, *row_filters, sizeof(BMASK_TYPE)*mask_size, MPI_BYTE, grid->col_comm);

#ifdef DEBUG_SPCOMM
    {
        int size = mask_size*(grid->row_size);
        BMASK_TYPE *h_A_map             = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*mask_size);
        BMASK_TYPE *h_B_map             = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*mask_size);
        BMASK_TYPE *h_recv_A_maps       = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*size);
        BMASK_TYPE *h_recv_B_maps       = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*size);
        BMASK_TYPE *h_all_intersections = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*size);
        BMASK_TYPE *h_col_filters       = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*size);
        BMASK_TYPE *h_row_filters       = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*size);
        CUDA_CHECK(cudaMemcpy(h_A_map,             A_map,             sizeof(BMASK_TYPE)*mask_size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(h_B_map,             B_map,             sizeof(BMASK_TYPE)*mask_size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(h_recv_A_maps,       recv_A_maps,       sizeof(BMASK_TYPE)*size,      cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(h_recv_B_maps,       recv_B_maps,       sizeof(BMASK_TYPE)*size,      cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(h_all_intersections, all_intersections, sizeof(BMASK_TYPE)*size,      cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(h_col_filters,       *col_filters,      sizeof(BMASK_TYPE)*size,      cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(h_row_filters,       *row_filters,      sizeof(BMASK_TYPE)*size,      cudaMemcpyDeviceToDevice));

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
}

// ----------------------------------------------------------------------------------------------
//                              Funtions to perform the data compression
// ----------------------------------------------------------------------------------------------

struct KeepFlagByIndex
{
    const int* s;
    int m;
    const BMASK_TYPE* c;

    __host__ __device__
    KeepFlagByIndex(const int* s_, int m_, const BMASK_TYPE* c_)
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
    int operator()(int pos) const{
        int seg = locate_segment_by_pos(pos);
        if (seg < 0) return 0; // skip if out of range
        int word = seg / MASK_SIZE;
        int bit  = seg % MASK_SIZE;
        return ((c[word] >> bit) & 1) ? 1 : 0;
    }
};

struct MaskedTransform
{
    const BMASK_TYPE* c; // bitmask

    __host__ __device__
    MaskedTransform(const BMASK_TYPE* c_) : c(c_) {}

    __device__ __forceinline__
    int operator()(const thrust::tuple<int,int>& t) const {
        int idx = thrust::get<0>(t);
        int val = thrust::get<1>(t);
        int word = idx / MASK_SIZE;
        int bit  = idx % MASK_SIZE;
        return ( (c[word] >> bit) & 1 ) ? val : 0;
    }
};

#define EXPLICIT_FLAGS

template<typename VT, typename PT> // Vector type (usually int if idx or float if val) and ptr type
int select_entries(const VT* input_vec, int n, const PT* ptr_vec, int m, const BMASK_TYPE* mask, VT **output, cudaStream_t stream = 0) {

#ifndef EXPLICIT_FLAGS
    auto counting = thrust::make_counting_iterator<int>(0);
    auto flags_it = thrust::make_transform_iterator(
        counting,
        KeepFlagByIndex(ptr_vec, m, mask)
    );
#else
    thrust::device_vector<unsigned char> d_flags(n);

    // counting iterator + transform -> flags
    auto counting = thrust::make_counting_iterator<int>(0);
    // KeepFlagByIndex must be __host__ __device__ and must accept device pointers
    thrust::transform(
        counting, counting + n,
        d_flags.begin(),
        KeepFlagByIndex(ptr_vec, m, mask) // must be device-callable and ptr_vec/mask must be device pointers
    );
    auto flags_it = thrust::raw_pointer_cast(d_flags.data());
#endif

    VT  *d_out;
    int *d_num_selected;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(VT)*n));
    CUDA_CHECK(cudaMalloc(&d_num_selected, sizeof(int)));

    // Temp storage
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CUDA_CHECK(cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes,
        input_vec, // values
        flags_it,                             // lazy flags
        d_out,
        d_num_selected,
        n
    ));

    if (temp_storage_bytes>0) { CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes)); }

    CUDA_CHECK(cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes,
        input_vec, // values
        flags_it,                             // lazy flags
        d_out,
        d_num_selected,
        n
    ));

    // Move num_selected on host
    int num_selected;
    CUDA_CHECK(cudaMemcpy(&num_selected, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost));

#ifdef DEBUG
    VT* h_out = (VT*)malloc(sizeof(VT)*num_selected);
    cudaMemcpy(h_out, d_out, sizeof(VT)*num_selected, cudaMemcpyDeviceToHost);

    std::cout << "Selected " << num_selected << " values:\n";
    for (int i=0; i<num_selected; i++) std::cout << h_out[i] << " ";
    std::cout << "\n";
    free(h_out);
#endif

    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_num_selected));

    *output = d_out;
    return(num_selected);
}

template<typename IT>
IT* select_ptrs(IT* raw_ptr, int m, BMASK_TYPE* mask, cudaStream_t stream = 0) {

#ifdef DEBUG_PTR_COMPRESS
    print_d_arr(raw_ptr, m, "input ptr: ");
#endif

    rowptrs_to_rownnz<IT>(raw_ptr, m-1); // The function require the number of rows/cols, not the real vecsize

#ifdef DEBUG_PTR_COMPRESS
    print_d_arr(raw_ptr, m, "rownnz ptr: ");
#endif

    int mask_size = (((m-1)%MASK_SIZE)==0) ? ((m-1)/MASK_SIZE) : (((m-1)/MASK_SIZE)+1) ;
    thrust::device_vector<BMASK_TYPE> d_mask(mask, mask + mask_size);
    // thrust::device_vector<IT> d_row(raw_ptr, raw_ptr + m);
    thrust::device_ptr<IT> thrust_ptr = thrust::device_pointer_cast(raw_ptr);


#ifdef DEBUG_PTR_COMPRESS
    std::vector<IT> h_row(m);
    cudaMemcpy(h_row.data(), raw_ptr, m * sizeof(IT), cudaMemcpyDeviceToHost);

    std::cout << "Nnz ptr_vec: ";
    for (int x : h_row) std::cout << x << " ";
    std::cout << "\n";
    fflush(stdout);
#endif

    auto counting = thrust::make_counting_iterator<int>(0);
    auto zipped = thrust::make_zip_iterator(thrust::make_tuple(counting, thrust_ptr+1));

    thrust::transform(
        zipped, zipped + (m-1),
        thrust_ptr+1,
        MaskedTransform(thrust::raw_pointer_cast(d_mask.data()))
    );

#ifdef DEBUG_PTR_COMPRESS
    cudaMemcpy(h_row.data(), raw_ptr, m * sizeof(IT), cudaMemcpyDeviceToHost);

    std::cout << "Masked ptr_vec: ";
    for (int x : h_row) std::cout << x << " ";
    std::cout << "\n";
    fflush(stdout);
#endif

    // raw_ptr = thrust::raw_pointer_cast(d_row.data());
    rownnz_to_rowptrs<int>(raw_ptr, m-1); // The function require the number of rows/cols, not the real vecsize

#ifdef DEBUG_PTR_COMPRESS
    cudaMemcpy(h_row.data(), raw_ptr, m * sizeof(IT), cudaMemcpyDeviceToHost);

    std::cout << "New ptr_vec: ";
    for (int x : h_row) std::cout << x << " ";
    std::cout << "\n";
    fflush(stdout);
#endif

    IT *output;
    CUDA_CHECK(cudaMalloc(&output, sizeof(IT)*m));
    CUDA_CHECK(cudaMemcpy(output, raw_ptr, m*sizeof(IT), cudaMemcpyDeviceToDevice));
    return(output);
}

template <typename IT, typename VT>
using KWrapDMat = typename KokkosWrap::DistribuitedMatrix<IT, IT, VT>;

template <typename IT, typename VT>
struct SpaCommHandler
{

    SpaCommHandler(mmio::CSX<IT,VT>* csx_A, mmio::CSX<IT,VT>* csx_B, dmmio::ProcessGrid* input_grid)
    {

        ASSERT(input_grid->row_size == input_grid->col_size, "Process grid must be squared");
        ASSERT(csx_A->ncols == csx_B->nrows, "A cols must be equal to B rows");

        grid     = input_grid;
        nfilters = input_grid->row_size;
        mask_len = spcomm_2D(csx_A, csx_B, input_grid, &A_column_filters, &B_rows_filters);

    }

    mmio::CSX<IT,VT>* Compress (const mmio::CSX<IT,VT> *M, int iteration_number, cudaStream_t stream = 0) {

        ASSERT(iteration_number < nfilters, "ERROR: provided an invalid iteration number");
        mmio::MajorDim layout = M->majordim;

        // Set-up parameeters according to A or B operand
        int ptr_size;
        BMASK_TYPE *mask;
        if (layout == mmio::MajorDim::ROWS) {
            ptr_size = M->nrows+1;
            mask = B_rows_filters + mask_len*iteration_number;
        } else {
            ptr_size = M->ncols+1;
            mask = A_column_filters + mask_len*iteration_number;
        }

#ifdef DEBUG_COMPRESSION
        char matchar = (layout == mmio::MajorDim::ROWS) ? ('B') : ('A') ;
        BMASK_TYPE *h_mask = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*mask_len);
        CHECK_CUDA(cudaMemcpy(h_mask, mask, sizeof(BMASK_TYPE) * mask_len, cudaMemcpyDeviceToHost));
        MPI_ALL_PRINT(
            fprintf(fp, "Entered in compression with iterid %d\n", iteration_number);
            fprintf(fp, "Matrix %c has mask: ", matchar);
            SpaComm::printBit_left2right(h_mask, mask_len, fp);
            fprintf(fp, "\n");
        )
        free(h_mask);
#endif

        // Compute compressed index vector
        IT* new_idx;
        int num_selected = select_entries<IT, IT>(
                M->idx_vec, M->nnz,
                M->ptr_vec, ptr_size,
                mask,
                &new_idx
        );

#ifdef DEBUG_COMPRESSION
        if (grid->global_rank==0 && layout == mmio::MajorDim::COLS) {
            print_d_arr(M->idx_vec, M->nnz,       "Old idx: ");
            print_d_arr(new_idx,    num_selected, "New idx: ");
        }
#endif

        // Compute compressed value vector
        VT* new_val;
        int num_selected_val = select_entries<VT, IT>(
                M->val, M->nnz,
                M->ptr_vec, ptr_size,
                mask,
                &new_val
        );

#ifdef DEBUG_COMPRESSION
        if (grid->global_rank==0 && layout == mmio::MajorDim::COLS) {
            print_d_arr(M->val,  M->nnz,           "Old val: ");
            print_d_arr(new_val, num_selected_val, "New val: ");
        }
#endif

        // Check
        if (num_selected != num_selected_val) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            fprintf(stderr, "Error rank %d iteration_number %d: num_selected_val (%d) differ from num_selected (%d)!\n",
                    rank, iteration_number, num_selected_val, num_selected);
            fprintf(stderr, "M->nnz: %d, M->val: %p, M->idx: %p", M->nnz, M->val, M->idx_vec);

            BMASK_TYPE *tmp;
            CUDA_CHECK(cudaMalloc(&tmp, sizeof(BMASK_TYPE)*mask_len));
            CUDA_CHECK(cudaMemcpy(tmp, mask, sizeof(BMASK_TYPE)*mask_len, cudaMemcpyDeviceToHost));
            SpaComm::printBit_left2right(tmp, mask_len, stderr);
            MPI_Abort(grid->world_comm, __LINE__);
        }

        // Compute compressed pointer vector
        IT *new_row;
        CUDA_CHECK(cudaMalloc(&new_row, sizeof(IT)*ptr_size));
        CUDA_CHECK(cudaMemcpy(new_row, M->ptr_vec, sizeof(IT)*ptr_size, cudaMemcpyDeviceToDevice));
        new_row = select_ptrs(new_row, ptr_size, mask); // Changes are done in place

#ifdef DEBUG_COMPRESSION
        if (grid->global_rank==0 && layout == mmio::MajorDim::COLS) {
            print_d_arr(M->ptr_vec, ptr_size, "Old ptr: ");
            print_d_arr(new_row,    ptr_size, "New ptr: ");
        }
#endif

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
    BMASK_TYPE* A_column_filters; // Filters to be applied to A(i,j) matrix before than send to Process (i,k)
    BMASK_TYPE* B_rows_filters;   // Filters to be applied to B(i,j) matrix before than send to Process (k,j)

    IT mask_len;  // This is the len of every single mask (i.e. A-B common dimension / MASK_SIZE)
    IT nfilters;  // This is the number of filters in each filter collection (i.e. A_column_filters, B_column_filters)

};


}
