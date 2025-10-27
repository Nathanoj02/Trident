#pragma once
#include "common.h"
#include "KokkosWrap.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/system/cuda/execution_policy.h>
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

// ----------------------------------------------------------------------------------------------
//                              Funtions to perform the data compression
// ----------------------------------------------------------------------------------------------

// #define KOKKOS_TEST

template<typename T>
struct KeepFlagByIndex
{
    const T* s;
    int m;
    const BMASK_TYPE* c;

#ifdef KOKKOS_TEST
    KOKKOS_INLINE_FUNCTION
#else
    __host__ __device__
#endif
    KeepFlagByIndex(const T* s_, int m_, const BMASK_TYPE* c_)
        : s(s_), m(m_), c(c_) {}

#ifdef KOKKOS_TEST
    KOKKOS_INLINE_FUNCTION
#else
    __device__ __forceinline__
#endif
    int locate_segment_by_pos(int pos) const {
        int low = 0, high = m;
        while (low < high) {
            int mid = (low + high) >> 1;
            if (s[mid] <= static_cast<T>(pos)) low = mid + 1;
            else high = mid;
        }
        return low - 1;
    }

#ifdef KOKKOS_TEST
    KOKKOS_INLINE_FUNCTION
#else
    __device__ __forceinline__
#endif
    int operator()(int pos) const{
        int seg = locate_segment_by_pos(pos);
        if (seg < 0) return 0; // skip if out of range
        int word = seg / MASK_SIZE;
        int bit  = seg % MASK_SIZE;
        return ((c[word] >> bit) & 1) ? 1 : 0;
    }
};

#ifdef KOKKOS_TEST
    template <typename PT>
    Kokkos::View<unsigned char*> compute_flags_kokkos(
        int n,
        const PT* ptr_vec,
        int m,
        const BMASK_TYPE* mask)
    {
        Kokkos::View<unsigned char*> d_flags("flags", n);

        KeepFlagByIndex<PT> functor(ptr_vec, m, mask);

        Kokkos::parallel_for("ComputeFlags", n, KOKKOS_LAMBDA(int i) {
            d_flags(i) = static_cast<unsigned char>(functor(i));
        });

        Kokkos::fence(); // ensure flags are ready

        return d_flags;
    }
#endif

#define EXPLICIT_FLAGS

template<typename VT, typename PT> // Vector type (usually int if idx or float if val) and ptr type
int select_entries(const VT* input_vec, int n, const PT* ptr_vec, int m, const BMASK_TYPE* mask, VT **output, cudaStream_t stream = 0) {

#ifndef EXPLICIT_FLAGS
    // This not work, we have to use cub if. Find how to implement it (if we want a lazy iterator)
    auto counting = thrust::make_counting_iterator<int>(0);
    auto flags_it = thrust::make_transform_iterator(
        counting,
        KeepFlagByIndex(ptr_vec, m, mask)
    );
#else
#ifdef KOKKOS_TEST
    auto d_flags = compute_flags_kokkos(n, ptr_vec, m, mask);
#else
    thrust::device_vector<unsigned char> d_flags(n);

    auto counting = thrust::make_counting_iterator<int>(0);
    thrust::transform(
    	thrust::cuda::par.on(stream),
    	counting, counting + n,
    	d_flags.begin(),
    	KeepFlagByIndex<PT>(ptr_vec, m, mask)
    );

    unsigned char* d_flags_ptr = thrust::raw_pointer_cast(d_flags.data());
#endif
#endif

    // -------------- Just as test --------------
    // unsigned char* d_flags_ptr;
    // CUDA_CHECK(cudaMalloc(&d_flags_ptr, sizeof(unsigned char)*n));
    // CUDA_CHECK(cudaMemset(d_flags_ptr, 1U, sizeof(unsigned char)*n));
    // ------------------------------------------

    VT  *d_out;
    int *d_num_selected;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(VT)*n));
    CUDA_CHECK(cudaMalloc(&d_num_selected, sizeof(int)));

    // Temp storage
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

#ifndef EXPLICIT_FLAGS
    // TODO: put here a 'cub::DeviceSelect::If' to use a lazy iterator
#else
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes,
        input_vec,    // values
#ifdef KOKKOS_TEST
	d_flags.data(),
#else
	d_flags_ptr,  // explicit flag vector
#endif
        d_out,
        d_num_selected,
        n,
        stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));
#endif

    if (temp_storage_bytes>0) { CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes)); }

#ifndef EXPLICIT_FLAGS
    // TODO: put here a 'cub::DeviceSelect::If' to use a lazy iterator
#else
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes,
        input_vec,    // values
#ifdef KOKKOS_TEST
        d_flags.data(),
#else
        d_flags_ptr,  // explicit flag vector
#endif
        d_out,
        d_num_selected,
        n,
        stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));
#endif

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

    if (temp_storage_bytes>0) { CUDA_CHECK(cudaFree(d_temp_storage)); }

    CUDA_CHECK(cudaFree(d_num_selected));

    // -------------- Just as test --------------
    // CUDA_CHECK(cudaFree(d_flags_ptr)); // This is for the previous test
    // int num_selected = n;
    // CUDA_CHECK(cudaMemset(d_out, 1U, sizeof(VT)*n));
    // ------------------------------------------

    *output = d_out;
    return(num_selected);
}

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

template<typename IT>
IT* select_ptrs(IT* raw_ptr, int m, BMASK_TYPE* mask, cudaStream_t stream = 0, cubTmpBuff *tmp_buff = nullptr) {

#ifdef DEBUG_PTR_COMPRESS
    print_d_arr(raw_ptr, m, "input ptr: ");
#endif

    rowptrs_to_rownnz<IT>(raw_ptr, m-1, stream, tmp_buff); // The function require the number of rows/cols, not the real vecsize

#ifdef DEBUG_PTR_COMPRESS
    print_d_arr(raw_ptr, m, "rownnz ptr: ");
#endif

    int mask_size = (((m-1)%MASK_SIZE)==0) ? ((m-1)/MASK_SIZE) : (((m-1)/MASK_SIZE)+1) ;
    // thrust::device_vector<BMASK_TYPE> d_mask(mask, mask + mask_size);
    thrust::device_ptr<BMASK_TYPE> d_mask_ptr = thrust::device_pointer_cast(mask);
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
    auto zipped   = thrust::make_zip_iterator(thrust::make_tuple(counting, thrust_ptr+1));

    thrust::transform(
        thrust::cuda::par.on(stream),
        zipped, zipped + (m-1),
        thrust_ptr+1,
        MaskedTransform(thrust::raw_pointer_cast(d_mask_ptr))
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));

#ifdef DEBUG_PTR_COMPRESS
    cudaMemcpy(h_row.data(), raw_ptr, m * sizeof(IT), cudaMemcpyDeviceToHost);

    std::cout << "Masked ptr_vec: ";
    for (int x : h_row) std::cout << x << " ";
    std::cout << "\n";
    fflush(stdout);
#endif

    // raw_ptr = thrust::raw_pointer_cast(d_row.data());
    rownnz_to_rowptrs<int>(raw_ptr, m-1, stream, tmp_buff); // The function require the number of rows/cols, not the real vecsize

#ifdef DEBUG_PTR_COMPRESS
    cudaMemcpy(h_row.data(), raw_ptr, m * sizeof(IT), cudaMemcpyDeviceToHost);

    std::cout << "New ptr_vec: ";
    for (int x : h_row) std::cout << x << " ";
    std::cout << "\n";
    fflush(stdout);
#endif

    // IT *output;
    // CUDA_CHECK(cudaMalloc(&output, sizeof(IT)*m));
    // CUDA_CHECK(cudaMemcpy(output, raw_ptr, m*sizeof(IT), cudaMemcpyDeviceToDevice));
    // return(output);
    return(raw_ptr);
}

// ----------------------------------------------------------------------------------------------
//                        Funtions to perform the data compression (raw CUDA)
// ----------------------------------------------------------------------------------------------

#define BLOCK_SIZE 1024
#define ITEM_PER_THREAD 16

#define MAKE_IT_CRASH { int *hats = nullptr; *hats = 12; }

template <typename IT, typename VT>
struct SpaCommBuffers {

    SpaCommBuffers(const mmio::CSX<IT,VT>* to_compress) {
        initialized = true;
        int nnz = to_compress->nnz;
        entries_grid_size = (nnz + (BLOCK_SIZE*ITEM_PER_THREAD-1)) / (BLOCK_SIZE*ITEM_PER_THREAD);
        int ptrsize = (to_compress->majordim == mmio::MajorDim::ROWS) ? (to_compress->nrows +1) : (to_compress->ncols +1) ;

#ifdef NVTX_PROFILING
        NVTX_PUSH_RANGE("Alloc holders & buffers",2);
#endif

        CUDA_CHECK(cudaMallocHost(&host_buffer,    sizeof(int) * entries_grid_size));
        CUDA_CHECK(cudaMalloc(&selected_per_block, sizeof(int) * entries_grid_size));
        CUDA_CHECK(cudaMalloc(&IT_partial_results, sizeof(IT)  * entries_grid_size * BLOCK_SIZE * ITEM_PER_THREAD));
        CUDA_CHECK(cudaMalloc(&VT_partial_results, sizeof(VT)  * entries_grid_size * BLOCK_SIZE * ITEM_PER_THREAD));

        if (nnz>0) {
            CUDA_CHECK(cudaMalloc(&compressed_values,   sizeof(VT) * nnz));
            CUDA_CHECK(cudaMalloc(&compressed_indices,  sizeof(IT) * nnz));
        } else {
            compressed_values  = nullptr;
            compressed_indices = nullptr;
        }

        if(ptrsize>0) {
            CUDA_CHECK(cudaMalloc(&compressed_pointers, sizeof(IT) * ptrsize));
        } else {
            compressed_pointers = nullptr;
        }

#ifdef NVTX_PROFILING
        NVTX_POP_RANGE;
#endif
    }

    void explicitFree(void) {
        CUDA_FREE_SAFE(selected_per_block);
        CUDA_FREE_SAFE(IT_partial_results);
        CUDA_FREE_SAFE(VT_partial_results);
        CUDA_FREE_SAFE(compressed_values);
        CUDA_FREE_SAFE(compressed_indices);
        CUDA_FREE_SAFE(compressed_pointers);
        CUDA_CHECK(cudaFreeHost(host_buffer));
        tmp_buff.explicitFree();
        initialized = false;
    }

    ~SpaCommBuffers() {
        if (initialized) explicitFree();
    }

    bool initialized;
    int *host_buffer;
    int entries_grid_size;
    int *selected_per_block;
    IT *IT_partial_results;
    VT *VT_partial_results;

    VT *compressed_values;
    IT *compressed_indices;
    IT *compressed_pointers;

    cubTmpBuff tmp_buff;
};

template <typename IT>
__device__ int locate_segment_by_pos(int pos, int size, const IT* ptr_vec) {
    if (static_cast<IT>(pos) >= ptr_vec[size-1]) return(size-1);

    int low = 0, high = size;
    while (low < high) {
        int mid = (low + high) >> 1;
        if (ptr_vec[mid] <= static_cast<IT>(pos)) low = mid + 1;
        else high = mid;
    }
    return low - 1;
}


template<typename VT, typename PT>
__global__ void select_entries_kernel(const VT* input_vec, int n, const PT* ptr_vec, int m, const BMASK_TYPE* mask, VT *partial_output, int *selected_vals) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
/*
    // ----- Just for debug ----
    
    for (int i=0; i<ITEM_PER_THREAD; i++) {
     	int access_idx = tid*ITEM_PER_THREAD + i;
	
	if (access_idx<n) {
	    if (access_idx%MASK_SIZE == 0) int mask_word  = mask[i/MASK_SIZE];
	    int tmp_input   = input_vec[access_idx];
	    int tmp_bsearch = locate_segment_by_pos(access_idx, m, ptr_vec);
	    // if (tmp_bsearch >= m) MAKE_IT_CRASH  // Just to trigger an error
	    partial_output[access_idx] = 0;
	}

	if (access_idx<m-1) {
	    if (ptr_vec[access_idx]>ptr_vec[access_idx+1]) MAKE_IT_CRASH
	}

    }
    
    int block_aggregate = 0;
    // -------------------------
*/
    using BlockScan = cub::BlockScan<int, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage temp_storage_scan;

    int myflags[ITEM_PER_THREAD], blockdispl[ITEM_PER_THREAD];
    for (int i=0; i < ITEM_PER_THREAD; i++) {
        if (tid*ITEM_PER_THREAD + i < n) {
            int row = locate_segment_by_pos(tid*ITEM_PER_THREAD + i, m, ptr_vec);
            // int row = 1; // Just for debug
            int mask_word = row / MASK_SIZE;
            int mask_bit  = row % MASK_SIZE;
            myflags[i] = ((mask[mask_word] >> mask_bit) & 1) ? 1 : 0 ;
            // myflags[i] = 0; // Just for debug
        } else {
            myflags[i] = 0;
        }
    }

    int block_aggregate;
    BlockScan(temp_storage_scan).ExclusiveSum(myflags, blockdispl, block_aggregate);

    for (int i=0; i<ITEM_PER_THREAD ; i++) {  // I check block_offset + blockdispl[i] already since the structure of myflags[i]
        if (myflags[i]) {
            int block_offset = blockIdx.x * BLOCK_SIZE * ITEM_PER_THREAD;
            partial_output[block_offset + blockdispl[i]] = input_vec[tid*ITEM_PER_THREAD + i];
        }
    }

    if (threadIdx.x == 0) selected_vals[blockIdx.x] = block_aggregate;
}

template<typename VT>
__global__ void compact_select_entries_kernel(const VT* partial_output, int n, const int *selected_vals_psum, int m, int nselected, VT *output) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int chunk_size = blockDim.x * ITEM_PER_THREAD ; // True just because I use the same blockDim of select_entries_kernel

    // ---------- proper vector compact ----------
    int access_point = tid;
    for (int i=0; i < ITEM_PER_THREAD; i++) { // full grid coalesent access
        if (access_point < nselected) {
            int mydispl_idx = locate_segment_by_pos(access_point, m, selected_vals_psum);
            int mydispl_val = selected_vals_psum[mydispl_idx];
            output[access_point] = partial_output[(chunk_size * mydispl_idx) + (access_point - mydispl_val)];
        }
        access_point += blockDim.x*gridDim.x;
    }
}

// #define DEBUG_SELECT_ENTRIES_CUDA

template<typename ET, typename IT, typename VT>
int select_entries_cuda(const ET* input_vec, int n, const IT* ptr_vec, int m, const BMASK_TYPE* mask, SpaCommBuffers<IT,VT> *buffs,
    cudaStream_t stream = 0) {
    if (n==0) return(0);

    // ----- Take pointers to the extern buffers -----
    int grid_size = buffs->entries_grid_size;
    int *h_selected_per_block = buffs->host_buffer;
    int *selected_per_block = buffs->selected_per_block;
    ET *partial_output, *output;
    if constexpr (std::is_same_v<ET, IT>) {
        output         = buffs->compressed_indices;
        partial_output = buffs->IT_partial_results;
    } else if constexpr (std::is_same_v<ET, VT>) {
        output         = buffs->compressed_values;
        partial_output = buffs->VT_partial_results;
    } else {
        fprintf(stderr, "Error: unsupported template in %s\n", __func__);
        exit(__LINE__);
    }

    select_entries_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(input_vec, n, ptr_vec, m, mask, partial_output, selected_per_block);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpyAsync(h_selected_per_block, selected_per_block, sizeof(int)*grid_size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // std::vector<int> h_displ(grid_size + 1);
    int *h_displ = (int*)malloc(sizeof(int)*(grid_size + 1));
    h_displ[0] = 0;
    for (int i = 0; i < grid_size; i++) {
        h_displ[i+1] = h_displ[i] + h_selected_per_block[i];
    }
    int nselected = h_displ[grid_size];

#ifdef DEBUG_SELECT_ENTRIES_CUDA
    // fprintf(stdout, "h_selected_per_block: ");
    // for (int i=0; i<grid_size; i++) fprintf(stdout, "%d ", h_selected_per_block[i]);
    // fprintf(stdout, "\n");
    fprintf(stdout, "nselected: %d\n", nselected);
#endif

    if (nselected>0) {
        // Just because h_selected_per_block was already malloc with cudaMallocHost
        mempcpy(h_selected_per_block, h_displ, sizeof(int)*(grid_size));
        free(h_displ);

        int *d_displ = selected_per_block;
        CUDA_CHECK(cudaMemcpyAsync(d_displ, h_selected_per_block, sizeof(int)*grid_size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int new_grid_size = (nselected + (BLOCK_SIZE*ITEM_PER_THREAD-1)) / (BLOCK_SIZE*ITEM_PER_THREAD);
        compact_select_entries_kernel<<<new_grid_size, BLOCK_SIZE, 0, stream>>>(partial_output, n, d_displ, grid_size, nselected, output);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // for (int i = 0; i < grid_size; i++) {
    //     if (h_selected_per_block[i] > 0) {
    //         int src_base = i * BLOCK_SIZE * ITEM_PER_THREAD;
    //         CUDA_CHECK(cudaMemcpyAsync(output + h_displ[i], partial_output + src_base, sizeof(ET) * h_selected_per_block[i], cudaMemcpyDeviceToDevice, stream));
    //     }
    // }

    return(nselected);
}

// DEBUG function
template<typename T>
__global__ void check_ptr_kernel(T* ptr_vec, int ptr_size, int* check) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x;

    if (tid<ptr_size-1) {
	    if (ptr_vec[tid]>ptr_vec[tid+1]) atomicAdd(check, 1);
	}
}

#define CHECK_PTRVEC(V,S) { \
    int mygrid = (S + (BLOCK_SIZE-1)) / BLOCK_SIZE, h_check, *check;  \
    CUDA_CHECK(cudaMalloc(&check, sizeof(int))); \
    CUDA_CHECK(cudaMemset(check, 0, sizeof(int))); \
    SpaComm::check_ptr_kernel<<<mygrid, BLOCK_SIZE>>>(V, S, check); \
    CUDA_CHECK(cudaMemcpy(&h_check, check, sizeof(int), cudaMemcpyDeviceToHost)); \
    int world_rank; \
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); \
    fprintf(stdout, "[%d] ptr_vec check at line %d... ", world_rank, __LINE__); \
    if (h_check > 0) {  \
        fprintf(stderr, "[%d] Error: ptr_vec is invalid at line %d\n", world_rank, __LINE__); \
        MPI_Abort(MPI_COMM_WORLD, __LINE__); \
    } else { fprintf(stdout, "[%d] check at line %d is fine\n", world_rank, __LINE__); } \
    CUDA_CHECK(cudaFree(check)); \
    fflush(stdout); fflush(stderr); \
    MPI_Barrier(MPI_COMM_WORLD); \
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
#ifndef DEBUGBITMASKGENERATION
        mask_len = spcomm_2D(csx_A, csx_B, input_grid, &A_column_filters, &B_rows_filters);
#else
        if (grid->global_rank == 0) fprintf(stderr, "DEBUG ONLY MODE: sparsity patter communication is not performed, filters are filled with 1s.\n");
        srand(8);
        int k = csx_A->ncols;
        int mask_size = ((k%MASK_SIZE)==0) ? (k/MASK_SIZE) : ((k/MASK_SIZE)+1) ;
        CUDA_CHECK(cudaMalloc(&A_column_filters, sizeof(BMASK_TYPE)*nfilters*mask_len));
        CUDA_CHECK(cudaMalloc(&B_rows_filters, sizeof(BMASK_TYPE)*nfilters*mask_len));
        BMASK_TYPE *tmp = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*nfilters*mask_len);
        k = 0U;
        for (int i=0; i<MASK_SIZE; i++) k = (k | 1 << i);        // Uncomment this for an all 1 Mask, comment for an all 0 mask
        for (int i=0; i<nfilters*mask_len; i++) tmp[i] = rand(); // k; // Use rand() for a random mask
        CUDA_CHECK(cudaMemcpy(A_column_filters, tmp, sizeof(BMASK_TYPE)*nfilters*mask_len, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B_rows_filters, tmp, sizeof(BMASK_TYPE)*nfilters*mask_len, cudaMemcpyHostToDevice));
        free(tmp);
#endif
        initialized = true;

	// ---------------------------------------------------------------------------------

    }

    mmio::CSX<IT,VT>* Compress (const mmio::CSX<IT,VT> *M, int iteration_number, SpaCommBuffers<IT,VT>* buffs, cudaStream_t stream = 0) {

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
        int num_selected = select_entries_cuda<IT, IT>(
                M->idx_vec, M->nnz,
                M->ptr_vec, ptr_size,
                mask,
                buffs,
                stream
        );

#ifdef DEBUG_COMPRESSION
        if (grid->global_rank==0 && layout == mmio::MajorDim::COLS) {
            print_d_arr(M->idx_vec, M->nnz,       "Old idx: ");
            print_d_arr(new_idx,    num_selected, "New idx: ");
        }
#endif

        // Compute compressed value vector
        int num_selected_val = select_entries_cuda<VT, IT>(
                M->val, M->nnz,
                M->ptr_vec, ptr_size,
                mask,
                buffs,
                stream
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
/*
	VT *new_val; IT *new_idx; int num_selected = M->nnz;
	CUDA_CHECK(cudaMalloc(&new_idx, sizeof(IT)*(M->nnz))); // Just to debug
	CUDA_CHECK(cudaMalloc(&new_val, sizeof(IT)*(M->nnz))); // Just to debug
*/
        // Compute compressed pointer vector
        IT *new_row = buffs->compressed_pointers;
        CUDA_CHECK(cudaMemcpyAsync(new_row, M->ptr_vec, sizeof(IT)*ptr_size, cudaMemcpyDeviceToDevice, stream));
        select_ptrs(new_row, ptr_size, mask, stream, &(buffs->tmp_buff)); // Changes are done in place

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
        output->val      = buffs->compressed_values;
        output->ptr_vec  = buffs->compressed_pointers;
        output->idx_vec  = buffs->compressed_indices;

        return(output);
    }


    void explicitFree(void)
    {
        CUDA_FREE_SAFE(A_column_filters);
        CUDA_FREE_SAFE(B_rows_filters);
        initialized = false;
    }

    ~SpaCommHandler()
    {
        if (initialized) explicitFree();
    }

    // Communicator grid
    bool initialized;
    dmmio::ProcessGrid * grid;

    // These two vectors are device vectors that the process will use to compress local tiles before the communication
    BMASK_TYPE* A_column_filters; // Filters to be applied to A(i,j) matrix before than send to Process (i,k)
    BMASK_TYPE* B_rows_filters;   // Filters to be applied to B(i,j) matrix before than send to Process (k,j)

    IT mask_len;  // This is the len of every single mask (i.e. A-B common dimension / MASK_SIZE)
    IT nfilters;  // This is the number of filters in each filter collection (i.e. A_column_filters, B_column_filters)

};


}
