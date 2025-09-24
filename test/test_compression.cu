#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_adjacent_difference.cuh>
#include <iostream>

#include <cub/cub.cuh>

#include <vector>
#include <iomanip>

#include "common.h"
#include "test_utils.cuh"

#define MASK_SIZE 4   // set to 32 in production
#define DEBUG

// A general function to compute result vector given a row/col pointer array
template<typename IT>
int8_t* gen_bitmask(const IT* ptr_d, int n, int mask_size) {
    thrust::device_vector<int8_t> presult(n);

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
    int num_segments  = (n%mask_size == 0) ? (n/mask_size) : ((n/mask_size)+1) ;
    // thrust::device_vector<int> result(num_segments);
    int8_t *result;
    cudaMalloc(&result, sizeof(int8_t)*num_segments);

    thrust::host_vector<int> h_offsets(num_segments + 1);
    for (int i = 0; i < num_segments; i++) {
        h_offsets[i] = i * segment_size;
    }
    h_offsets[num_segments] = num_segments * segment_size;

    thrust::device_vector<int> d_offsets = h_offsets;

    // Determine temporary device storage requirements
    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::Reduce(
       d_temp_storage, temp_storage_bytes, 
       presult.begin(), result,
       num_segments, d_offsets.data(), d_offsets.data()+1, 
       op, initial_value
    );

    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    // Run reduction
    cub::DeviceSegmentedReduce::Reduce(
 	d_temp_storage, temp_storage_bytes, 
	presult.begin(), result,
	num_segments, d_offsets.data(), d_offsets.data()+1, 
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
    cudaMalloc(&d_out, sizeof(T) * n);

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

// -------------------
// For the compression
// -------------------

template<typename IT, typename VT>
void printCSRMatrix(const std::vector<IT>& colIdx,
                    const std::vector<VT>& values,
                    const std::vector<IT>& rowPtr) {
    int nrows = rowPtr.size() - 1;
    int ncols = 0;

    // Find maximum column index to determine number of columns
    for (int c : colIdx) {
        if (c > ncols) ncols = c;
    }
    ncols += 1; // since indices start at 0

    for (int r = 0; r < nrows; r++) {
        std::vector<std::string> row(ncols, " - ");

        // Fill row with values
        for (int j = rowPtr[r]; j < rowPtr[r + 1]; j++) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1) << values[j]; // e.g. "3.0"
            row[colIdx[j]] = ss.str();
        }

        // Print row
        for (int c = 0; c < ncols; c++) {
            std::cout << std::setw(4) << row[c] << " ";
        }
        std::cout << "\n";
    }
}

template<typename IT, typename VT>
void printCSCMatrix(const std::vector<IT>& rowIdx,
                    const std::vector<VT>& values,
                    const std::vector<IT>& colPtr) {
    int ncols = colPtr.size() - 1;
    int nrows = 0;

    // Find maximum row index to determine number of rows
    for (int r : rowIdx) {
        if (r > nrows) nrows = r;
    }
    nrows += 1; // since indices start at 0

    for (int r = 0; r < nrows; r++) {
        std::vector<std::string> row(ncols, " - ");

        // Fill row r with values
        for (int c = 0; c < ncols; c++) {
            for (int j = colPtr[c]; j < colPtr[c + 1]; j++) {
                if (rowIdx[j] == r) {
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(1) << values[j];
                    row[c] = ss.str();
                }
            }
        }

        // Print row
        for (int c = 0; c < ncols; c++) {
            std::cout << std::setw(4) << row[c] << " ";
        }
        std::cout << "\n";
    }
}


template<typename T>
void printEntriesByRow(const thrust::host_vector<T>& colIdx,
                       const thrust::host_vector<int>& rowPtr) {
    int nrows = rowPtr.size() - 1;
    for (int r = 0; r < nrows; r++) {
        for (int j = rowPtr[r]; j < rowPtr[r + 1]; j++) {
            std::cout << colIdx[j] << " ";
        }
        if (r < nrows - 1) std::cout << "| ";
    }
    std::cout << "\n";
}

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


// ---------- Copied from outside, just include them ----------

template <typename T>
struct DiffOp2
{
    DiffOp2(){}
    __host__ __device__ __forceinline__
    T operator()(const T& lhs, const T& rhs)
    {
        return lhs - rhs;
    }
};

template <typename IT>
void rownnz_to_rowptrs(IT * d_rowptrs, const IT nrows) {
    void * d_tmp = NULL;
    size_t tmp_size = 0;
    cub::DeviceScan::InclusiveSum(d_tmp, tmp_size, d_rowptrs+1, nrows);
    cudaMalloc(&d_tmp, tmp_size);
    cub::DeviceScan::InclusiveSum(d_tmp, tmp_size, d_rowptrs+1, nrows);
    cudaFree(d_tmp);
    cudaDeviceSynchronize();
}

template <typename IT>
void rowptrs_to_rownnz(IT * d_rowptrs, const IT nrows) {
    void * d_tmp = NULL;
    size_t tmp_size = 0;
    cub::DeviceAdjacentDifference::SubtractLeft(d_tmp, tmp_size, d_rowptrs, nrows+1, DiffOp2<IT>{});
    cudaMalloc(&d_tmp, tmp_size);
    cub::DeviceAdjacentDifference::SubtractLeft(d_tmp, tmp_size, d_rowptrs, nrows+1, DiffOp2<IT>{});
    cudaFree(d_tmp);
    cudaDeviceSynchronize();
}
// ------------------------------------------------------------

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

    rowptrs_to_rownnz<IT>(raw_ptr, m);

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
        zipped, zipped + m,
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
    rownnz_to_rowptrs<int>(raw_ptr, m);

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

int tmp_test1 () {
    // Example input
    thrust::host_vector<float> h_val{1.0,
                                   2.0,   3.0,  4.0,
                                   5.0,   6.0,  7.0,  8.0,  9.0, 10.0,
                                   11.0, 12.0, 13.0, 14.0, 15.0,
                                   16.0, 17.0, 18.0, 19.0, 20.0,
                                   21.0, 22.0,
                                   23.0, 24.0, 25.0,
                                   26.0, 27.0
    };
    thrust::host_vector<int> h_col{0,
                                 1, 3, 7,
                                 0, 1, 4, 7, 8, 10,
                                 0, 2, 4, 6, 8,
                                 1, 3, 5, 7, 9,
                                 2, 8,
                                 1, 5, 7,
                                 0, 10
    };
    thrust::host_vector<int> h_row{0, 1, 4, 10, 15, 20, 22, 25, 27}; // 8 segments
    thrust::host_vector<int> h_mask(2); // 8 bits total (2 words * MASK_SIZE=4)

    printCSRMatrix(std::vector<int>(h_col.begin(), h_col.end()),
                   std::vector<float>(h_val.begin(), h_val.end()),
                   std::vector<int>(h_row.begin(), h_row.end())
    );
    printEntriesByRow(h_col, h_row);
    printEntriesByRow(h_val, h_row);
    fflush(stdout);

    // condition mask:
    // let's enable segments {0,1,4,7} for fun
    // word0 covers segments [0..3], word1 covers [4..7]
    h_mask[0] = (1 << 0) | (1 << 1);       // bits 0 and 1 set
    h_mask[1] = (1 << 0) | (1 << 3);       // bits 4 and 7 set

    int n = h_col.size();
    int m = h_row.size();

    std::cout << "Input vector v: ";
    for (int x : h_col) std::cout << x << " ";
    std::cout << "\nSearch vector s: ";
    for (int x : h_row) std::cout << x << " ";
    std::cout << "\nBitmask vector c: ";
    for (int x : h_mask) std::cout << x << " ";
    std::cout << "\n";
    fflush(stdout);

    // Copy to device
    thrust::device_vector<float> d_val  = h_val;
    thrust::device_vector<int>   d_col  = h_col;
    thrust::device_vector<int>   d_row  = h_row;
    thrust::device_vector<int>   d_mask = h_mask;

    int* new_col;
    int num_selected = select_entries<int, int>(
            thrust::raw_pointer_cast(d_col.data()), n,
            thrust::raw_pointer_cast(d_row.data()), m,
            thrust::raw_pointer_cast(d_mask.data()),
            &new_col
    );
    {
        int *tmp = (int*)malloc(sizeof(int)*num_selected);
        cudaMemcpy(tmp, new_col, sizeof(int)*num_selected, cudaMemcpyDeviceToHost);

        std::cout << "[out] Selected " << num_selected << " values:\n";
        for (int i=0; i<num_selected; i++) std::cout << tmp[i] << " ";
        std::cout << "\n";
        free(tmp);
    }
    fflush(stdout);

    float* new_val;
    int num_selected_val = select_entries<float, int>(
            thrust::raw_pointer_cast(d_val.data()), n,
            thrust::raw_pointer_cast(d_row.data()), m,
            thrust::raw_pointer_cast(d_mask.data()),
            &new_val
    );
    {
        float *tmp = (float*)malloc(sizeof(float)*num_selected_val);
        cudaMemcpy(tmp, new_val, sizeof(float)*num_selected_val, cudaMemcpyDeviceToHost);

        std::cout << "[out] Selected " << num_selected_val << " values:\n";
        for (int i=0; i<num_selected_val; i++) std::cout << tmp[i] << " ";
        std::cout << "\n";
        free(tmp);
    }
    fflush(stdout);

    if (num_selected != num_selected_val) {
        fprintf(stderr, "Error: num_selected_val (%d) differ from num_selected (%d)!\n", num_selected_val, num_selected);
        exit(__LINE__);
    }

    // Part for row pointer
    int *new_row = select_ptrs(thrust::raw_pointer_cast(d_row.data()), m, thrust::raw_pointer_cast(d_mask.data()));
    fflush(stdout);

    std::cout << "Test at line " << __LINE__ << std::endl; fflush(stdout);

    thrust::device_ptr<float> d_val_ptr(new_val);
    thrust::device_ptr<int>   d_col_ptr(new_col);
    thrust::device_ptr<int>   d_row_ptr(new_row);

    std::cout << "Test at line " << __LINE__ << std::endl; fflush(stdout);

    std::cout << "new_val ptr: " << new_val << ", new_col ptr: " << new_col
          << ", d_row ptr: " << new_row
          << ", num_selected: " << num_selected
          << ", m: " << m << std::endl;

    // Copy to host vectors
    thrust::host_vector<float> h_new_val(num_selected);
    thrust::host_vector<int>   h_new_col(num_selected);
    thrust::host_vector<int>   h_new_row(m);

    thrust::copy(d_val_ptr, d_val_ptr + num_selected, h_new_val.begin());
    thrust::copy(d_col_ptr, d_col_ptr + num_selected, h_new_col.begin());
    thrust::copy(d_row_ptr, d_row_ptr + m, h_new_row.begin());

    printCSRMatrix(std::vector<int>(h_new_col.begin(), h_new_col.end()),
                   std::vector<float>(h_new_val.begin(), h_new_val.end()),
                   std::vector<int>(h_new_row.begin(), h_new_row.end())
    );
    fflush(stdout);

    return 0;
}

bool div(int i, int j, int k) { return((i%j)==k); }
bool even(int i) { return(div(i, 2, 0)); }
bool  odd(int i) { return(div(i, 2, 1)); }
bool zeroOthree(int i) { return(div(i, 3, 0)); }
bool oneOthree(int i)  { return(div(i, 3, 1)); }
bool twoOthree(int i)  { return(div(i, 3, 2)); }

typedef bool(*ConditionFn)(int);
ConditionFn syntetic_selections(int i) {

}

template<typename IT, typename VT>
mmio::CSX<IT, VT>* gen_syntetic_matrix(int seed, int n, mmio::MajorDim layout, bool print=false) {

    bool(*empty_row_condition)(int);
    switch(seed) {
        case 0:
            empty_row_condition = even;
            break;
        case 1:
            empty_row_condition = odd;
            break;
        case 2:
            empty_row_condition = zeroOthree;
            break;
        case 3:
            empty_row_condition = oneOthree;
            break;
        case 4:
            empty_row_condition = twoOthree;
            break;
        default:
            fprintf(stderr, "Unrecognized seed (%d)\n", seed);
            exit(__LINE__);
    }

    std::vector<IT> vec_ptr(n+1);
    vec_ptr[0] = 0;
    IT nnz = 0;

    for (int i=0; i<n; i++) {
        if (empty_row_condition(i)) {
            IT row_nnz = (i<(n/2)) ? (i+1) : (n-i);
            vec_ptr[i+1] = vec_ptr[i] + row_nnz;
            nnz += row_nnz;
        } else {
            vec_ptr[i+1] = vec_ptr[i];
        }
    }

    std::vector<IT> vec_idx(nnz);
    std::vector<VT> vec_val(nnz);
    for (int i=0; i<n; i++) {
        for (int j=0; j<(vec_ptr[i+1] - vec_ptr[i]); j++) {
            vec_idx[vec_ptr[i] + j] = (i<(n/2)) ? (i + j) : (n - j - 1) ;
            vec_val[vec_ptr[i] + j] = (vec_ptr[i] + j) * 1.0 ;
        }
    }

    if (print) {
        std::cout << "---------- Syntetic matrix ( " << seed << ", " << n << ") ----------" << std::endl;

        if (layout == mmio::MajorDim::ROWS) {
            std::cout << "----- CSR -----" << std::endl;
            printCSRMatrix(vec_idx, vec_val, vec_ptr);
        } else {
            std::cout << "----- CSC -----" << std::endl;
            printCSCMatrix(vec_idx, vec_val, vec_ptr);
        }
    }

    mmio::CSX<IT,VT> *csx = (mmio::CSX<IT,VT>*)malloc(sizeof(mmio::CSX<IT,VT>));
    csx->majordim = layout;
    csx->nnz      = nnz;
    csx->nrows    = n;
    csx->ncols    = n;
    csx->val     = (VT*)malloc(nnz * sizeof(VT));
    csx->idx_vec = (IT*)malloc(nnz * sizeof(IT));
    csx->ptr_vec = (IT*)malloc((n+1) * sizeof(IT));

    std::copy(vec_val.begin(), vec_val.end(), csx->val);
    std::copy(vec_idx.begin(), vec_idx.end(), csx->idx_vec);
    std::copy(vec_ptr.begin(), vec_ptr.end(), csx->ptr_vec);

    return(csx);

}


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
    auto A_map = gen_bitmask(Acsc->ptr_vec, MASK_SIZE);
    auto B_map = gen_bitmask(Bcsr->ptr_vec, MASK_SIZE);
    IT *raw_A_map = thrust::raw_pointer_cast(A_map.data());
    IT *raw_B_map = thrust::raw_pointer_cast(B_map.data());

    // ---------- Ghatering all the required maps ----------
    int8_t *recv_A_maps = (int8_t*)malloc(sizeof(int8_t)*mask_size*(grid->row_size));  // BUG: This must be done with cudaMalloc
    MPI_Allgather(raw_A_map, mask_size, MPI_INT8_T, recv_A_maps, mask_size, MPI_INT8_T, grid->row_comm);

    int8_t *recv_B_maps = (int8_t*)malloc(sizeof(int8_t)*mask_size*(grid->col_size));  // BUG: This must be done with cudaMalloc
    MPI_Allgather(raw_B_map, mask_size, MPI_INT8_T, recv_B_maps, mask_size, MPI_INT8_T, grid->col_comm);

    // ---------- Performing the mask intersection ----------
    // since mapsizes are equal, we can comput all the intersections togheter
    auto all_intersections = intersect_bitmasks(recv_A_maps, recv_B_maps, mask_size * grid->row_size); // grid->row_size == grid->col_size

    // ---------- Alltoall data sisplacement back ----------
    *col_filters = (int8_t*)malloc(sizeof(int8_t)*mask_size*(grid->row_size));  // BUG: This must be done with cudaMalloc
    MPI_Alltoall(all_intersections, mask_size, MPI_INT8_T, col_filters, mask_size, MPI_INT8_T, grid->row_comm);

    *row_filters = (int8_t*)malloc(sizeof(int8_t)*mask_size*(grid->col_size));  // BUG: This must be done with cudaMalloc
    MPI_Alltoall(all_intersections, mask_size, MPI_INT8_T, row_filters, mask_size, MPI_INT8_T, grid->col_comm);

    free(recv_A_maps);
    free(recv_B_maps);
    // free(raw_A_map);  // Now managen by cub
    // free(raw_B_map);  // Now managen by cub
}

int main(int argc, char ** argv) {

    MPI_Init(&argc, &argv);

    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Barrier(MPI_COMM_WORLD);

    Config * config = (Config *)(malloc(sizeof(Config)));
    parse_args(argc, argv, config);
    MPI_Barrier(MPI_COMM_WORLD);

    // Set the process grid parameeters
    int nprocrows = config->nprocrows, nproccols = config->nproccols, nprocpergroup = world_size/(nprocrows*nproccols);
    dmmio::ProcessGrid *grid = dmmio::io::ProcessGrid_create(nprocrows, nproccols, nprocpergroup);

    // ----------------------------------------------------------------------------------------------------

    if (world_rank == 0) {
        thrust::device_vector<int> test_vector{0, 0, 1, 1, 3, 3, 6, 6};

        thrust::device_vector<int> A_rowptr{0, 1, 2, 2, 3, 4, 4, 5, 6};
        thrust::device_vector<int> B_colptr{0, 0, 1, 2, 2, 3, 4, 4, 5};

        // Apply the same function to both
        int8_t *result_test = gen_bitmask(thrust::raw_pointer_cast(test_vector.data()), test_vector.size()-1, MASK_SIZE);
        int8_t *resultA     = gen_bitmask(thrust::raw_pointer_cast(A_rowptr.data()), A_rowptr.size()-1, MASK_SIZE);
        int8_t *resultB     = gen_bitmask(thrust::raw_pointer_cast(B_colptr.data()), B_colptr.size()-1, MASK_SIZE);

        int num_segments_test = ((test_vector.size()-1)%MASK_SIZE == 0) ? ((test_vector.size()-1)/MASK_SIZE) : (((test_vector.size()-1)/MASK_SIZE)+1);
        int num_segments_A    = ((A_rowptr.size()-1)%MASK_SIZE == 0) ? ((A_rowptr.size()-1)/MASK_SIZE) : (((A_rowptr.size()-1)/MASK_SIZE)+1);
        int num_segments_B    = ((B_colptr.size()-1)%MASK_SIZE == 0) ? ((B_colptr.size()-1)/MASK_SIZE) : (((B_colptr.size()-1)/MASK_SIZE)+1);
        std::vector<int8_t> h_test(num_segments_test);
        std::vector<int8_t> h_resultA(num_segments_A);
        std::vector<int8_t> h_resultB(num_segments_B);

        cudaMemcpy(h_test.data(),    result_test, sizeof(int8_t) * num_segments_test, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_resultA.data(), resultA,     sizeof(int8_t) * num_segments_A, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_resultB.data(), resultB,     sizeof(int8_t) * num_segments_B, cudaMemcpyDeviceToHost);

        std::cout << "Result test: ";
        for (auto v : h_test) std::cout << static_cast<int>(v) << " ";
        std::cout << "\nResult A: ";
        for (auto v : h_resultA) std::cout << static_cast<int>(v) << " ";
        std::cout << "\nResult B: ";
        for (auto v : h_resultB) std::cout << static_cast<int>(v) << " ";
        std::cout << std::endl;

        ASSERT(num_segments_A == num_segments_B, "For the intersection num_segments_A == num_segments_B");
        int8_t *intersection = intersect_bitmasks(resultA, resultB, num_segments_A);

        std::vector<int8_t> h_intersection(num_segments_A);
        cudaMemcpy(h_intersection.data(), intersection, sizeof(int8_t) * num_segments_A, cudaMemcpyDeviceToHost);

        std::cout << "Intersection: ";
        for (auto v : h_intersection) std::cout << static_cast<int>(v) << " ";
        std::cout << std::endl;

        std::cout << "----- Test for compression -----" << std::endl;
        tmp_test1();

        std::cout << "----- Test syntetic matrices -----" << std::endl;
        for (int i=0; i<8; i++) free(gen_syntetic_matrix<int,float>(i%5, 8, (i<=4) ? (mmio::MajorDim::ROWS) : (mmio::MajorDim::COLS), true));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ----------------------------------------------------------------------------------------------------

    if (world_rank==0) std::cout << "-------------------- Parallel part --------------------" << std::endl; fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    dmmio::utils::ProcessGrid_graph(grid, stdout);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Here I am generating the matrices in the way that:
     *                     B:
     *                     ---------------
     *                     | 2%3  | 1%3  |
     *                     ---------------
     *                     | 2%3  | 1%3  |
     *                     ---------------
     *                        ||    ||
     * A:                     \/    \/
     * ---------------     ---------------
     * | even | even | --> | 2%6  | 4%6  |
     * ---------------     ---------------
     * | odd  | odd  | --> | 5%6  | 1%6  |
     * ---------------     ---------------
     *
     */
    mmio::CSX<int,float> *A_csx = gen_syntetic_matrix<int,float>(grid->col_rank   , 8, mmio::MajorDim::COLS);
    mmio::CSX<int,float> *B_csx = gen_syntetic_matrix<int,float>(grid->row_rank +3, 8, mmio::MajorDim::ROWS);

    MPI_ALL_PRINT(
        fprintf(fp, "global_rank: %d, row_rank: %d, col_rank: %d, node_rank: %d\n",
                grid->global_rank, grid->row_rank, grid->col_rank, grid->node_rank
        );
        mmio::utils::CSX_print_as_dense(A_csx, "A matrix", fp);
        mmio::utils::CSX_print_as_dense(B_csx, "B matrix", fp);
    )

    return 0;
}

