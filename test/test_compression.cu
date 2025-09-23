#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/device/device_segmented_reduce.cuh>
#include <iostream>

#include <cub/cub.cuh>

#include <vector>
#include <iomanip>

#define MASK_SIZE 4   // set to 32 in production

// A general function to compute result vector given a row/col pointer array
thrust::device_vector<int> compress_row_or_col(const thrust::device_vector<int>& ptr, int mask_size) {
    auto ptr_d = thrust::raw_pointer_cast(ptr.data());
    size_t n = ptr.size() - 1;  // number of rows/cols

    thrust::device_vector<int> presult(n);

    thrust::transform(
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(n),
        presult.begin(),
        [=] __device__(int i) {
            int a = ptr_d[i];
            int b = ptr_d[i + 1];
            int c = 0;
            if (a != b) {
                c = 1 << (i % mask_size);
            }
            return c;
        });

    auto op = cub::Sum();
    int initial_value = 0;
    int segment_size  = mask_size;
    int num_segments  = (n%mask_size == 0) ? (n/mask_size) : ((n/mask_size)+1) ;
    thrust::device_vector<int> result(num_segments);

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
       presult.begin(), result.begin(), 
       num_segments, d_offsets.data(), d_offsets.data()+1, 
       op, initial_value
    );

    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    // Run reduction
    cub::DeviceSegmentedReduce::Reduce(
 	d_temp_storage, temp_storage_bytes, 
	presult.begin(), result.begin(), 
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

thrust::device_vector<int> bitwise_and_transform(
    const thrust::device_vector<int>& a,
    const thrust::device_vector<int>& b)
{
    int n = a.size();
    thrust::device_vector<int> d_out(n);

    // Zip iterators over both inputs
    auto zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin()));
    auto zipped_end   = zipped_begin + n;

    // Create transform iterator that applies bitwise AND
    auto transform_iter = thrust::make_transform_iterator(zipped_begin, BitwiseAnd());

    // Use thrust::copy for simplicity (CUB transform API is experimental and more verbose)
    thrust::copy(transform_iter, transform_iter + n, d_out.begin());

    return d_out;
}

// -------------------
// For the compression
// -------------------

void printCSRMatrix(const std::vector<int>& colIdx, const std::vector<int>& rowPtr) {
    int nrows = rowPtr.size() - 1;
    int ncols = 0;

    // Find maximum column index to determine number of columns
    for (int c : colIdx) {
        if (c > ncols) ncols = c;
    }
    ncols += 1; // since indices start at 0

    for (int r = 0; r < nrows; r++) {
        std::vector<char> row(ncols, '-');
        for (int j = rowPtr[r]; j < rowPtr[r + 1]; j++) {
            row[colIdx[j]] = '1';
        }
        // Print row
        for (int c = 0; c < ncols; c++) {
            std::cout << row[c] << " ";
        }
        std::cout << "\n";
    }
}

void printEntriesByRow(const thrust::host_vector<int>& colIdx,
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

int tmp_test1 () {
    // Example input
    thrust::host_vector<int> h_v{0,
                                 1, 3, 7,
                                 0, 1, 4, 7, 8, 10,
                                 0, 2, 4, 6, 8,
                                 1, 3, 5, 7, 9,
                                 2, 8,
                                 1, 5, 7,
                                 0, 10
    };
    thrust::host_vector<int> h_s{0, 1, 4, 10, 15, 20, 22, 25, 27}; // 8 segments
    thrust::host_vector<int> h_c(2); // 8 bits total (2 words * MASK_SIZE=4)

    printCSRMatrix(std::vector<int>(h_v.begin(), h_v.end()),
               std::vector<int>(h_s.begin(), h_s.end()));
    printEntriesByRow(h_v, h_s);


    // condition mask:
    // let's enable segments {0,1,4,7} for fun
    // word0 covers segments [0..3], word1 covers [4..7]
    h_c[0] = (1 << 0) | (1 << 1);       // bits 0 and 1 set
    h_c[1] = (1 << 0) | (1 << 3);       // bits 4 and 7 set

    int n = h_v.size();
    int m = h_s.size();

    std::cout << "Input vector v: ";
    for (int x : h_v) std::cout << x << " ";
    std::cout << "\nSearch vector s: ";
    for (int x : h_s) std::cout << x << " ";
    std::cout << "\nBitmask vector c: ";
    for (int x : h_c) std::cout << x << " ";
    std::cout << "\n";

    // Copy to device
    thrust::device_vector<int> d_v = h_v;
    thrust::device_vector<int> d_s = h_s;
    thrust::device_vector<int> d_c = h_c;
    thrust::device_vector<int> d_out(n);
    thrust::device_vector<int> d_num_selected(1);

    auto counting = thrust::make_counting_iterator<int>(0);
    auto flags_it = thrust::make_transform_iterator(
        counting,
        KeepFlagByIndex(thrust::raw_pointer_cast(d_s.data()), m,
                        thrust::raw_pointer_cast(d_c.data()))
    );

    // Temp storage
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes,
        thrust::raw_pointer_cast(d_v.data()), // values
        flags_it,                             // lazy flags
        thrust::raw_pointer_cast(d_out.data()),
        thrust::raw_pointer_cast(d_num_selected.data()),
        n
    );

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes,
        thrust::raw_pointer_cast(d_v.data()), // values
        flags_it,                             // lazy flags
        thrust::raw_pointer_cast(d_out.data()),
        thrust::raw_pointer_cast(d_num_selected.data()),
        n
    );

    // Copy results back
    int num_selected = d_num_selected[0];
    thrust::host_vector<int> h_out(d_out.begin(), d_out.begin() + num_selected);

    std::cout << "Selected " << num_selected << " values:\n";
    for (int x : h_out) std::cout << x << " ";
    std::cout << "\n";

    cudaFree(d_temp_storage);
    return 0;
}


int main() {
    thrust::device_vector<int> test_vector{0, 0, 1, 1, 3, 3, 6, 6};

    thrust::device_vector<int> A_rowptr{0, 1, 2, 2, 3, 4, 4, 5, 6};
    thrust::device_vector<int> B_colptr{0, 0, 1, 2, 2, 3, 4, 4, 5};

    // Apply the same function to both
    auto result_test = compress_row_or_col(test_vector, MASK_SIZE);
    auto resultA = compress_row_or_col(A_rowptr, MASK_SIZE);
    auto resultB = compress_row_or_col(B_colptr, MASK_SIZE);

    thrust::host_vector<int> h_test = result_test;
    thrust::host_vector<int> h_resultA = resultA;
    thrust::host_vector<int> h_resultB = resultB;

    std::cout << "Result test: ";
    for (auto v : h_test) std::cout << v << " ";
    std::cout << "\nResult A: ";
    for (auto v : h_resultA) std::cout << v << " ";
    std::cout << "\nResult B: ";
    for (auto v : h_resultB) std::cout << v << " ";
    std::cout << std::endl;

    auto intersection = bitwise_and_transform(resultA, resultB);
    thrust::host_vector<int> h_intersection = intersection;

    std::cout << "Intersection: ";
    for (auto v : h_intersection) std::cout << v << " ";
    std::cout << std::endl;

    std::cout << "----- Test for compression -----" << std::endl;
    tmp_test1();

    return 0;
}

