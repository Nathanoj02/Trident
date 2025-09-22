#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/device/device_segmented_reduce.cuh>
#include <iostream>

#define MASK_SIZE 4

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

int main() {
    thrust::device_vector<int> test_vector{0, 0, 1, 1, 3, 3, 6, 6};

    thrust::device_vector<int> A_rowptr{0, 1, 3, 6, 6};
    thrust::device_vector<int> B_colptr{0, 0, 2, 3, 5};

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

    return 0;
}

