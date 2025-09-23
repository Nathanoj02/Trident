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


// Functor that maps value -> index in segments
struct SegmentLocator {
    const int* s;   // pointer to search vector
    int len;        // length of search vector

    SegmentLocator(const int* s_, int len_) : s(s_), len(len_) {}

    __device__ int operator()(int x) const {
        // binary search for largest j with s[j] <= x
        int lo = 0, hi = len;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (s[mid] <= x) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo -1 ; // index of segment
    }
};

int tmp_test() {
    // Search vector (segment boundaries)
    thrust::device_vector<int> s{0, 0, 1, 4, 4, 10, 15};
    // Input vector
    thrust::device_vector<int> v{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    thrust::device_vector<int> r(v.size());

    thrust::host_vector<int> h_s = s;
    thrust::host_vector<int> h_v = v;
    std::cout << "Search vector: ";
    for (auto x : h_s) std::cout << x << " ";
    std::cout << "\nInput vector: ";
    for (auto x : h_v) std::cout << x << " ";
    std::cout << "\n";

    // Apply transform
    auto s_ptr = thrust::raw_pointer_cast(s.data());
    thrust::transform(v.begin(), v.end(), r.begin(), SegmentLocator(s_ptr, s.size()));

    // Copy back to host
    thrust::host_vector<int> h_r = r;
    std::cout << "Result: ";
    for (auto x : h_r) std::cout << x << " ";
    std::cout << "\n";

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
    tmp_test();

    return 0;
}

