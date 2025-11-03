#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define MPI_CHECK(call) do { \
    int err = call; \
    if (err != MPI_SUCCESS) { \
        fprintf(stderr, "MPI error at %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while(0)

// Benchmark blocking sendrecv operations
void benchmark_sendrecv(char * d_sendbuf, char * d_recvbuf, size_t buffer_size, int rank, int num_ranks) {
    int partner = (rank + 1) % num_ranks;
    const int num_iterations = 100;
    int nt = omp_get_max_threads();
    printf("nthreads -- %d\n", nt);
    size_t loc_buf_size = buffer_size / nt;

    double start_time = MPI_Wtime();

#pragma omp parallel
    for (int i = 0; i < num_iterations; i++)
    {
        int tid = omp_get_thread_num();
        size_t offset = tid * loc_buf_size;
        MPI_Sendrecv(d_sendbuf + offset, loc_buf_size, MPI_BYTE, partner, 0,
                     d_recvbuf + offset, loc_buf_size, MPI_BYTE, partner, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (rank == 0) {
        printf("Blocking SendRecv benchmark:\n");
        printf("  Total time: %.6f seconds\n", elapsed);
        printf("  Average time per operation: %.6f ms\n", (elapsed / (num_iterations)) * 1000.0);
    }
}

// Benchmark non-blocking Isend/Irecv operations
void benchmark_isend_irecv(char * d_sendbuf, char * d_recvbuf, size_t buffer_size, int rank, int num_ranks) {
    int partner = (rank + 1) % num_ranks;
    const int num_iterations = 100;
    int nt = omp_get_max_threads();
    printf("nthreads -- %d\n", nt);
    std::vector<MPI_Request> requests(num_iterations * 2);
    size_t loc_buf_size = buffer_size / nt;

    double start_time = MPI_Wtime();

    #pragma omp parallel
    for (int i = 0; i < num_iterations ; i++)
    {
        int tid = omp_get_thread_num();
        size_t offset = tid * loc_buf_size;
        MPI_Isend(d_sendbuf + offset, loc_buf_size, MPI_BYTE, partner, 0,
                  MPI_COMM_WORLD, &requests[i * 2]);
        MPI_Irecv(d_recvbuf + offset, loc_buf_size, MPI_BYTE, partner, 0,
                  MPI_COMM_WORLD, &requests[i * 2 + 1]);
    }

    // Wait for all operations to complete
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (rank == 0) {
        printf("Non-blocking Isend/Irecv benchmark:\n");
        printf("  Total time: %.6f seconds\n", elapsed);
        printf("  Average time per operation: %.6f ms\n", (elapsed / (num_iterations)) * 1000.0);
    }
}

int main(int argc, char** argv)
{
    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);

    int rank, num_ranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &num_ranks));

    if (num_ranks < 2) {
        if (rank == 0) {
            fprintf(stderr, "This benchmark requires at least 2 MPI ranks\n");
        }
        MPI_Finalize();
        return 1;
    }

    int nt = omp_get_max_threads();

    // Parse command line argument
    bool use_blocking = true;  // Default to blocking sendrecv
    if (argc > 1) {
        if (strcmp(argv[1], "isend") == 0 || strcmp(argv[1], "nonblocking") == 0) {
            use_blocking = false;
        } else if (strcmp(argv[1], "sendrecv") == 0 || strcmp(argv[1], "blocking") == 0) {
            use_blocking = true;
        } else {
            if (rank == 0) {
                printf("Usage: %s [sendrecv|blocking|isend|nonblocking]\n", argv[0]);
                printf("  sendrecv/blocking    - Use blocking MPI_Sendrecv (default)\n");
                printf("  isend/nonblocking    - Use non-blocking MPI_Isend/Irecv\n");
            }
            MPI_Finalize();
            return 1;
        }
    }

    const long long buffer_size = std::atoll(argv[2]) * nt;

    char * d_sendbuf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sendbuf, buffer_size));
    CUDA_CHECK(cudaMemset(d_sendbuf, 0, buffer_size));

    char * d_recvbuf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_recvbuf, buffer_size));
    CUDA_CHECK(cudaMemset(d_recvbuf, 0, buffer_size));

    if (rank == 0) {
        printf("P2P Multi-threaded Benchmark\n");
        printf("Buffer size: %zu bytes (%.2f MB)\n", buffer_size/nt, buffer_size/(1e6 * nt));
        printf("Number of iterations: 100\n");
        printf("Mode: %s\n\n", use_blocking ? "Blocking SendRecv" : "Non-blocking Isend/Irecv");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Run the selected benchmark
    if (use_blocking) {
        benchmark_sendrecv(d_sendbuf, d_recvbuf, buffer_size, rank, num_ranks);
    } else {
        benchmark_isend_irecv(d_sendbuf, d_recvbuf, buffer_size, rank, num_ranks);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_sendbuf));
    CUDA_CHECK(cudaFree(d_recvbuf));

    MPI_CHECK(MPI_Finalize());
    return 0;
}
