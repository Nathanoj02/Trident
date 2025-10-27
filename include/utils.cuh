#pragma once
#include "common.h"
#define CUSPARSE_CHECK(call) do {                                    \
    cusparseStatus_t err = call;                                     \
    if (err != CUSPARSE_STATUS_SUCCESS) {                            \
        fprintf(stderr, "cuSPARSE error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cusparseGetErrorString(err));    \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while(0)

#define OPSTR(X) ((X == dmmio::Operation::None) ? ("None") : ("Transpose") )

#define CHECK_PTR(PT, IT) {  \
    if (PT == nullptr) {  \
        std::cerr << "Process " << world_rank << ", " << __func__ << ":" << __LINE__ << "nullptr\n";  \
    } else {  \
        CUmemorytype memType;  \
        CUresult res = cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, reinterpret_cast<CUdeviceptr>(PT));  \
        if (res != CUDA_SUCCESS) {  \
            int world_rank;  \
            const char* errName = nullptr;  \
            cuGetErrorName(res, &errName);  \
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); \
            std::cerr << "Process " << world_rank << " call " << IT << ", " << __func__ << ":" << __LINE__ <<  \
                    " --- Error code: " << res << ", name: " << (errName ? errName : "Unknown") << "\n";  \
        } else { \
            std::string memStr; \
            switch (memType) { \
                case CU_MEMORYTYPE_HOST:   memStr = "HOST"; break; \
                case CU_MEMORYTYPE_DEVICE: memStr = "DEVICE"; break; \
                case CU_MEMORYTYPE_ARRAY:  memStr = "ARRAY"; break; \
                default:                   memStr = "UNKNOWN"; break; \
            } \
            if (memType != CU_MEMORYTYPE_DEVICE) std::cerr << "Pointer is not from device, memory type: " << memStr << "\n"; \
            else std::cout << "Pointer of proc " << world_rank << " call " << IT << ", " << __func__ << ":" << __LINE__ << " is fine\n"; \
        } \
    }  \
}

typedef struct {
    int rowidx;
    int colidx;
    int nodeidx;
    int flag;
} RemoteTile;

typedef struct {
    int rowidx;
    int colidx;
    int nodeidx;
} LocalTile;


template <typename T, typename... Args>
void print_h_arr(T * h_arr, const uint32_t n, const char * prefix, Args... args)
{
    std::cout<<'\n';
    fprintf(stdout, prefix, args...);
    for (uint32_t i=0; i<n; i++)
    {
        std::cout<<h_arr[i]<<',';
    }
    std::cout<<'\n';
    FLUSH_WAIT(500000);
}


template <typename T, typename... Args>
void print_d_arr(T * d_arr, const uint32_t n, const char * prefix, Args... args)
{
    T * h_arr = (T*)d2h_copy(d_arr, n);
    print_h_arr(h_arr, n, prefix, args...);
    free(h_arr);
}


template <typename... Args>
void log_write(FILE * file, const char * msg, Args... args)
{
    fprintf(file, "\n");
    fprintf(file, msg, args...);
    fprintf(file, "\n");
    fflush(file);
}


template <typename T, typename... Args>
void log_h_arr(FILE * file, T * buf, size_t n, const char * prefix, Args... args)
{
    int limit = 100;
    fprintf(file, "\n");
    fprintf(file, prefix, args...);
    fprintf(file, "\n");
    fflush(file);
    for (size_t i=0; i<n; i++)
    {
        if (i < limit || (n-i) < limit)
        {
            if constexpr (std::is_integral<T>::value)
                fprintf(file, "%d, ", buf[i]);
            if constexpr (std::is_floating_point<T>::value)
                fprintf(file, "%f, ", buf[i]);
        }
        else
        {
            fprintf(file, "...");
            i = (n - limit);
        }
    }
    fprintf(file, "\n");
    fflush(file);
}

template <typename T, typename... Args>
void log_h_arr_unlimited(FILE * file, T * buf, size_t n, const char * prefix, Args... args)
{
    fprintf(file, "\n");
    fprintf(file, prefix, args...);
    fprintf(file, "\n");
    for (size_t i=0; i<n; i++)
    {
        if constexpr (std::is_integral<T>::value)
        {
            fprintf(file, "%d, ", buf[i]);
        }
        if constexpr (std::is_floating_point<T>::value)
        {
            fprintf(file, "%f, ", buf[i]);
        }
        if ((i + 1) % 50 == 0)
        {
            fprintf(file, "\n");
        }
    }
    fprintf(file, "\n");
    fflush(file);
}


template <typename T, typename... Args>
void log_d_arr(FILE * file, T * d_buf, size_t n, const char * prefix, Args... args)
{
    T * h_buf = (T*)d2h_copy(d_buf, n);
    log_h_arr<T, Args...>(file, h_buf, n, prefix, args...);
    freeMem(h_buf);
}


template <typename T, typename... Args>
void log_d_arr_unlimited(FILE * file, T * d_buf, size_t n, const char * prefix, Args... args)
{
    T * h_buf = d2h_copy(d_buf, n);
    log_h_arr_unlimited<T, Args...>(file, h_buf, n, prefix, args...);
    freeMem(h_buf);
}

template <typename... Args>
void print_rk0(dmmio::ProcessGrid * grid, const char * msg, Args... args)
{
    if (grid->global_rank==0) 
    {
        fprintf(stdout, "\n");
        fprintf(stdout, msg, args...);
        fprintf(stdout, "\n");
    }
    FLUSH_WAIT(500000);
}


template <typename... Args>
void print_rkn(dmmio::ProcessGrid * grid, const char * msg, Args... args)
{
    print_rkn(grid->global_rank, msg, args...);
}


template <typename... Args>
void print_rkn(int rank, const char * msg, Args... args)
{
    fprintf(stdout, "\n" GREEN "Process %d --- " RESET, rank);
    fprintf(stdout, msg, args...);
    fprintf(stdout, "\n");
    FLUSH_WAIT(500000);
}

template<typename IT>
void move2gpu(IT** ptr, uint64_t size) {
    /*  // NOTE nice to set but it don't work
    CUmemorytype memType;
    cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, reinterpret_cast<CUdeviceptr>(ptr));

    std::string memStr;
    switch (memType) {
        case CU_MEMORYTYPE_HOST:   memStr = "HOST"; break;
        case CU_MEMORYTYPE_DEVICE: memStr = "DEVICE"; break;
        case CU_MEMORYTYPE_ARRAY:  memStr = "ARRAY"; break;
        default:                   memStr = "UNKNOWN"; break;
    }

    if (memType != CU_MEMORYTYPE_HOST) std::cerr << "Error: function " << __func__ << " took in input a non host pointer: " << memStr;
    */

    IT *tmp;
    CUDA_CHECK(cudaMalloc(&tmp, sizeof(IT)*size));
    CUDA_CHECK(cudaMemcpy(tmp, *ptr, sizeof(IT)*size, cudaMemcpyHostToDevice));
    free(*ptr);
    *ptr = tmp;
}

template<typename IT>
void move2host(IT** ptr, uint64_t size) {
    /* // NOTE nice to set but it don't work
    CUmemorytype memType;
    cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, reinterpret_cast<CUdeviceptr>(ptr));

    std::string memStr;
    switch (memType) {
        case CU_MEMORYTYPE_HOST:   memStr = "HOST"; break;
        case CU_MEMORYTYPE_DEVICE: memStr = "DEVICE"; break;
        case CU_MEMORYTYPE_ARRAY:  memStr = "ARRAY"; break;
        default:                   memStr = "UNKNOWN"; break;
    }

    if (memType != CU_MEMORYTYPE_DEVICE) std::cerr << "Error: function " << __func__ << " took in input a non device pointer: " << memStr;
    */

    IT *tmp = (IT*)malloc(sizeof(IT)*size);
    CUDA_CHECK(cudaMemcpy(tmp, *ptr, sizeof(IT)*size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(*ptr));
    *ptr = tmp;
}

template<typename IT, typename VT>
void moveCSX2device (mmio::CSX<IT,VT> *csx) {
    IT ptrdim = (csx->majordim == mmio::MajorDim::ROWS) ? (csx->ncols+1) : (csx->nrows+1) ;

    move2gpu(&(csx->ptr_vec), ptrdim);
    move2gpu(&(csx->idx_vec), csx->nnz);
    move2gpu(&(csx->val), csx->nnz);
}

template<typename IT, typename VT>
void moveCSX2host (mmio::CSX<IT,VT> *csx) {
    IT ptrdim = (csx->majordim == mmio::MajorDim::ROWS) ? (csx->ncols+1) : (csx->nrows+1) ;

    move2host(&(csx->ptr_vec), ptrdim);
    move2host(&(csx->idx_vec), csx->nnz);
    move2host(&(csx->val), csx->nnz);
}

// ---------- Moved from tile_holder.cuh to use them also from spacomm ----------

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

struct cubTmpBuff {
    void*  tmp_buffer   = nullptr;
    size_t current_size = 0;

    // Ensure buffer has at least 'bytes' capacity
    void* ensure(size_t bytes) {
        if (bytes > current_size) {
            if (tmp_buffer!=nullptr) CUDA_CHECK(cudaFree(tmp_buffer));
            CUDA_CHECK(cudaMalloc(&tmp_buffer, 2 * bytes)); // grow with factor
            current_size = 2 * bytes;
        }
        return tmp_buffer;
    }

    void explicitFree(void) {
        if (current_size > 0) {
            CUDA_CHECK(cudaFree(tmp_buffer));
        }
        tmp_buffer = nullptr;
        current_size = 0;
    }

    ~cubTmpBuff() {
        if (current_size > 0) explicitFree();
    }
};

template <typename IT>
void rownnz_to_rowptrs(IT * d_rowptrs, const IT nrows, cudaStream_t stream = 0, cubTmpBuff *tmp_buff = nullptr) {
    void * d_tmp = NULL;
    size_t tmp_size = 0;
    cub::DeviceScan::InclusiveSum(d_tmp, tmp_size, d_rowptrs+1, nrows, stream);
    if (tmp_buff == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_tmp, tmp_size));
    } else {
        d_tmp = tmp_buff->ensure(tmp_size);
    }
    cub::DeviceScan::InclusiveSum(d_tmp, tmp_size, d_rowptrs+1, nrows, stream);
    if (tmp_buff == nullptr) {
        CUDA_CHECK(cudaFree(d_tmp));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename IT>
void rowptrs_to_rownnz(IT * d_rowptrs, const IT nrows, cudaStream_t stream = 0, cubTmpBuff *tmp_buff = nullptr) {
    void * d_tmp = NULL;
    size_t tmp_size = 0;
    cub::DeviceAdjacentDifference::SubtractLeft(d_tmp, tmp_size, d_rowptrs, nrows+1, DiffOp2<IT>{}, stream);
    if (tmp_buff == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_tmp, tmp_size));
    } else {
        d_tmp = tmp_buff->ensure(tmp_size);
    }
    cub::DeviceAdjacentDifference::SubtractLeft(d_tmp, tmp_size, d_rowptrs, nrows+1, DiffOp2<IT>{}, stream);
    if (tmp_buff == nullptr) {
        CUDA_CHECK(cudaFree(d_tmp));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
// ------------------------------------------------------------

template <typename... Args>
void par_print(const char * str, Args... args)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);
    fprintf(stdout, "---Process %d---\n", rank);
    fprintf(stdout, str, args...);
    fprintf(stdout, "----------------\n");
    fflush(stdout);
    sleep(1);
    MPI_Barrier(MPI_COMM_WORLD);
}


template<typename IT, typename VT>
void CSX_destroy_device(mmio::CSX<IT, VT> **csx) {
    if (*csx != NULL) {
        if ((*csx)->ptr_vec != NULL) {
        CUDA_CHECK(cudaFree((*csx)->ptr_vec));
        (*csx)->ptr_vec = NULL;
        }
        if ((*csx)->idx_vec != NULL) {
        CUDA_CHECK(cudaFree((*csx)->idx_vec));
        (*csx)->idx_vec = NULL;
        }
        if ((*csx)->val != NULL) {
        CUDA_CHECK(cudaFree((*csx)->val));
        (*csx)->val = NULL;
        }
        free(*csx);
        *csx = NULL;
    }
}

template<typename IT, typename VT>
struct CsxBuffers {
    uint64_t nnz;
    uint64_t ptr_dim;
    int initialized;
    VT *d_node_vals;
    IT *d_node_colinds;
    IT *d_node_rowptrs;
    cubTmpBuff tmp_buffer;

    CsxBuffers(uint64_t input_nnz, uint64_t input_ptr_dim) {
        initialized = 1;

        nnz = input_nnz;
        ptr_dim = input_ptr_dim;
        CUDA_CHECK(cudaMalloc(&d_node_vals,    sizeof(VT)*nnz));
        CUDA_CHECK(cudaMalloc(&d_node_colinds, sizeof(IT)*nnz));
        CUDA_CHECK(cudaMalloc(&d_node_rowptrs, sizeof(IT)*ptr_dim));
    }

    CsxBuffers(void) {
        initialized    = 0;
        d_node_vals    = nullptr;
        d_node_colinds = nullptr;
        d_node_rowptrs = nullptr;
    }

    void ensure(uint64_t input_nnz, uint64_t input_ptr_dim) {
        if (!initialized) {
            new (this) CsxBuffers(input_nnz, input_ptr_dim);
        } else {
            if (input_nnz > nnz) {
                CUDA_CHECK(cudaFree(d_node_vals));
                CUDA_CHECK(cudaFree(d_node_colinds));

                nnz = input_nnz;
                CUDA_CHECK(cudaMalloc(&d_node_vals,    sizeof(VT)*nnz));
                CUDA_CHECK(cudaMalloc(&d_node_colinds, sizeof(IT)*nnz));
            }
            if (input_ptr_dim > ptr_dim) {
                CUDA_CHECK(cudaFree(d_node_rowptrs));

                ptr_dim = input_ptr_dim;
                CUDA_CHECK(cudaMalloc(&d_node_rowptrs, sizeof(IT)*ptr_dim));
            }
        }
    }

    void ensure_tmp(uint64_t required_size) {
        if (!initialized) {
            new (this) CsxBuffers();
        }
        tmp_buffer.ensure(required_size);
    }

    void explicitFree(void) {
        if (initialized) {
            cudaFree(d_node_colinds);
            cudaFree(d_node_rowptrs);
            cudaFree(d_node_vals);
            initialized = 0;
            ptr_dim = 0;
            nnz = 0;
        }
    }

    ~CsxBuffers() {
        if (initialized) explicitFree();
    }
};

// Threads pool
class ThreadPool {
public:
    explicit ThreadPool(size_t n) : stop(false) {
        for (size_t i = 0; i < n; ++i) {
            workers.emplace_back([this]() {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this]() {
                            return stop || !tasks.empty();
                        });
                        if (stop && tasks.empty())
                            return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template <typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>
    {
        using return_type = typename std::invoke_result<F, Args...>::type;
        auto task_ptr = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop)
                throw std::runtime_error("enqueue on stopped pool");
            tasks.emplace([task_ptr]() { (*task_ptr)(); });
        }

        condition.notify_one();
        return task_ptr->get_future();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (auto &t : workers)
            t.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

class SimpleBarrier {
public:
    explicit SimpleBarrier(std::size_t count) : thread_count(count), arrived(0) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mtx);
        arrived++;
        if (arrived == thread_count) {
            arrived = 0;            // optional, for reuse
            cv.notify_all();
        } else {
            cv.wait(lock, [this] { return arrived == 0; });
        }
    }

private:
    std::mutex mtx;
    std::condition_variable cv;
    std::size_t thread_count;
    std::size_t arrived;
};

inline void mutex_MPI_Testall(MPI_Request *requests, int nrequests, std::mutex& mpi_mutex) {
    int ready = 0;
    while (!ready) {
        std::lock_guard<std::mutex> lock(mpi_mutex);
        MPI_Testall(nrequests, requests, &ready, MPI_STATUS_IGNORE);
    }
}

inline void mutex_MPI_Test(MPI_Request *request, std::mutex& mpi_mutex) {
    mutex_MPI_Testall(request, 1, mpi_mutex);
}


inline void mutex_MPI_Wintest(MPI_Win win, std::mutex& mpi_mutex) {
    int ready = 0;
    while (!ready) {
        std::lock_guard<std::mutex> lock(mpi_mutex);
        MPI_Win_test(win, &ready);
    }
}
