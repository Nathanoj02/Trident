#ifndef COMMON_H
#define COMMON_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <cstdint>
#include <memory>
#include <iostream>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <numeric>
#include <utility>
#include <thread>

// #define NVTX_PROFILING
#ifdef NVTX_PROFILING
#include <cuda_profiler_api.h>
#define CCUTILS_ENABLE_NVTX 1
#endif

#include <ccutils/timers.h>
#include <ccutils/colors.h>
#include <ccutils/mpi/mpi_macros.h>
#include <ccutils/macros.h>
#include <ccutils/cuda/cuda_macros.h>
#include <ccutils/cuda/cuda_utils.hpp>
#include <ccutils/cuda/cuda_timers.h>

#include <mmio/io.h>
#include <mmio/mmio.h>
#include <mmio/utils.h>
#include <dmmio/dio.h>
#include <dmmio/dmmio.h>
#include <dmmio/dutils.h>
#include <dmmio/partitioning.h>

#include "MPIType.h"


#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cusparse.h>

#include <cub/cub.cuh>

// #define COMP_THRESHOLD 1e8
#define COMP_THRESHOLD 0
#define MPI_ALLGATHERV
//#define NCCL_ALLGATHERV

#define BMASK_TYPE uint8_t
#define MPI_BMASK_TYPE MPI_UINT8_T
#define MASK_SIZE 8   // set to 8 in production

extern FILE * logfile;


// For thread pools
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>

enum Implementation {
    ASYNC,
    WORKSTEALING
};


#include <nccl.h>
#include <type_traits>

template <typename T>
constexpr ncclDataType_t NCCLType() {
    if constexpr (std::is_same_v<T, int8_t>)  return ncclInt8;
    else if constexpr (std::is_same_v<T, uint8_t>) return ncclUint8;
    else if constexpr (std::is_same_v<T, int32_t>) return ncclInt32;
    else if constexpr (std::is_same_v<T, uint32_t>) return ncclUint32;
    else if constexpr (std::is_same_v<T, int64_t>) return ncclInt64;
    else if constexpr (std::is_same_v<T, uint64_t>) return ncclUint64;
    else if constexpr (std::is_same_v<T, float>) return ncclFloat32;
    else if constexpr (std::is_same_v<T, double>) return ncclFloat64;
    else if constexpr (std::is_same_v<T, __half>) return ncclFloat16;

#ifdef __CUDA_BF16_TYPES_EXIST__
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) return ncclBfloat16;
#endif

    else {
        static_assert(!sizeof(T*), "NCCL does not support this datatype.");
    }
}

#include "spacomm.cuh"

#endif
