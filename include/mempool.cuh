
#ifndef MEMPOOL_CUH
#define MEMPOOL_CUH
#include "common.h"



// This will reserve a chunk of pool_sz for a pool
inline void setup_mempool(size_t pool_sz, const int devid, cudaStream_t * stream)
{
    cudaMemPool_t pool;

    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, devid));

    CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &pool_sz));

    void * d_tmp;
    CUDA_CHECK(cudaMallocAsync(&d_tmp, pool_sz, *stream));
    CUDA_CHECK(cudaFreeAsync(d_tmp, *stream));

    CUDA_CHECK(cudaDeviceSynchronize());
}


inline void teardown_mempool(const int devid)
{
    cudaMemPool_t pool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, devid));
    CUDA_CHECK(cudaMemPoolTrimTo(pool, 0));
}


#endif
