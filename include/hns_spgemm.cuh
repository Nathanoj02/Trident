#pragma once
#include "common.h"
#include "message_queue.cuh"
#include "tile_holder.cuh"
#include "task_queue.cuh"
#include "mempool.cuh"
#include "cusparse_helpers.cuh"

#include <ccutils/cuda/cuda_timers.h>

#include <condition_variable>
#include <mutex>
#include <atomic>
#include <memory>
#include <optional>

#define DEBUG_MAIN 0
#define DETAILED_TIMERS
//#define DEBUG_MEM
//#define VERBOSE
#define KOKKOS

#ifdef KOKKOS
#include "KokkosWrap.hpp"
#endif

#ifdef KOKKOS
using namespace KokkosWrap;
#endif


template <typename IT, typename VT>
struct AccumThreadHandle
{
    using LocalMatrix = LocalMatrix<IT,IT,VT>;

    std::optional<LocalMatrix> C_prod;
    LocalMatrix * C_local;
    CusparseCSX<IT, VT> * C_local_csx; 
    CusparseCSX<IT, VT> * C_accum_csx; 
    cusparseHandle_t * cusparse_handle;

    std::atomic<int> * ready_flag;
    int n_accum_total;
    int dev_id;
    bool empty_matrix;

    static constexpr int IDLE = 1;
    static constexpr int ACTIVE = 2;
    static constexpr int EMPTY = 3;

    AccumThreadHandle(cusparseHandle_t * cusparse_handle, CusparseCSX<IT, VT> * C_local_csx, CusparseCSX<IT, VT> * C_accum_csx, int n_accum_total, int dev_id):
        cusparse_handle(cusparse_handle), C_local_csx(C_local_csx), C_accum_csx(C_accum_csx), n_accum_total(n_accum_total), dev_id(dev_id)
    {
        ready_flag = new std::atomic<int>(IDLE);
        empty_matrix = true;
    }


    AccumThreadHandle(LocalMatrix * C_local, int n_accum_total, int dev_id):
        cusparse_handle(nullptr), C_local(C_local), n_accum_total(n_accum_total), dev_id(dev_id)
    {
        ready_flag = new std::atomic<int>(IDLE);
        empty_matrix = true;
    }


    void spadd(LocalMatrix& C_prod_)
    {
        if (cusparse_handle == nullptr)
        {
            spadd_kokkos(C_prod_);
        }
        else
        {
            spadd_cusparse(C_prod_);
        }
    }


    void spadd_cusparse(LocalMatrix& C_prod_)
    {
        if (C_prod_.storage.nnz() == 0)
        {
            return;
        }

        CusparseCSX<IT, VT> * C_prod_csx = new CusparseCSX<IT, VT>(C_prod_);
        cusparse_spgeam(cusparse_handle, C_prod_csx, C_local_csx, C_accum_csx);
        std::swap(C_local_csx, C_accum_csx);
        free(C_prod_csx->mat);
        delete C_prod_csx;
    }


    void spadd_kokkos(LocalMatrix& C_prod_)
    {
        LocalMatrix::spadd(C_prod_, *C_local);
    }


    void wait_until_idle()
    {
        ready_flag->wait(ACTIVE, std::memory_order_relaxed);
    }


    void start_accum(LocalMatrix&& C_prod_)
    {
        empty_matrix = false;
        C_prod.emplace(C_prod_);
        ready_flag->store(ACTIVE, std::memory_order_release);
        ready_flag->notify_all();
    }


    void signal_empty()
    {
        empty_matrix = true;
        ready_flag->store(ACTIVE, std::memory_order_release);
        ready_flag->notify_all();
    }


    bool is_empty()
    {
        return empty_matrix;
    }

};


template <typename IT, typename VT>
DistCusparseCSX<IT,VT> * hns_spgemm_workstealing(DistCusparseCSX<IT, VT> * kwd_A, DistCusparseCSX<IT, VT> * kwd_B, const Implementation impl, ThreadPool& pool, SpaComm::SpaCommHandler<IT, VT> *spcomm, size_t C_remote_size, bool skipspgemm=false, bool skipws=false);


template <typename IT, typename VT>
DistCusparseCSX<IT,VT> * hns_spgemm_async(DistCusparseCSX<IT, VT> * kwd_A, DistCusparseCSX<IT, VT> * kwd_B, const Implementation impl, ThreadPool& pool, SpaComm::SpaCommHandler<IT, VT> *spcomm,  bool skipspgemm=false);


template <typename IT, typename VT>
DistCusparseCSX<IT,VT> * hns_spgemm_main(DistCusparseCSX<IT, VT> * kwd_A, DistCusparseCSX<IT, VT> * kwd_B, const Implementation impl, ThreadPool& pool, SpaComm::SpaCommHandler<IT, VT> *spcomm, size_t C_remote_size, bool skipspgemm=false, bool skipws=false);
