#pragma once
#include "common.h"
#include "message_queue.cuh"
#include "tile_holder.cuh"
#include "mempool.cuh"
#include "cusparse_helpers.cuh"

#include <ccutils/cuda/cuda_timers.h>


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
DistCusparseCSX<IT,VT> * hns_spgemm_workstealing(DistCusparseCSX<IT, VT> * kwd_A, DistCusparseCSX<IT, VT> * kwd_B, const Implementation impl, ThreadPool& pool, SpaComm::SpaCommHandler<IT, VT> *spcomm,  bool skipspgemm=false);


template <typename IT, typename VT>
DistCusparseCSX<IT,VT> * hns_spgemm_async(DistCusparseCSX<IT, VT> * kwd_A, DistCusparseCSX<IT, VT> * kwd_B, const Implementation impl, ThreadPool& pool, SpaComm::SpaCommHandler<IT, VT> *spcomm,  bool skipspgemm=false);


template <typename IT, typename VT>
DistCusparseCSX<IT,VT> * hns_spgemm_main(DistCusparseCSX<IT, VT> * kwd_A, DistCusparseCSX<IT, VT> * kwd_B, const Implementation impl, ThreadPool& pool, SpaComm::SpaCommHandler<IT, VT> *spcomm,  bool skipspgemm=false);
