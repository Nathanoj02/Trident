#pragma once
#include "common.h"
#include "message_queue.cuh"
#include "tile_holder.cuh"
#include "cusparse_helpers.cuh"

#include <ccutils/cuda/cuda_timers.h>

#define DEBUG_MAIN 0
#define DETAILED_TIMERS
//#define VERBOSE


template <typename IT, typename VT>
DistCusparseCSX<IT,VT> *  hns_spgemm_main(DistCusparseCSX<IT, VT> * kwd_A, DistCusparseCSX<IT, VT> * kwd_B, const Implementation impl, ThreadPool& pool, SpaComm::SpaCommHandler<IT, VT> *spcomm, bool skipspgemm=false);
