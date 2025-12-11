#pragma once
#include "common.h"
#include "tile_holder.cuh"
#include "KokkosWrap.hpp"
#include "cusparse_helpers.cuh"
#include <ccutils/cuda/cuda_timers.h>

template <typename IT, typename VT>
DistCusparseCSX<IT, VT> * sparse_summa(DistCusparseCSX<IT, VT> * A, DistCusparseCSX<IT, VT> * B);





