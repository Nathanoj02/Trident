#pragma once
#include "common.h"
#include "message_queue.cuh"
#include "tile_holder.cuh"
#include "kokkos_helpers.cuh"


template <typename IT, typename VT>
struct DistCSR
{
    using LocalCSR = typename KokkosTypes<IT, VT>::CrsMatrix;
    LocalCSR * csr;
    dmmio::Partitioning * partitioning;
};


template <typename IT, typename VT>
DistCSR<IT, VT> * DistCSR_convert(dmmio::DCOO<IT, VT> * dcoo)
{
    DistCSR<IT, VT> * result = new DistCSR<IT, VT>();
    result->partitioning = dcoo->partitioning;
    result->csr = coo_to_kokkos_crs(dcoo->coo);
    return result;
}




template <typename IT, typename VT>
DistCSR<IT, VT> * hns_spgemm_main(DistCSR<IT, VT> * dist_A, DistCSR<IT, VT> * dist_B);
