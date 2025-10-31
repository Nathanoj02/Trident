#pragma once
#include "common.h"
#include "message_queue.cuh"
#include "tile_holder.cuh"
#include "kokkos_helpers.cuh"
#include "cusparse_helpers.cuh"

#include "KokkosWrap.hpp"
#include <ccutils/cuda/cuda_timers.h>

#define DEBUG_MAIN 0
//#define DETAILED_TIMERS
//#define VERBOSE


template <typename IT, typename VT>
struct DistCSR
{
    using LocalCSR = typename KokkosTypes<IT, VT>::CrsMatrix;
    LocalCSR * csr;
    dmmio::Partitioning * partitioning;

    ~DistCSR()
    {
        delete csr;
    }
};


template <typename IT, typename VT>
DistCSR<IT, VT> * DistCSR_convert(dmmio::DCOO<IT, VT> * dcoo)
{
    using namespace dmmio::partitioning::indextransform;

    DistCSR<IT, VT> * result = new DistCSR<IT, VT>();
    result->partitioning = dcoo->partitioning;

    while (dcoo->coo->nrows % (dcoo->partitioning->grid->col_size * dcoo->partitioning->grid->node_size) != 0)
    {
        dcoo->coo->nrows++;
    }

    while (dcoo->coo->ncols % (dcoo->partitioning->grid->row_size) != 0)
    {
        dcoo->coo->ncols++;
    }

    IT max_dim = max(dcoo->coo->ncols, dcoo->coo->nrows);
    dcoo->coo->ncols = max_dim;
    dcoo->coo->nrows = max_dim;

    //TODO: All this should probably happen in dmmio, right?
    dcoo->coo->nrows /= (dcoo->partitioning->grid->col_size * dcoo->partitioning->grid->node_size);
    dcoo->coo->ncols /= dcoo->partitioning->grid->row_size;

    for (IT i=0; i<dcoo->coo->nnz; i++)
    {
        dcoo->coo->row[i] = global2local::row(dcoo->partitioning, dcoo->coo->row[i]);
        dcoo->coo->col[i] = global2local::col(dcoo->partitioning, dcoo->coo->col[i]);
    }

    result->csr = coo_to_kokkos_crs(dcoo->coo);
    return result;
}



template <typename IT, typename VT>
DistCusparseCSX<IT,VT> *  hns_spgemm_main(DistCusparseCSX<IT, VT> * kwd_A, DistCusparseCSX<IT, VT> * kwd_B, const Implementation impl, ThreadPool& pool, SpaComm::SpaCommHandler<IT, VT> *spcomm, bool skipspgemm=false);
