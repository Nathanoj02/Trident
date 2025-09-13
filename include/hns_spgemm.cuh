#pragma once
#include "common.h"
#include "message_queue.cuh"
#include "tile_holder.cuh"
#include "kokkos_helpers.cuh"


template <typename IT, typename VT>
struct DistCSR
{
    using LocalCSR = KokkosTypes<IT, VT>::CrsMatrix;
    LocalCSR * csr;
    dmmio::Partitioning * partitioning;
};


template <typename IT, typename VT>
DistCSR<IT, VT> * DistCSR_convert(dmmio::DCOO<IT, VT> * dcoo)
{
    using namespace dmmio::partitioning::indextransform;

    DistCSR<IT, VT> * result = new DistCSR<IT, VT>();
    result->partitioning = dcoo->partitioning;

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
DistCSR<IT, VT> * hns_spgemm_main(DistCSR<IT, VT> * dist_A, DistCSR<IT, VT> * dist_B);
