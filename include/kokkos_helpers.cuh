#pragma once
#include "common.h"

template <typename IT, typename VT>
struct KokkosTypes 
{
    using ExecutionSpace = Kokkos::Cuda;
    using Ordinal = IT;
    using SizeType = IT;
    using Scalar = VT;
    using MemorySpace = typename ExecutionSpace::memory_space;
    using CrsMatrix = KokkosSparse::CrsMatrix<Scalar, Ordinal, ExecutionSpace, void, SizeType>;
};


template <typename IT, typename VT>
struct Triple
{
    IT row;
    IT col;
    VT val;
};


template <typename IT, typename VT>
KokkosTypes<IT, VT>::CrsMatrix * coo_to_kokkos_crs(mmio::COO<IT, VT> * coo)
{
    using Tr = Triple<IT, VT>;
    using KokkosCRS = KokkosTypes<IT, VT>::CrsMatrix;

    // Sort by row
    std::vector<Tr> triples(coo->nnz);
    for (IT i=0; i<coo->nnz; i++)
    {
        triples[i].row = coo->row[i];
        triples[i].col = coo->col[i];
        triples[i].val = coo->val[i];
    }

    std::sort(triples.begin(), triples.end(), 
        [](auto& t1, auto& t2)
        {
            return t1.row <= t2.row;
        }
    );
                
    // First convert the local COO representation to a CSR representation on the host
    std::vector<IT> rowptrs(coo->nrows + 1, 0);
    std::for_each(triples.begin(), triples.end(),
        [&](auto& t)
        {
            rowptrs[t.row+1]++;
        }
    );

    std::vector<IT> colinds(coo->nnz);
    std::transform(triples.begin(), triples.end(),
                   colinds.begin(), 
        [&](auto& t)
        {
            return t.col;
        }
    );

    std::vector<VT> vals(coo->nnz);
    std::transform(triples.begin(), triples.end(),
                   vals.begin(), 
        [&](auto& t)
        {
            return t.val;
        }
    );

    std::inclusive_scan(rowptrs.begin() + 1, rowptrs.end(), rowptrs.begin() + 1);

    // Now, copy buffers to the device
    IT * d_rowptrs = d2h_copy(rowptrs.data(), coo->nrows + 1);
    IT * d_colinds = d2h_copy(colinds.data(), coo->nnz);
    VT * d_vals = d2h_copy(vals.data(), coo->nnz);


    // Now we can just call the kokkos constructor
    KokkosCRS * crs_mat = new KokkosCRS(
            nullptr, 
            coo->nrows, 
            coo->ncols, 
            coo->nnz, 
            Kokkos::View<VT*>(d_vals), 
            Kokkos::View<IT*>(d_rowptrs), 
            Kokkos::View<IT*>(d_colinds)
    );

    return crs_mat;
}


template <typename IT, typename VT>
KokkosTypes<IT, VT>::CrsMatrix * csr_to_kokkos_crs(const IT nrows, const IT ncols, const IT nnz, 
                                                   VT * d_vals, IT * d_colinds, IT * d_rowptrs)
{
    using KokkosCRS = KokkosTypes<IT, VT>::CrsMatrix;
    KokkosCRS * crs_mat = new KokkosCRS(
            nullptr, 
            nrows, 
            ncols, 
            nnz, 
            Kokkos::View<VT*>(d_vals), 
            Kokkos::View<IT*>(d_rowptrs), 
            Kokkos::View<IT*>(d_colinds)
    );

    return crs_mat;

}
