#pragma once
#include "common.h"
#include "utils.cuh"


#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_CcsMatrix.hpp>
#include <KokkosSparse_spgemm.hpp>
#include <KokkosSparse_spadd.hpp>
#include <KokkosSparse_CooMatrix.hpp>
#include <KokkosSparse_spadd.hpp>

#include "KokkosWrapTriple.hpp"

template <typename IT, typename VT>
struct KokkosTypes 
{
    using ExecutionSpace = Kokkos::Cuda;
    using Ordinal = IT;
    using SizeType = IT;
    using Scalar = VT;
    using MemorySpace = typename ExecutionSpace::memory_space;
    using CrsMatrix = KokkosSparse::CrsMatrix<Scalar, Ordinal, ExecutionSpace, void, SizeType>;
    using CcsMatrix = KokkosSparse::CcsMatrix<Scalar, Ordinal, ExecutionSpace, void, SizeType>;
    using UM = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
};


// NOTE: we should be able to remove this
template <typename IT, typename VT>
typename KokkosTypes<IT, VT>::CrsMatrix csr_to_kokkos_crs(const IT nrows, const IT ncols, const IT nnz,
                                                   VT * d_vals, IT * d_colinds, IT * d_rowptrs)
{
    using KokkosCRS = typename KokkosTypes<IT, VT>::CrsMatrix;
    KokkosCRS crs_mat (
            "A", 
            nrows, 
            ncols, 
            nnz, 
            Kokkos::View<VT*, typename KokkosTypes<IT, VT>::MemorySpace, typename KokkosTypes<IT, VT>::UM>(d_vals, nnz), 
            Kokkos::View<IT*, typename KokkosTypes<IT, VT>::MemorySpace, typename KokkosTypes<IT, VT>::UM>(d_rowptrs, nrows+1), 
            Kokkos::View<IT*, typename KokkosTypes<IT, VT>::MemorySpace, typename KokkosTypes<IT, VT>::UM>(d_colinds, nnz)
    );

    return crs_mat;

}


// NOTE: we should be able to remove this
template <typename IT, typename VT>
typename KokkosTypes<IT, VT>::CrsMatrix coo_to_kokkos_crs(mmio::COO<IT, VT> * coo)
{
    using Tr = KokkosWrap::Triple<IT, VT>;
    using KokkosCRS = typename KokkosTypes<IT, VT>::CrsMatrix;

    // Sort by row
    std::vector<Tr> triples(coo->nnz);
    for (IT i=0; i<coo->nnz; i++)
    {
        triples[i].row = coo->row[i];
        triples[i].col = coo->col[i];
        triples[i].val = coo->val[i];
    }

    //MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Copied\n"));
    //FLUSH_WAIT(1000000);

    std::sort(triples.begin(), triples.end(), 
        [](auto& t1, auto& t2)
        {
            return t1.row < t2.row;
        }
    );

    //MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Sorted\n"));
    //FLUSH_WAIT(1000000);
    //            
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

    //MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Scanned\n"));
    //FLUSH_WAIT(1000000);

    // Now, copy buffers to the device
    IT * d_rowptrs = h2d_copy(rowptrs.data(), coo->nrows + 1);
    IT * d_colinds = h2d_copy(colinds.data(), coo->nnz);
    VT * d_vals = h2d_copy(vals.data(), coo->nnz);

    //MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("converting\n"));
    //FLUSH_WAIT(1000000);
    //CUDA_CHECK(cudaDeviceSynchronize());

    // Convert to a kokkos crs matrix
    return csr_to_kokkos_crs(coo->nrows, coo->ncols, coo->nnz, d_vals, d_colinds, d_rowptrs);
}

template <typename IT, typename VT>
mmio::CSX<IT,VT>* coo_to_row_csx(mmio::COO<IT, VT> * coo)
{
    using Tr = KokkosWrap::Triple<IT, VT>;
    using KokkosCRS = typename KokkosTypes<IT, VT>::CrsMatrix;

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
            return t1.row < t2.row;
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

    IT * d_rowptrs = h2d_copy(rowptrs.data(), coo->nrows + 1);
    IT * d_colinds = h2d_copy(colinds.data(), coo->nnz);
    VT * d_vals = h2d_copy(vals.data(), coo->nnz);

    // Convert to a mmio row_major CSX matrix (i.e. a csr csx)
    return(mmio::CSX_create(coo->nrows, coo->ncols, coo->nnz, mmio::MajorDim::ROWS,
                            d_rowptrs, d_colinds, d_vals));
}

// NOTE: we should be able to remove this
template <typename IT, typename VT>
typename KokkosTypes<IT, VT>::CcsMatrix csc_to_kokkos_ccs(const IT nrows, const IT ncols, const IT nnz,
                                                   VT * d_vals, IT * d_rowinds, IT * d_colptrs)
{
    using KokkosCCS = typename KokkosTypes<IT, VT>::CcsMatrix;
    KokkosCCS ccs_mat (
            "A",
            nrows,
            ncols,
            nnz,
            Kokkos::View<VT*, typename KokkosTypes<IT, VT>::MemorySpace, typename KokkosTypes<IT, VT>::UM>(d_vals, nnz),
            Kokkos::View<IT*, typename KokkosTypes<IT, VT>::MemorySpace, typename KokkosTypes<IT, VT>::UM>(d_colptrs, ncols+1),
            Kokkos::View<IT*, typename KokkosTypes<IT, VT>::MemorySpace, typename KokkosTypes<IT, VT>::UM>(d_rowinds, nnz)
    );

    return ccs_mat;

}

// NOTE: we should be able to remove this
template <typename IT, typename VT>
typename KokkosTypes<IT, VT>::CcsMatrix coo_to_kokkos_ccs(mmio::COO<IT, VT> * coo)
{
    using Tr = KokkosWrap::Triple<IT, VT>;
    using KokkosCRS = typename KokkosTypes<IT, VT>::CrsMatrix;

    // Sort by row
    std::vector<Tr> triples(coo->nnz);
    for (IT i=0; i<coo->nnz; i++)
    {
        triples[i].row = coo->row[i];
        triples[i].col = coo->col[i];
        triples[i].val = coo->val[i];
    }

    //MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Copied\n"));
    //FLUSH_WAIT(1000000);

    std::sort(triples.begin(), triples.end(),
        [](auto& t1, auto& t2)
        {
            return t1.col < t2.col;
        }
    );

    //MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Sorted\n"));
    //FLUSH_WAIT(1000000);
    //
    // First convert the local COO representation to a CSR representation on the host
    std::vector<IT> colptrs(coo->ncols + 1, 0);
    std::for_each(triples.begin(), triples.end(),
        [&](auto& t)
        {
            colptrs[t.col+1]++;
        }
    );

    std::vector<IT> rowinds(coo->nnz);
    std::transform(triples.begin(), triples.end(),
                   rowinds.begin(),
        [&](auto& t)
        {
            return t.row;
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

    std::inclusive_scan(colptrs.begin() + 1, colptrs.end(), colptrs.begin() + 1);

    //MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Scanned\n"));
    //FLUSH_WAIT(1000000);

    // Now, copy buffers to the device
    IT * d_colptrs = h2d_copy(colptrs.data(), coo->ncols + 1);
    IT * d_rowinds = h2d_copy(rowinds.data(), coo->nnz);
    VT * d_vals = h2d_copy(vals.data(), coo->nnz);

    //MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("converting\n"));
    //FLUSH_WAIT(1000000);
    //CUDA_CHECK(cudaDeviceSynchronize());

    // Convert to a kokkos ccs matrix
    return csc_to_kokkos_ccs(coo->nrows, coo->ncols, coo->nnz, d_vals, d_rowinds, d_colptrs);
}

template <typename IT, typename VT>
mmio::CSX<IT,VT>* coo_to_col_csx(mmio::COO<IT, VT> * coo)
{
    using Tr = KokkosWrap::Triple<IT, VT>;
    using KokkosCRS = typename KokkosTypes<IT, VT>::CrsMatrix;

    // Sort by row
    std::vector<Tr> triples(coo->nnz);
    for (IT i=0; i<coo->nnz; i++)
    {
        triples[i].row = coo->row[i];
        triples[i].col = coo->col[i];
        triples[i].val = coo->val[i];
    }

    //MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Copied\n"));
    //FLUSH_WAIT(1000000);

    std::sort(triples.begin(), triples.end(),
        [](auto& t1, auto& t2)
        {
            return t1.col < t2.col;
        }
    );

    //MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Sorted\n"));
    //FLUSH_WAIT(1000000);
    //
    // First convert the local COO representation to a CSR representation on the host
    std::vector<IT> colptrs(coo->ncols + 1, 0);
    std::for_each(triples.begin(), triples.end(),
        [&](auto& t)
        {
            colptrs[t.col+1]++;
        }
    );

    std::vector<IT> rowinds(coo->nnz);
    std::transform(triples.begin(), triples.end(),
                   rowinds.begin(),
        [&](auto& t)
        {
            return t.row;
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

    std::inclusive_scan(colptrs.begin() + 1, colptrs.end(), colptrs.begin() + 1);

    //MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Scanned\n"));
    //FLUSH_WAIT(1000000);

    // Now, copy buffers to the device
    IT * d_colptrs = h2d_copy(colptrs.data(), coo->ncols + 1);
    IT * d_rowinds = h2d_copy(rowinds.data(), coo->nnz);
    VT * d_vals = h2d_copy(vals.data(), coo->nnz);

    //MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("converting\n"));
    //FLUSH_WAIT(1000000);
    //CUDA_CHECK(cudaDeviceSynchronize());

    // Convert to a kokkos ccs matrix
    return(mmio::CSX_create(coo->nrows, coo->ncols, coo->nnz, mmio::MajorDim::COLS,
                            d_colptrs, d_rowinds, d_vals));
}





