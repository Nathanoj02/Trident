#pragma once
#include "common.h"
#include "utils.cuh"

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


template <typename IT, typename VT>
struct Triple
{
    IT row;
    IT col;
    VT val;
};


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



template <typename IT, typename VT>
typename KokkosTypes<IT, VT>::CrsMatrix coo_to_kokkos_crs(mmio::COO<IT, VT> * coo)
{
    using Tr = Triple<IT, VT>;
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
    //FLUSH_WAIT(1.0);

    std::sort(triples.begin(), triples.end(), 
        [](auto& t1, auto& t2)
        {
            return t1.row < t2.row;
        }
    );

    //MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Sorted\n"));
    //FLUSH_WAIT(1.0);
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
    //FLUSH_WAIT(1.0);

    // Now, copy buffers to the device
    IT * d_rowptrs = d2h_copy(rowptrs.data(), coo->nrows + 1);
    IT * d_colinds = d2h_copy(colinds.data(), coo->nnz);
    VT * d_vals = d2h_copy(vals.data(), coo->nnz);

    //MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("converting\n"));
    //FLUSH_WAIT(1.0);
    //CUDA_CHECK(cudaDeviceSynchronize());

    // Convert to a kokkos crs matrix
    return csr_to_kokkos_crs(coo->nrows, coo->ncols, coo->nnz, d_vals, d_colinds, d_rowptrs);
}

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

template <typename IT, typename VT>
typename KokkosTypes<IT, VT>::CcsMatrix coo_to_kokkos_ccs(mmio::COO<IT, VT> * coo)
{
    using Tr = Triple<IT, VT>;
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
    //FLUSH_WAIT(1.0);

    std::sort(triples.begin(), triples.end(),
        [](auto& t1, auto& t2)
        {
            return t1.col < t2.col;
        }
    );

    //MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Sorted\n"));
    //FLUSH_WAIT(1.0);
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
    //FLUSH_WAIT(1.0);

    // Now, copy buffers to the device
    IT * d_colptrs = d2h_copy(colptrs.data(), coo->nrows + 1);
    IT * d_rowinds = d2h_copy(rowinds.data(), coo->nnz);
    VT * d_vals = d2h_copy(vals.data(), coo->nnz);

    //MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("converting\n"));
    //FLUSH_WAIT(1.0);
    //CUDA_CHECK(cudaDeviceSynchronize());

    // Convert to a kokkos ccs matrix
    return csc_to_kokkos_ccs(coo->nrows, coo->ncols, coo->nnz, d_vals, d_rowinds, d_colptrs);
}

template <typename IT, typename VT>
void kokkos_spgemm(typename KokkosTypes<IT, VT>::CrsMatrix& A, typename KokkosTypes<IT, VT>::CrsMatrix& B, typename KokkosTypes<IT, VT>::CrsMatrix& C)
{

    using LocalCSR = typename KokkosTypes<IT, VT>::CrsMatrix ;

    // First, spgemm
    LocalCSR C_new = KokkosSparse::spgemm<LocalCSR, LocalCSR, LocalCSR>(A, false, B, false);

    if (C.numRows() == 0)
    {
        C = std::move(C_new);
        return;
    }

    // Now, accumulate
    // TODO: move this outside?
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle
        <IT, IT, VT, Kokkos::Cuda, Kokkos::CudaSpace, Kokkos::CudaSpace>;
    KernelHandle spadd_handle;
    spadd_handle.create_spadd_handle();

    LocalCSR C_accum;
    KokkosSparse::spadd_symbolic<KernelHandle, LocalCSR, LocalCSR, LocalCSR>(&spadd_handle, C, C_new, C_accum);
    KokkosSparse::spadd_numeric<KernelHandle, VT, LocalCSR, VT, LocalCSR, LocalCSR>(&spadd_handle, VT(1), C, VT(1), C_new, C_accum);

    spadd_handle.destroy_spadd_handle();
    Kokkos::fence();

    C = std::move(C_accum);
}




