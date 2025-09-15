#pragma once

// KokkosWrap.hpp
#include <dmmio/dmmio.h>

#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_CcsMatrix.hpp>

namespace KokkosWrap {
    enum class MajorDim {
        ROWS, // i.e. CSR
        COLS  // i.e. CSC
    };

    template <typename KIT, typename DIT, typename VT>
    struct DistribuitedMatrix {
        dmmio::Partitioning* partitioning;  // still raw pointer to external partitioning

        // Instead of a raw union, use std::variant for safety
        std::variant<
            KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT>,
            KokkosSparse::CcsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT>
        > storage;

        MajorDim layout;  // which one we actually hold

        // mmio structure to access the raw pointers
        std::variant<
            mmio::CSR<KIT, VT>,
            mmio::CSC<KIT, VT>
        > dev_mmio;

        // ---- Constructor ----
        DistribuitedMatrix(dmmio::DCOO<DIT, VT>* dcoo, MajorDim T);
    };

    template <typename KIT, typename DIT, typename VT>
    struct LocalMatrix {
        bool initialized;
        KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> storage;

        // ---- Constructor ----
        LocalMatrix();
        LocalMatrix(mmio::CSR<DIT, VT>* mmio_csr);
        LocalMatrix(mmio::CSC<DIT, VT>* mmio_csc);

        // In-place SpGEMM: C = C + A*B
        static void sp_mma(const LocalMatrix& A, const LocalMatrix& B, LocalMatrix& C);
    };

    // These two function are exposed temporary for the test_kokkos C matrix
    template<typename KIT, typename VT>
    mmio::CSR<KIT, VT> rawptr_get(KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> kokkos_csr);

    template<typename KIT, typename VT>
    mmio::CSC<KIT, VT> rawptr_get(KokkosSparse::CcsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> kokkos_csc);

    template<typename KIT, typename VT>
    mmio::CSR<KIT, VT> rawptr_get(LocalMatrix<KIT, KIT, VT> kokkos_csr);
}

// KokkosWrap.cpp
#include <variant>
#include <dmmio/partitioning.h>

#include "../include/KokkosWrap.hpp"

// CHECK IF STILL REQUIRED
#include <cstdint>
#include <memory>
// -----------------------

#include <KokkosSparse_CooMatrix.hpp>
#include <KokkosSparse_coo2crs.hpp>
#include <KokkosSparse_crs2ccs.hpp>
#include <KokkosSparse_ccs2crs.hpp>
#include <KokkosSparse_spadd.hpp>

namespace KokkosWrap {

    template<typename KIT, typename VT>
    mmio::CSR<KIT, VT> rawptr_get(KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> kokkos_csr) {
        auto val_view     = kokkos_csr.values;
        auto rowmap_view  = kokkos_csr.graph.row_map;
        auto entries_view = kokkos_csr.graph.entries;

        mmio::CSR<KIT, VT> csr;
        csr.nnz     = kokkos_csr.nnz();
        csr.nrows   = kokkos_csr.numRows();
        csr.ncols   = kokkos_csr.numCols();
        csr.val     = val_view.data();
        csr.row_ptr = const_cast<int32_t*>(rowmap_view.data());
        csr.col_idx = const_cast<int32_t*>(entries_view.data());

        return(csr);
    }

    template<typename KIT, typename VT>
    mmio::CSC<KIT, VT> rawptr_get(KokkosSparse::CcsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> kokkos_csc) {
        auto val_view     = kokkos_csc.values;
        auto colmap_view  = kokkos_csc.graph.col_map;
        auto entries_view = kokkos_csc.graph.entries;

        mmio::CSC<KIT, VT> csc;
        csc.nnz     = kokkos_csc.nnz();
        csc.nrows   = kokkos_csc.numRows();
        csc.ncols   = kokkos_csc.numCols();
        csc.val     = val_view.data();
        csc.col_ptr = const_cast<int32_t*>(colmap_view.data());
        csc.row_idx = const_cast<int32_t*>(entries_view.data());

        return(csc);
    }

    template<typename KIT, typename VT>
    mmio::CSR<KIT, VT> rawptr_get(LocalMatrix<KIT, KIT, VT> kokkos_csr) {
        return(rawptr_get(kokkos_csr.storage));
    }

    template <typename KIT, typename DIT, typename VT>
    DistribuitedMatrix<KIT,DIT,VT>::DistribuitedMatrix(dmmio::DCOO<DIT, VT>* dcoo, MajorDim T)
            : partitioning(dcoo->partitioning), layout(T)
    {
        int nnz = dcoo->coo->nnz;
        Kokkos::View<KIT*> row_d("row", nnz);
        Kokkos::View<KIT*> col_d("col", nnz);
        Kokkos::View<VT*> val_d("val", nnz);

        auto row_h = Kokkos::create_mirror_view(row_d);
        auto col_h = Kokkos::create_mirror_view(col_d);
        auto val_h = Kokkos::create_mirror_view(val_d);

        for (int i = 0; i < nnz; i++) {
            row_h(i) = static_cast<KIT>(dcoo->coo->row[i]);
            col_h(i) = static_cast<KIT>(dcoo->coo->col[i]);
            val_h(i) = static_cast<VT>(1.0); // TODO: use real values if available
        }

        Kokkos::deep_copy(row_d, row_h);
        Kokkos::deep_copy(col_d, col_h);
        Kokkos::deep_copy(val_d, val_h);

        // --- Step 1: COO → CSR (row-major) ---
        auto csr = KokkosSparse::coo2crs(dcoo->coo->nrows, dcoo->coo->ncols, row_d, col_d, val_d);

        // --- Step 2: Decide layout ---
        if (T == MajorDim::ROWS) {
            storage   = csr; // keep CSR
            dev_mmio  = rawptr_get(csr);
        } else {
            // CSR → CSC (column-major)
            auto csc  = KokkosSparse::crs2ccs(csr);
            storage   = csc; // store CSC
            dev_mmio  = rawptr_get(csc);
        }
    }

    using ordinal_view_t = Kokkos::View<int32_t*, Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using values_view_t  = Kokkos::View<float*,   Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    template <typename KIT, typename DIT, typename VT>
    LocalMatrix<KIT,DIT,VT>::LocalMatrix() : storage(), initialized(false) {}

    template <typename KIT, typename DIT, typename VT>
    LocalMatrix<KIT,DIT,VT>::LocalMatrix(mmio::CSR<DIT, VT>* mmio_csr) : initialized(true)
    {
        // TODO check KIT == DIT or find a way to a static cast
        ordinal_view_t rowmap(mmio_csr->row_ptr, mmio_csr->nrows + 1);
        ordinal_view_t colidx(mmio_csr->col_idx, mmio_csr->nnz);
        values_view_t  values(mmio_csr->val,     mmio_csr->nnz);

        KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> kokkos_csr("kokkos_csr",
                                        mmio_csr->nrows, mmio_csr->ncols, mmio_csr->nnz,
                                        values,
                                        rowmap,
                                        colidx
        );

        storage = kokkos_csr;
    }

    template <typename KIT, typename DIT, typename VT>
    LocalMatrix<KIT,DIT,VT>::LocalMatrix(mmio::CSC<DIT, VT>* mmio_csc) : initialized(true)
    {
        // TODO check KIT == DIT or find a way to a static cast
        ordinal_view_t colmap(mmio_csc->col_ptr, mmio_csc->ncols + 1);
        ordinal_view_t rowidx(mmio_csc->row_idx, mmio_csc->nnz);
        values_view_t  values(mmio_csc->val,     mmio_csc->nnz);

        KokkosSparse::CcsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> kokkos_csc("kokkos_csc",
                                        mmio_csc->nrows, mmio_csc->ncols, mmio_csc->nnz,
                                        values,
                                        colmap,
                                        rowidx
        );

        storage = KokkosSparse::ccs2crs(kokkos_csc);
    }

    template <typename KIT, typename DIT, typename VT>
    void LocalMatrix<KIT,DIT,VT>::sp_mma(const LocalMatrix& A, const LocalMatrix& B, LocalMatrix& C) {

        using csr_matrix_type = typename KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT>;
        csr_matrix_type product = KokkosSparse::spgemm<csr_matrix_type>(A.storage, false, B.storage, false);

        if (C.initialized == false) {
            C.storage = product;
        } else {
            csr_matrix_type accumulator;

            // Create KokkosKernelHandle
            using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
                KokkosKernels::default_size_type, KIT, VT,
                Kokkos::DefaultExecutionSpace,
                typename Kokkos::DefaultExecutionSpace::memory_space,
                typename Kokkos::DefaultExecutionSpace::memory_space>;


            KernelHandle kh;
            kh.create_spadd_handle(false);

            KokkosSparse::spadd_symbolic(&kh, product, C.storage, accumulator);
            KokkosSparse::spadd_numeric(&kh, 1.0, product, 1.0, C.storage, accumulator);
            kh.destroy_spadd_handle();

            C.storage = accumulator;
        }
    }

    template struct DistribuitedMatrix<int32_t, uint32_t, double>;
    template struct DistribuitedMatrix<int32_t, uint32_t, float>;
}

